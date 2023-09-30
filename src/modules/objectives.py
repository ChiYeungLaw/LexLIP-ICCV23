import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from .dist_utils import all_gather

SMALL_NUM = np.log(1e-45)

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def compute_mlm(pl_module, ret, mode):
    mlm_logits = ret[f"{mode}_logits"]
    if "self" in mode:
        mlm_labels = ret["encoder_text_labels_mlm"]
    else:
        mlm_labels = ret["decoder_text_labels_mlm"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    new_ret = {
        f"{mode}_mlm_loss": mlm_loss
    }

    phase = "train" if pl_module.training else "val"
    loss_mlm = getattr(pl_module, f"{phase}_{mode}_loss")(mlm_loss)
    pl_module.log(f"{mode}/{phase}/{mode}_loss", loss_mlm)
    return new_ret


def FLOAP(batch_rep):
    return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

def compute_contrastive(pl_module, ret):
    # Query
    if pl_module.training_mode == "both":
        text_reps = F.normalize(ret["text_bottleneck_repre"][1])
        image_reps = F.normalize(ret["image_bottleneck_repre"][1])
    else:
        text_reps = F.normalize(ret["text_bottleneck_repre"])
        image_reps = F.normalize(ret["image_bottleneck_repre"])
    
    FLOAP_loss = 0.002 * (FLOAP(text_reps) + FLOAP(image_reps))
    
    all_text_reps = pl_module.gather(text_reps)
    all_image_reps = pl_module.gather(image_reps)

    # in-batch contrastive
    # Cross Entropy
    logits_per_text = torch.einsum("nc,ck->nk", [all_text_reps, all_image_reps.transpose(-2, -1)]) / pl_module.T
    contrastive_loss = clip_loss(logits_per_text)

    new_ret = {
        "contrastive_loss": contrastive_loss + FLOAP_loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_contrastive_loss")(new_ret["contrastive_loss"])
    pl_module.log(f"contrastive/{phase}/loss", loss)

    return new_ret

@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_ids = _b["text_ids"].to(pl_module.device)
        text_masks = _b["text_masks"].to(pl_module.device)
        text_preload.append(
            {
                "img_index": _b["img_index"],
                "text_reps": pl_module.encode_text(
                text_ids, text_masks)[1]
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)
    
    image_preload = dict()
    image_preload_reps = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        img_index = _b["img_index"][0]
        if img_index not in image_preload:
            image_features = _b["image_features"].to(pl_module.device)
            img_reps = pl_module.encode_image(image_features) # [bsz, 768]
            image_preload[img_index] = 1
            image_preload_reps.append((img_reps, _b["img_index"]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload_reps, desc="rank loop"):
        _img_reps, _iid = img_batch # [bsz, 768]
        _img_reps = _img_reps / torch.norm(_img_reps, dim=-1, keepdim=True)

        img_batch_score = list()
        for txt_batch in text_preload:
            _text_reps = txt_batch["text_reps"] # [bsz, 768]
            _text_reps = _text_reps / torch.norm(_text_reps, dim=-1, keepdim=True)
            with torch.cuda.amp.autocast():
                score = torch.einsum('nc,cm->nm', [_img_reps, _text_reps.transpose(-1, -2)])
            img_batch_score.append(score)
        img_batch_score = torch.cat(img_batch_score, dim=-1) # [bsz, num_texts]
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids += _iid
    
    ###
    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    ################################
    tmp = []
    for rank_iids in gather_rank_iids:
        tmp += rank_iids
    gather_rank_iids = tmp

    tmp = []
    for rank_scores in gather_rank_scores:
        tmp += rank_scores
    gather_rank_scores = tmp
    ###############################

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    ###

    topk5 = scores.topk(5, dim=0)
    topk5_iids = iids[topk5.indices] # [5, 25010]

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices] # [5000, 10]
    topk5_iids = tiids[topk5.indices] # [5000, 5]
    topk1_iids = tiids[topk1.indices] # [5000, 1]


    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices] # [10, 25010]
    topk5_iids = iids[topk5.indices] # [5, 25010]
    topk1_iids = iids[topk1.indices] # [1, 25010]
    # tiids [25010]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()