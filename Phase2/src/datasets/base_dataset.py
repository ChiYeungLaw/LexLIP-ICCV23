import random
import torch
import io
import pandas as pd
import os

from PIL import Image, ImageFile
from ..transforms import keys_to_transforms

from transformers import ViTFeatureExtractor
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CsvDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        input_filename,
        transform_keys,
        image_size,
        patch_size,
        img_key,
        caption_key,
        img_id_key=None,
        label_key=None,
        sep="\t",
        tokenizer=None,
        dataset_name=None,
        image_feature_extractor=None,
        max_text_len=40,
        image_only=False,
    ):
        assert len(transform_keys) >= 1
        super().__init__()
        self.data_dir = f"{data_dir}/{dataset_name}"
        self.image_only = image_only
        self.input_filename = input_filename
        if input_filename is not None:
            df = pd.read_csv(f"{self.data_dir}/{input_filename}", sep=sep)
            self.images = df[img_key].tolist()
        else:
            self.images = None
        self.captions, self.img_ids = None, None
        if not image_only and input_filename is not None:
            self.captions = df[caption_key].tolist()
        if img_id_key and input_filename is not None:
            self.img_ids = df[img_id_key].tolist()
        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.transforms_small = keys_to_transforms(transform_keys, size=image_size//2)
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.patch_size = patch_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.images) if self.images else 0
    
    def get_image(self, idx):
        image_features = self.transforms[0](Image.open(f"{self.data_dir}/{str(self.images[idx])}")).unsqueeze(0)
        image_features_small = self.transforms_small[0](Image.open(f"{self.data_dir}/{str(self.images[idx])}")).unsqueeze(0)
        num_patches = (self.image_size // self.patch_size) ** 2

        return {
            "image_features": image_features, # [1, 3, H, W]
            "image_features_small": image_features_small, # [1, 3, H, W]
            "raw_index": idx,
            "img_index": int(self.img_ids[idx]) if self.img_ids else -100,
            "img_dirs": f"{self.data_dir}/{str(self.images[idx])}",
        }

    def get_text(self, idx):
        text = str(self.captions[idx]).lower()
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "raw_index": idx,
            "img_index": int(self.img_ids[idx]) if self.img_ids else -100,
        }

    def get_suite(self, idx):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(idx))
                if not self.image_only:
                    ret.update(self.get_text(idx))
                result = True
            except Exception as e:
                print(f"Error while read file idx {idx} in {self.images[idx]} -> {e}")
                print(str(self.captions[idx]).lower())
                idx = random.randint(0, len(self.images) - 1)

        return ret

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        batch_image_features = torch.cat(dict_batch["image_features"], dim=0) # [bs, 3, H, W]
        batch_image_features_small = torch.cat(dict_batch["image_features_small"], dim=0) # [bs, 3, H, W]
        dict_batch["image_features"] = batch_image_features
        dict_batch["image_features_small"] = batch_image_features_small

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]

            # Prepare for text encoder
            mlm_collator.mlm_probability = 0.3
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_masks"] = attention_mask
                dict_batch[f"encoder_{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"encoder_{txt_key}_labels_mlm"] = mlm_labels
            
            # Prepare for text decoder
            mlm_collator.mlm_probability = 0.5
            flatten_mlms = mlm_collator(flatten_encodings)
            for i, txt_key in enumerate(txt_keys):
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                dict_batch[f"decoder_{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"decoder_{txt_key}_labels_mlm"] = mlm_labels

        return dict_batch