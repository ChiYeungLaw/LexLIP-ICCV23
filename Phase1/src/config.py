from sacred import Experiment

ex = Experiment("ITR", save_git_info=False)


def _loss_names(d):
    ret = {
        "contrastive": 0,
        "i2t": 0,
        "t2t": 0,
        "self_t": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "ITR"
    seed = 0
    datasets = ["gcc"]
    loss_names = _loss_names({"contrastive": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    temperature = 0.05

    # Image setting
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    image_size = 224
    patch_size = 16

    # Text Setting
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.3

    # Transformer Setting
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'google/vit-base-patch32-224-in21k'
    hidden_size = 768
    num_heads = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    get_recall_metric = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 8
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 32
    training_mode = "bottle"
    # resume_from = ""
    clip_grad = 1.0

@ex.named_config
def task_bottle_pretrain():
    exp_name = "Bottle_Pretrain"
    datasets = ["f30k"]
    loss_names = _loss_names({
        "contrastive": 0,
        "i2t": 1,
        "t2t": 1,
        "self_t": 1,
    })
    batch_size = 256
    temperature = 0.05
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 40
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    learning_rate = 5e-5
    val_check_interval = 1.0
    hidden_size = 768
    num_heads = 12
    training_mode = "bottle"

@ex.named_config
def task_contrastive_train():
    exp_name = "Contrastive_Train"
    datasets = ["coco"]
    loss_names = _loss_names({
        "contrastive": 1,
        "i2t": 0,
        "t2t": 0,
        "self_t": 0,
    })
    batch_size = 256
    temperature = 0.05
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 40
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    learning_rate = 5e-5
    val_check_interval = 1.0
    hidden_size = 768
    num_heads = 12
    training_mode = "con"

@ex.named_config
def task_Text_MAE_Contrastive_train():
    exp_name = "Text_only_MAE_Contrastive"
    datasets = ["f30k"]
    loss_names = _loss_names({
        "contrastive": 1,
        "i2t": 1,
        "t2t": 1,
        "self_t": 1,
    })
    batch_size = 256
    temperature = 0.05
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 40
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    learning_rate = 5e-5
    val_check_interval = 1.0
    hidden_size = 768
    num_heads = 12
    training_mode = "both"

@ex.named_config
def task_IRTR_evaluate():
    exp_name = "IRTR_Evaluate"
    datasets = ["f30k"]
    loss_names = _loss_names({"contrastive": 0})
    whole_word_masking = True
    test_only = True
    get_recall_metric = True

    vocab_size = 30522
    max_text_len = 40
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    val_check_interval = 1.0
    hidden_size = 768
    num_heads = 12
    per_gpu_batchsize = 128
    training_mode = "con"

# visual encoder
@ex.named_config
def beit16_base224():
    vit = "microsoft/beit-base-patch16-224-pt22k-ft22k"
    patch_size = 16
    image_size = 224
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    input_image_embed_size = 768

@ex.named_config
def vit32_base224():
    vit = "google/vit-base-patch32-224-in21k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    input_image_embed_size = 768

@ex.named_config
def vit16_base224():
    vit = "google/vit-base-patch16-224-in21k"
    patch_size = 16
    image_size = 224
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    input_image_embed_size = 768

@ex.named_config
def vit16_base384():
    vit = "google/vit-base-patch16-384"
    patch_size = 16
    image_size = 384
    train_transform_keys = ["vit"]
    val_transform_keys = ["vit"]
    input_image_embed_size = 768

# text encoder
@ex.named_config
def text_bert():
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    input_text_embed_size = 768

# random augmentation
@ex.named_config
def vit_randaug():
    train_transform_keys = ["vit_randaug"]
