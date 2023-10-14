# Phase 2: Momentum Lexicon-Contrastive Pretraining

## Pre-Training
```bash
mkdir "/path/to/save/ckpts"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py with data_root="/path/to/data" \
    num_gpus=8 num_nodes=1 task_contrastive_train per_gpu_batchsize=90 \
    beit16_base224 text_bert image_size=224 vit_randaug batch_size=2880 queue_size=11520 DR=True \
    log_dir="/path/to/save/ckpts" precision=16 max_steps=40000 learning_rate=5e-5 \
    load_path= "/path/to/phase_1/model.ckpt"
```

## Fine-Tuning
```bash
mkdir "/path/to/save/ckpts"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py with data_root="/path/to/data" \
    num_gpus=8 num_nodes=1 task_contrastive_train per_gpu_batchsize=24 get_recall_metric=True \
    beit16_base224 text_bert image_size=384 vit_randaug batch_size=1536 queue_size=11520 DR=False \
    log_dir="/path/to/save/ckpts" precision=16 max_epoch=10 learning_rate=5e-5 \
    load_path="/path/to/phase_2/model.ckpt"
```

In the `src/config.py`, you can set which kinds of data you want to use for training.
```python
datasets = ["f30k", "gcc", "sbu", "coco"]
```

In the `src/datasets/base_dataset.py`, you can set how many data you want to use for different datasets:
```python
df = pd.read_csv(f"{self.data_dir}/{input_filename}", sep=sep, nrows=100000)
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python run.py with data_root="/path/to/data" \
    num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True \
    beit16_base224 text_bert image_size=384 queue_size=11520 \
    log_dir="/path/to/save/ckpts" precision=16 \
    load_path="/path/to/Fine_Tuned/model.ckpt"
```
