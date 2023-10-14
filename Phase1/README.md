# Phase 1: Lexicon-Bottlenecked Pre-training

## Pre-Training
```bash
mkdir "/path/to/save/ckpts"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py with data_root="/path/to/data" \
    num_gpus=8 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=100 \
    beit16_base224 text_bert image_size=224 vit_randaug batch_size=800 \
    log_dir="/path/to/save/ckpts" precision=16 max_epoch=20 learning_rate=5e-5
```

In the `src/config.py`, you can set which kinds of data you want to use for training.
```python
datasets = ["f30k", "gcc", "sbu", "coco"]
```

In the `src/datasets/base_dataset.py`, you can set how many data you want to use for different datasets:
```python
df = pd.read_csv(f"{self.data_dir}/{input_filename}", sep=sep, nrows=100000)
```