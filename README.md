# LexLIP

Official PyTorch implementation for ICCV23 paper **LexLIP: Lexicon-Bottlenecked Language-Image Pre-Training for Large-Scale Image-Text Sparse Retrieval**.

[[`paper`](https://openaccess.thecvf.com/content/ICCV2023/papers/Luo_LexLIP_Lexicon-Bottlenecked_Language-Image_Pre-Training_for_Large-Scale_Image-Text_Sparse_Retrieval_ICCV_2023_paper.pdf)] [[`appendix`](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Luo_LexLIP_Lexicon-Bottlenecked_Language-Image_ICCV_2023_supplemental.pdf)]

## News :tada: 
- 📣 Sep 2023 - Codes Released.
- 📣 July 2023 - Paper Accepted by ICCV-23.

## LexLIP Training and Inference

[Codes](ChiYeungLaw/LexLIP-ICCV23/Phase1) for Phase 1: Lexicon-Bottlenecked Pre-training

[Codes](ChiYeungLaw/LexLIP-ICCV23/Phase2) for Phase 2: Momentum Lexicon-Contrastive Pretraining

## Pre-Training and Evaluation Data Downloads

You can follow [VILT](https://github.com/dandelin/ViLT/blob/master/DATA.md) to get the datasets (gcc, f30k, coco, and sbu). Then organize the dataset as following structure:
```
F30k
├── f30k_data            
│   ├── xxx.jpg           
│   └── ...          
├── f30k_test.tsv
├── f30k_val.tsv
└── f30k_train.tsv
```
The format of the tsv file should be:
```
title   filepath        image_id
The man with...       f30k_data/1007129816.jpg        25
A man with...       f30k_data/1007129816.jpg        25
...
```

## Citing LexLIP
If you find this repository useful, please consider giving a star :star: and citation:
```
@article{Luo_2023_ICCV,
  title={LexLIP: Lexicon-Bottlenecked Language-Image Pre-Training for Large-Scale Image-Text Sparse Retrieval},
  author={Ziyang Luo and Pu Zhao and Can Xu and Xiubo Geng and Tao Shen and Chongyang Tao and Jing Ma and Qingwei lin and Daxin Jiang},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER/tree/main).
