# Searching for BurgerFormer with Micro-Meso-Macro Space Design (ICML 2022)

This is an official pytorch implementation for "[Searching for BurgerFormer with Micro-Meso-Macro Space Design](https://proceedings.mlr.press/v162/yang22f.html)".
![BurgerFormer-img1](img/design.png)
![BurgerFormer-img2](img/space.png)

# Requirements
* PyTorch 1.8.0
* timm 0.4.12

# BurgerFormer Models

Pre-trained checkpoints are released [google drive](https://drive.google.com/drive/folders/1malgIz2qzHUjQH78Dya55TgegD4rAVSm?usp=sharing)/[baiduyun](https://pan.baidu.com/s/1sreW9PclWNHjMAzwqXfsvg). Place them in the .checkpoints/ folder.

Note: access code for `baiduyun` is `gvfl`.

| model | FLOPs/G | Params/M | ImageNet Top1/% |
|  :----: | :----: | :----: | :----: |
| BurgerFormer-tiny | 1.0 | 10 | 78.0 |
| BurgerFormer-small | 2.1 | 14 | 80.4 |
| BurgerFormer-base | 3.9 | 26 | 82.7 |
| BurgerFormer-large | 6.5 | 36 | 83.0 |

# Validation
To evaluate a pre-trained BurgerFormer model on ImageNet, run:
```shell
bash script/test.sh
```

# Train
To retrain a BurgerFormer model on ImageNet, run:
```shell
bash script/train.sh
```

# Search
1. Split ImageNet training dataset to get sub-train & sub-val
```shell
bash script/imagenet_build.sh
```

2. Supernet Training (8 V100 32G)
```shell
bash script/train_supernet.sh
```

3. Evolution Search
```shell
bash script/search_evo.sh
```
Then the searched "myburger" will exist in arch.py

# Citation
Please cite our paper if you find anything helpful.
```
@InProceedings{yang2022burgerformer,
  title={Searching for BurgerFormer with Micro-Meso-Macro Space Design},
  author={Yang, Longxing and Hu, Yu and Lu, Shun and Sun, Zihao and Mei, Jilin and Han, Yinhe and Li, Xiaowei},
  booktitle={ICML},
  year={2022}
}
```

# Acknowledgment
This code is heavily based on [poolformer](https://github.com/sail-sg/poolformer), [ViT-ResNAS](https://github.com/yilunliao/vit-search), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [mmdetection](https://github.com/open-mmlab/mmdetection). Great thanks to their contributions.