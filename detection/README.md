Object Detection and Instance Segmentation

# Environement Setup

Install [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0) from souce cocde

# Models on COCO

Pre-trained checkpoints are released [google drive](https://drive.google.com/drive/folders/1malgIz2qzHUjQH78Dya55TgegD4rAVSm?usp=sharing)/[baiduyun](https://pan.baidu.com/s/1sreW9PclWNHjMAzwqXfsvg). Place them in the .detectction/work_dirs/ folder.

Note: access code for `baiduyun` is `gvfl`.

# Evaluation
To evaluate BurgerFormer-Base + RetinaNet and BurgerFormer-Base + Mask R-CNN on COCO val2017, run :
```
cd detection
bash test.sh
```

# Training
To evaluate BurgerFormer-Base + RetinaNet or BurgerFormer-Base + Mask R-CNN on COCO val2017, run :
```
cd detection
bash train.sh
```