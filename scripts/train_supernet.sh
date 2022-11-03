#!/bin/bash
gpu="0,1,2,3,4,5,6,7"
NUM_PROC=8
port=12345

IMAGENET_PATH="/IMAGENET_PATH"  # YOUR_PATH, make /IMAGENET_PATH/sub-train & /IMAGENET_PATH/sub-val  exist
log_path=train_supernet

mkdir -p logs/${log_path}
CUDA_VISIBLE_DEVICES=$gpu \
nohup python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $port train_supernet.py \
    $IMAGENET_PATH \
    --train-split sub-train \
    --val-split sub-val \
    --model unifiedarch_s20 \
    --hybrid \
    -b 32 \
    --lr 1e-3 \
    --layer_scale_init_value 1e-6 \
    --weight-decay 0.05 \
    --epochs 120 \
    --clip-grad 5 \
    --apex-amp \
    --num-classes 1000 \
    --seed 0 \
    -j 4 \
    --checkpoint-hist 1 \
    --output logs/${log_path} \
    > logs/${log_path}/output.log  2>&1 &


