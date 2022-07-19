#!/bin/bash
gpu="0,1,2,3,4,5,6,7"
NUM_PROC=8
port=12345

IMAGENET_PATH="/public/home/public/imagenet/" # YOUR_PATH
arch_name="burgerformer_tiny" # "burgerformer_small" "burgerformer_base" "burgerformer_large"
log_path=retrain_$arch_name

mkdir -p logs/${log_path}
CUDA_VISIBLE_DEVICES=$gpu \
    nohup python -u -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $port train.py \
        $IMAGENET_PATH \
        --net_config $arch_name \
        --model burgerformer \
        -b 128 \
        --accumulate_grad 1 \
        --lr 1e-3 \
        --epochs 300 \
        --drop-path 0.1 \
        --num-classes 1000 \
        --checkpoint-hist 1 \
        -j 4 \
        --seed 0 \
        --output logs/${log_path} \
        > logs/${log_path}/output.log  2>&1 &

 # drop path rates [0.1, 0.1, 0.2, 0.2] responding to model [BurgerFormer-Tiny, BurgerFormer-Small, BurgerFormer-Base, BurgerFormer-Large]


