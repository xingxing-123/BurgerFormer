#!/bin/bash
gpu="0,1,2,3,4,5,6,7"
NUM_PROC=8
port=12345

IMAGENET_PATH="/public/home/public/imagenet/"  # YOUR_PATH, make /IMAGENET_PATH/sub-val exist
log_path=search_evo

target=flops
flops=1.0e9
params=-1
resume="YOUR_SUPERNET_BEST_CHECKPOINT" # supernet checkpoint

mkdir -p logs/$log_path
CUDA_VISIBLE_DEVICES=$gpu \
nohup python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port $port search_evo.py \
    --data-dir $IMAGENET_PATH \
    --model unifiedarch_s20 \
    -b 500 \
    --val-split sub-val \
    --resume ${resume} \
    --num-classes 1000 \
    --target ${target} \
    --target_flops ${flops} \
    --target_params ${params} \
    --seed 0 \
    --output logs/${log_path} \
    > logs/${log_path}/output.log 2>&1 &

