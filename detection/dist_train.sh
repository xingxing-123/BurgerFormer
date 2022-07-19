#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=66663 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=66663 \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} 2>&1 &
