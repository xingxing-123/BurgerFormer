gpu="0"
arch="burgerformer_tiny" # "burgerformer_small" "burgerformer_base" "burgerformer_large"
IMAGENET_PATH="/public/home/public/imagenet/" # YOUR ImageNet Path
resume="checkpoints/burgerformer_tiny.pth.tar"

CUDA_VISIBLE_DEVICES=$gpu \
    python test.py \
        --data-dir $IMAGENET_PATH \
        --model burgerformer \
        -b 500 \
        --resume ${resume} \
        --net_config $arch \
        --num-classes 1000 \
        --seed 0