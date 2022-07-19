# RetinaNet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    nohup bash dist_train.sh configs/retinanet_burgerformer_base_fpn_1x_coco.py 8 \
    > retinanet.log
# MaskRCNN
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     nohup bash dist_train.sh configs/mask_rcnn_poolformer_s12_fpn_1x_coco.py 8 \
#     > maskrcnn.log