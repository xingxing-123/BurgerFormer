# RetinaNet
bash dist_test.sh \
    work_dirs/retinanet_burgerformer_base_fpn_1x_coco/retinanet_burgerformer_base_fpn_1x_coco.py \
    work_dirs/retinanet_burgerformer_base_fpn_1x_coco/latest.pth 8 \
    --out results.pkl \
    --eval bbox
# MaskRCNN
bash dist_test.sh \
    work_dirs/mask_rcnn_burgerformer_base_fpn_1x_coco/mask_rcnn_burgerformer_base_fpn_1x_coco.py \
    work_dirs/mask_rcnn_burgerformer_base_fpn_1x_coco/latest.pth 8 \
    --out results.pkl \
    --eval bbox segm