_base_ = ['_base_/models/retinanet_r50_fpn.py', '_base_/datasets/coco_detection.py', '_base_/schedules/schedule_1x.py', '_base_/default_runtime.py']

# optimizer
model = dict(
    # pretrained='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar', # for old version of mmdetection
    backbone=dict(
        type='burgerformer_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=\
                '../checkpoints/burgerformer_base.pth.tar',
            ),

        ),
    neck=dict(
        type='FPN',
        # in_channels=[64, 128, 320, 512],
        in_channels=[96, 128, 320, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)
# resume_from = '/home/sunzihao/PoolFormer/detection/work_dirs/retinanet_poolformer_s12_fpn_1x_coco/latest.pth'
