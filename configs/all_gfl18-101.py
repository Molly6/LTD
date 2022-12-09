model = dict(
    type='KD_GFL_ALL',
    tea_pretrained="/workspace/code/mmdet2_19/work_dirs/gfl101_50e_nwpu/gfl18-101_50e_nwpu.pth",
    imitation_loss_weigth=1e-3,
    imitation_fuse_loss_weigth=0.5,
    mul_fuse=True,
    tea_model=dict(
        type='GFL',
        pretrained=None,
        backbone=dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=None),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048], # [64, 128, 256, 512]
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        bbox_head=dict(
            type='GFLHead',
            num_classes=10,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128]),
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0),
            loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
            reg_max=16,
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0))),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/workspace/code/mmdet2_19/pretrained_model/resnet18-5c106cde.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='TASDHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.025, T=10),
        loss_ld_vlr=dict(
            type='KnowledgeDistillationKLDivLoss',
            loss_weight=0.025,  # 0.025
            T=10),
        loss_kd=dict(
            type='KnowledgeDistillationKLDivLoss',
            loss_weight=0.5,  # 0.5
            T=2),
        loss_kd_vlr=dict(
            type='KnowledgeDistillationKLDivLoss',
            loss_weight=0.5,  # 0.5
            T=2),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=2000))

# dataset settings
dataset_type = 'NWPUDataset'
data_root = '/workspace/code/mmdet2_19/NWPU/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),

    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[
                data_root + 'ImageSets/Main/train.txt',
            ],
        img_prefix=[data_root],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 80,
    step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)

checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

