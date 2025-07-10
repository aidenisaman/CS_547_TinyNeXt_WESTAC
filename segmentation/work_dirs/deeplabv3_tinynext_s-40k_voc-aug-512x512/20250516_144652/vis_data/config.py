crop_size = (
    512,
    512,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'tinynext',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        128.0,
        128.0,
        128.0,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        1.0,
        1.0,
        1.0,
    ],
    type='SegDataPreProcessor')
data_root = '/home/lrh/data/VOCdevkit/VOC2012'
dataset_aug = dict(
    ann_file='ImageSets/Segmentation/aug.txt',
    data_prefix=dict(
        img_path='JPEGImages', seg_map_path='SegmentationClassAug'),
    data_root='/home/lrh/data/VOCdevkit/VOC2012',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            keep_ratio=True,
            ratio_range=(
                0.5,
                2.0,
            ),
            scale=(
                2048,
                512,
            ),
            type='RandomResize'),
        dict(cat_max_ratio=0.75, crop_size=(
            512,
            512,
        ), type='RandomCrop'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PhotoMetricDistortion'),
        dict(size=(
            512,
            512,
        ), type='Pad'),
        dict(type='PackSegInputs'),
    ],
    type='PascalVOCDataset')
dataset_train = dict(
    ann_file='ImageSets/Segmentation/train.txt',
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
    data_root='/home/lrh/data/VOCdevkit/VOC2012',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            keep_ratio=True,
            ratio_range=(
                0.5,
                2.0,
            ),
            scale=(
                2048,
                512,
            ),
            type='RandomResize'),
        dict(cat_max_ratio=0.75, crop_size=(
            512,
            512,
        ), type='RandomCrop'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PhotoMetricDistortion'),
        dict(size=(
            512,
            512,
        ), type='Pad'),
        dict(type='PackSegInputs'),
    ],
    type='PascalVOCDataset')
dataset_type = 'PascalVOCDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=96,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=21,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        cfg=[
            [
                'mv2',
                32,
                3,
                2,
            ],
            [
                'mv2',
                64,
                3,
                2,
            ],
            [
                'former',
                96,
                8,
                2,
            ],
            [
                'se',
                192,
                3,
                2,
            ],
        ],
        frozen_stages=-1,
        norm_eval=True,
        out_indices=(
            1,
            2,
            3,
            4,
        ),
        pretrained='checkpoint/tinynext_s.pth',
        sync_bn=True,
        type='TinyNeXt'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            128.0,
            128.0,
            128.0,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=192,
        in_index=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=21,
        type='ASPPHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.00012, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), lr=0.00012, type='AdamW', weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0.0,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root='/home/lrh/data/VOCdevkit/VOC2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalVOCDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        datasets=[
            dict(
                ann_file='ImageSets/Segmentation/train.txt',
                data_prefix=dict(
                    img_path='JPEGImages', seg_map_path='SegmentationClass'),
                data_root='/home/lrh/data/VOCdevkit/VOC2012',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            2048,
                            512,
                        ),
                        type='RandomResize'),
                    dict(
                        cat_max_ratio=0.75,
                        crop_size=(
                            512,
                            512,
                        ),
                        type='RandomCrop'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PhotoMetricDistortion'),
                    dict(size=(
                        512,
                        512,
                    ), type='Pad'),
                    dict(type='PackSegInputs'),
                ],
                type='PascalVOCDataset'),
            dict(
                ann_file='ImageSets/Segmentation/aug.txt',
                data_prefix=dict(
                    img_path='JPEGImages',
                    seg_map_path='SegmentationClassAug'),
                data_root='/home/lrh/data/VOCdevkit/VOC2012',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            2048,
                            512,
                        ),
                        type='RandomResize'),
                    dict(
                        cat_max_ratio=0.75,
                        crop_size=(
                            512,
                            512,
                        ),
                        type='RandomCrop'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PhotoMetricDistortion'),
                    dict(size=(
                        512,
                        512,
                    ), type='Pad'),
                    dict(type='PackSegInputs'),
                ],
                type='PascalVOCDataset'),
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(size=(
        512,
        512,
    ), type='Pad'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root='/home/lrh/data/VOCdevkit/VOC2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalVOCDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/deeplabv3_tinynext_s-40k_voc-aug-512x512'
