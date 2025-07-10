# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
custom_imports = dict(
    imports=['tinynext'],
    allow_failed_imports=False)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(512, 512),
    mean=[128., 128., 128.],
    std=[1., 1., 1.],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='TinyNeXt',
        cfg=[["mv2", 32, 4, 2],
             ["mv2", 64, 4, 2],
             ["former", 128, 9, 2],
             ["se", 256, 4, 1.5]],
        pretrained=None,
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        norm_eval=True,
        sync_bn=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=256,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
