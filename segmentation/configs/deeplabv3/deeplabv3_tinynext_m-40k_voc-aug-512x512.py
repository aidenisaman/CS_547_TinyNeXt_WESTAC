_base_ = [
    '../_base_/models/deeplabv3_tinynext_default.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    backbone=dict(
        type='TinyNeXt',
        cfg=[["mv2", 32, 4, 2],
             ["mv2", 64, 4, 2],
             ["former", 128, 9, 2],
             ["se", 256, 4, 1.5]],
        pretrained="checkpoint/tinynext_m.pth",
    ),
    decode_head=dict(in_channels=256, channels=256, num_classes=21),
    auxiliary_head=dict(in_channels=128, num_classes=21),
    )

# data
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
)
test_dataloader = val_dataloader