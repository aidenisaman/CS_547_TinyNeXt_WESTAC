from torch import nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from models.modules import Stem, Embed, MV2Block, FormerBlock, SeBlock


def gen_block(name, channels, ratio):
    expand_channel = int(ratio * channels)
    if name == "mv2":
        return MV2Block(in_channel=channels, out_channel=channels, stride=1, expand_channel=expand_channel)
    elif name == "former":
        return FormerBlock(dim=channels, mlp_ratio=ratio)
    elif name == "se":
        return SeBlock(in_chs=channels, mlp_ratio=ratio, reduction=4)
    else:
        raise ValueError("invalid block type")


class TinyNeXt(nn.Module):
    def __init__(self, num_classes, cfg=None, distillation=False):
        super(TinyNeXt, self).__init__()
        self.cfg = cfg
        input_channel = self.cfg[0][1]
        self.embeds = nn.ModuleList([Stem(3, input_channel)])
        self.stages = nn.ModuleList([])
        for i in range(4):
            name, width, depth, ratio = self.cfg[i]
            if i > 0:
                self.embeds.append(Embed(input_channel, width))
            stage = nn.Sequential(
                *[gen_block(name, width, ratio) for j in range(depth)]
            )
            self.stages.append(stage)
            input_channel = width
        self.norm = nn.BatchNorm2d(input_channel)
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.class_head = nn.Linear(input_channel, num_classes)
        self.distillation = distillation
        if self.distillation:
            self.dist_head = nn.Linear(input_channel, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.embeds[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.global_pool(x)
        if self.distillation:
            cls_out = self.class_head(x), self.dist_head(x)
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.class_head(x)
        return cls_out


@register_model
def tinynext_t(pretrained=False, num_classes=1000, distillation=False):
    # block width depth ratio
    cfg = [
        ["mv2", 32, 3, 2],
        ["mv2", 64, 3, 2],
        ["former", 96, 6, 2],
        ["se", 192, 2, 2],
    ]
    model = TinyNeXt(num_classes=num_classes, cfg=cfg, distillation=distillation)
    return model


@register_model
def tinynext_s(pretrained=False, num_classes=1000, distillation=False):
    # block width depth ratio
    cfg = [
        ["mv2", 32, 3, 2],
        ["mv2", 64, 3, 2],
        ["former", 96, 8, 2],
        ["se", 192, 3, 2],
    ]
    model = TinyNeXt(num_classes=num_classes, cfg=cfg, distillation=distillation)
    return model


@register_model
def tinynext_m(pretrained=False, num_classes=1000, distillation=False):
    # block width depth ratio
    cfg = [
        ["mv2", 32, 4, 2],
        ["mv2", 64, 4, 2],
        ["former", 128, 9, 2],
        ["se", 256, 4, 1.5],
    ]
    model = TinyNeXt(num_classes=num_classes, cfg=cfg, distillation=distillation)
    return model