import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Reduce
from timm.models import register_model
from timm.models.layers import trunc_normal_


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        if self.heads == 1:
            qkv = nn.functional.gelu(qkv)
        qkv = qkv.chunk(3, dim=-1)

        if self.heads > 1:
            q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        else:
            q, k, v = qkv

        dots = (q @ k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = attn @ v
        if self.heads > 1:
            out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


class SeModule(nn.Module):
    def __init__(self, dim, squeeze_dim):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, squeeze_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(squeeze_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_dim, dim, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class SeBlock(nn.Module):
    def __init__(self, dim, depth, exp_dim, squeeze_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.Sequential(
                # pw
                nn.Conv2d(dim, exp_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(exp_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(exp_dim, exp_dim, 3, 1, 1, groups=exp_dim, bias=False),
                nn.BatchNorm2d(exp_dim),
                nn.SiLU(),
                SeModule(exp_dim, squeeze_dim),
                # pw-linear
                nn.Conv2d(exp_dim, dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(dim),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class SeStage(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, exp_dim, squeeze_dim):
        super().__init__()
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.se = SeBlock(dim, depth, exp_dim, squeeze_dim)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # SE
        x = self.se(x)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, num_heads, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, num_heads, dim // num_heads, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(self, num_classes, stem_channels=None, stage_cfg=None, mv2_exp_mult=4):
        super().__init__()
        patch_size = (2, 2)
        self.stem = nn.Sequential(
            conv_nxn_bn(3, stem_channels[0], 3, 2),
            MV2Block(stem_channels[0], stem_channels[1], 1, mv2_exp_mult),
            MV2Block(stem_channels[1], stem_channels[2], 2, mv2_exp_mult),
            MV2Block(stem_channels[2], stem_channels[2], 1, mv2_exp_mult),
            MV2Block(stem_channels[2], stem_channels[2], 1, mv2_exp_mult))
        in_channel = stem_channels[-1]
        self.stages = nn.ModuleList([])
        for settings in stage_cfg:
            block_type = settings[0]
            stage = []
            if block_type == "mvit":
                depth, out_channel, dim, mlp_dim, num_heads = settings[1:]
                stage.append(MV2Block(in_channel, out_channel, stride=2, expansion=mv2_exp_mult))
                stage.append(MobileViTBlock(dim, depth, out_channel, 3, patch_size, mlp_dim, num_heads))
                in_channel = out_channel
            elif block_type == "se":
                depth, out_channel, dim, exp_dim, squeeze_dim = settings[1:]
                stage.append(MV2Block(in_channel, out_channel, stride=2, expansion=mv2_exp_mult))
                stage.append(SeStage(dim, depth, out_channel, 3, exp_dim, squeeze_dim))
                in_channel = out_channel
            self.stages.append(nn.Sequential(*stage))
        self.classifier = nn.Sequential(
            conv_1x1_bn(in_channel, in_channel * 4),
            Reduce('b c h w-> b c', 'mean'),
            nn.Linear(in_channel * 4, num_classes, bias=True)
        )
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
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.classifier(x)
        return x


@register_model
def mobilevit_xxs(pretrained=False, num_classes=1000, distillation=False):
    return MobileViT(
        num_classes=num_classes,
        stem_channels=[16, 16, 24],
        stage_cfg=[
            # type depth out_chn dim mlp_dim num_head
            ["mvit", 2, 48, 64, 128, 4],
            ["mvit", 4, 64, 80, 160, 4],
            ["mvit", 3, 80, 96, 192, 4],
        ],
        mv2_exp_mult=2
    )


@register_model
def mobilevit_shsas0_xxs(pretrained=False, num_classes=1000, distillation=False):
    return MobileViT(
        num_classes=num_classes,
        stem_channels=[16, 16, 24],
        stage_cfg=[
            # type depth out_chn dim mlp_dim num_head
            ["mvit", 2, 48, 64, 128, 1],
            ["mvit", 4, 64, 80, 160, 4],
            ["mvit", 3, 80, 96, 192, 4],
        ],
        mv2_exp_mult=2
    )


@register_model
def mobilevit_se_xxs(pretrained=False, num_classes=1000, distillation=False):
    return MobileViT(
        num_classes=num_classes,
        stem_channels=[16, 16, 24],
        stage_cfg=[
            # type depth out_chn dim mlp_dim num_head
            ["mvit", 2, 48, 64, 128, 4],
            ["mvit", 4, 64, 80, 160, 4],
            # type depth out_chn dim exp_dim squeeze_dim
            ["se", 3, 80, 96, 192, 96],
        ],
        mv2_exp_mult=2
    )


@register_model
def mobilevit_xs(pretrained=False, num_classes=1000, distillation=False):
    return MobileViT(
        num_classes=num_classes,
        stem_channels=[16, 32, 48],
        stage_cfg=[
            # type depth out_chn dim mlp_dim num_head
            ["mvit", 2, 64, 96, 192, 4],
            ["mvit", 4, 80, 120, 240, 4],
            ["mvit", 3, 96, 144, 288, 4],
        ],
        mv2_exp_mult=4
    )


@register_model
def mobilevit_se_xs(pretrained=False, num_classes=1000, distillation=False):
    return MobileViT(
        num_classes=num_classes,
        stem_channels=[16, 32, 48],
        stage_cfg=[
            # type depth out_chn dim mlp_dim num_head
            ["mvit", 2, 64, 96, 192, 4],
            ["mvit", 4, 80, 120, 240, 4],
            # type depth out_chn dim exp_dim squeeze_dim
            ["se", 3, 96, 144, 288, 144],
        ],
        mv2_exp_mult=4
    )


@register_model
def mobilevit_s(pretrained=False, num_classes=1000, distillation=False):
    return MobileViT(
        num_classes=num_classes,
        stem_channels=[16, 32, 64],
        stage_cfg=[
            # type depth out_chn dim mlp_dim num_head
            ["mvit", 2, 96, 144, 288, 4],
            ["mvit", 4, 128, 192, 384, 4],
            ["mvit", 3, 160, 240, 480, 4],
        ],
        mv2_exp_mult=4
    )


@register_model
def mobilevit_se_s(pretrained=False, num_classes=1000, distillation=False):
    return MobileViT(
        num_classes=num_classes,
        stem_channels=[16, 32, 64],
        stage_cfg=[
            # type depth out_chn dim mlp_dim num_head
            ["mvit", 2, 96, 144, 288, 4],
            ["mvit", 4, 128, 192, 384, 4],
            # type depth out_chn dim exp_dim squeeze_dim
            ["se", 3, 160, 240, 480, 240],
        ],
        mv2_exp_mult=4
    )
