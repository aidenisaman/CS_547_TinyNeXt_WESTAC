from mmseg.registry import MODELS
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from torch.nn.modules.batchnorm import _BatchNorm


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, identity, fx):
        return identity + fx


class Mul(nn.Module):
    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, act1, act2):
        return act1 * act2


class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, act1, act2):
        return torch.matmul(act1, act2)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )


class Stem(nn.Sequential):
    def __init__(self, inp, oup):
        super(Stem, self).__init__(
            ConvBNReLU(inp, oup // 2, 3, 2),
            ConvBNReLU(oup // 2, oup, 3, 2)
        )


class MV2Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_channel):
        super(MV2Block, self).__init__()
        self.shortcut = stride == 1 and in_channel == out_channel
        self.layers = nn.Sequential(
            ConvBNReLU(in_channel, expand_channel, kernel_size=1),
            ConvBNReLU(expand_channel, expand_channel, stride=stride, groups=expand_channel),
            nn.Conv2d(expand_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.add = Add() if self.shortcut else None

    def forward(self, x):
        if self.shortcut:
            return self.add(x, self.layers(x))
        else:
            return self.layers(x)


class Embed(nn.Sequential):
    def __init__(self, inp, oup):
        super(Embed, self).__init__(MV2Block(inp, oup, 2, oup))


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.scale = dim ** -0.5
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.matmul1 = MatMul()
        self.matmul2 = MatMul()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-1, -2).contiguous()
        attn = self.matmul1(self.linear1(x), x.transpose(-2, -1))
        attn = self.softmax(attn * self.scale)
        x = self.matmul2(attn, self.linear2(x))
        x = x.transpose(-1, -2).view(B, C, H, W).contiguous()
        return x


class Mlp(nn.Sequential):
    def __init__(self, in_channels, ratio):
        super(Mlp, self).__init__(
            nn.Conv2d(in_channels, int(ratio * in_channels), kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(int(ratio * in_channels), in_channels, kernel_size=1, bias=True),
        )


class FormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super(FormerBlock, self).__init__()
        self.attention = nn.Sequential(nn.BatchNorm2d(dim), Attention(dim))
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, 1, groups=dim, bias=False)
        self.mlp = nn.Sequential(nn.BatchNorm2d(dim), Mlp(dim, mlp_ratio))
        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()

    def forward(self, x):
        x = self.add1(x, self.attention(x))
        x = self.add2(x, self.local(x))
        x = self.add3(x, self.mlp(x))
        return x


class SeModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SeModule, self).__init__()
        hidden_channel = max(channel // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, hidden_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, channel, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
        self.mul = Mul()

    def forward(self, x):
        return self.mul(x, self.se(x))


class SeBlock(nn.Module):
    def __init__(self, in_chs, mlp_ratio, reduction=4):
        super(SeBlock, self).__init__()
        self.se = nn.Sequential(nn.BatchNorm2d(in_chs), SeModule(in_chs, reduction))
        self.local = nn.Conv2d(in_chs, in_chs, 3, 1, 1, 1, groups=in_chs, bias=False)
        self.mlp = nn.Sequential(nn.BatchNorm2d(in_chs), Mlp(in_chs, mlp_ratio))
        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()

    def forward(self, x):
        x = self.add1(x, self.se(x))
        x = self.add2(x, self.local(x))
        x = self.add3(x, self.mlp(x))
        return x



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


@MODELS.register_module()
class TinyNeXt(nn.Module):
    def __init__(self, cfg=None, pretrained=None, out_indices=(3, 4), frozen_stages=-1, norm_eval=False, sync_bn=True):
        super(TinyNeXt, self).__init__()
        self.cfg = cfg
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.sync_bn = sync_bn

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

        self._initialize_weights()
        self._sync_bn() if sync_bn else None
        self._freeze_stages()

    def _initialize_weights(self):
        if self.pretrained:
            state_dict = torch.load(self.pretrained, map_location='cpu')['model']
            self_state_dict = self.state_dict()
            for k, v in state_dict.items():
                if k in self_state_dict.keys():
                    self_state_dict.update({k: v})
            self.load_state_dict(self_state_dict, strict=True)
        else:
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

    def _sync_bn(self):
        for i in range(4):
            self.embeds[i] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.embeds[i])
            self.stages[i] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.stages[i])

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            self.embeds[i].eval()
            for param in self.embeds[i].parameters():
                param.requires_grad = False
            self.stages[i].eval()
            for param in self.stages[i].parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer freezed."""
        super(TinyNeXt, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        out = []
        for i in range(4):
            x = self.embeds[i](x)
            x = self.stages[i](x)
            if (i + 1) in self.out_indices:
                out.append(x)
        return tuple(out)

