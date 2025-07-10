import torch
import torch.nn as nn
from einops import rearrange


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
