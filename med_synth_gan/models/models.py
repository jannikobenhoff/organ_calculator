import random

import torch
import torch.nn as nn


LAYER_LOOKUP = {
    "2d": dict(
        Conv=nn.Conv2d,
        Norm=lambda: nn.InstanceNorm2d(affine=True),# TODO
        Pool=lambda: nn.MaxPool2d(2),
        upsample_mode="bilinear",
    ),
    "3d": dict(
        Conv=nn.Conv3d,
        Norm=lambda: nn.InstanceNorm3d(affine=True),
        Pool=lambda : nn.MaxPool3d(2),
        upsample_mode="trilinear",
    ),
}

class UNet(nn.Module):
    def __init__(
        self,
        dim: str = "2d",
        input_channels: int = 1,
        output_channels: int = 1,
        is_disc: bool = False,
    ):
        super().__init__()
        assert dim in ("2d", "3d"), "dim must be '2d' or '3d'"
        self.dim   = dim
        self.is_disc = is_disc

        Conv  = LAYER_LOOKUP[dim]["Conv"]
        self.up = nn.Upsample(scale_factor=2,
                              mode=LAYER_LOOKUP[dim]["upsample_mode"],
                              align_corners=True)
        # -------- encoder --------
        self.conv0_0 = VGGBlock(input_channels*2,  32, dim=dim)
        self.conv1_0 = VGGBlock(32,  64, stride=2, dim=dim)
        self.conv2_0 = VGGBlock(64, 128, stride=2, dim=dim)
        self.conv3_0 = VGGBlock(128,256, stride=2, dim=dim)
        self.conv4_0 = VGGBlock(256,512, stride=2, dim=dim)
        self.conv5_0 = VGGBlock(512,512, stride=2, dim=dim)
        self.conv6_0 = VGGBlock(512,512, stride=1, dim=dim)

        # -------- decoder --------
        self.conv4_1 = VGGBlock(512+512, 512, dim=dim)
        self.conv3_1 = VGGBlock(256+512, 256, dim=dim)
        self.conv2_1 = VGGBlock(128+256, 128, dim=dim)
        self.conv1_1 = VGGBlock( 64+128,  64, dim=dim)
        self.conv0_1 = VGGBlock( 32+ 64,  32, dim=dim)

        # two-channel scale-and-shift field
        self.final = Conv(32, 2, kernel_size=1, bias=False)

    # -----------------------------------------------------------------
    def forward(self, x, real=None):
        # Same “scale-and-shift” trick you had before
        if real is not None and random.random() > 0.5:
            x = torch.cat([x, real], 1)
        else:
            x = torch.cat([x, x], 1)

        # -------- encoder --------
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x5_0 = self.conv5_0(x4_0)
        x6_0 = self.conv6_0(x5_0)

        # -------- decoder --------
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x6_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_1)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))

        scale_field = self.final(x0_1)
        out = x[:, :1] * scale_field[:, :1] + scale_field[:, 1:]
        return out, scale_field

class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, relu="lrelu", dim="2d"):
        super().__init__()
        assert dim in ("2d", "3d"), "dim must be '2d' or '3d'"
        Conv = LAYER_LOOKUP[dim]["Conv"]
        Norm = LAYER_LOOKUP[dim]["Norm"]

        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu == "lrelu" else nn.ReLU(inplace=True)
        self.conv1 = Conv(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        self.bn1   = Norm(out_ch, affine=True)
        self.conv2 = Conv(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = Norm(out_ch, affine=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, )
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_ch, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_ch, affine=True)

        self.nonlin = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        if stride != 1 or in_ch != out_ch:
            self.skip_down = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=stride)
        else:
            self.skip_down = None

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.nonlin(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.nonlin(out)
        out = self.conv2(out)

        if self.skip_down is not None:
            identity = self.skip_down(x)

        out += identity
        return out