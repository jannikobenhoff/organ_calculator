import random

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, is_disc = False): # changed this to 2
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        nb_filter = [32, 64, 128, 256, 512, 512]
        self.nb_filter = nb_filter
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(2, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0],  nb_filter[1],stride=2)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2],stride=2)
        self.conv3_0 = VGGBlock(nb_filter[2],  nb_filter[3],stride=2)
        self.conv4_0 = VGGBlock(nb_filter[3],  nb_filter[4],stride=2)
        self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[5],stride=2)
        self.conv6_0 = VGGBlock(nb_filter[5], nb_filter[5],stride=1)

        self.conv4_1 = VGGBlock(nb_filter[4]+nb_filter[5],  nb_filter[4])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4],  nb_filter[3])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2],  nb_filter[1])
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1],  nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 2, kernel_size=1,bias=False)
        self.is_disc = is_disc
        # torch.nn.init.normal_(self.final.weight, mean=0.0, std=1e-2)
        #torch.nn.init.constant_(self.final.bias[0], 0.0)  # scale channel -> 0 => +1 = 1
        #torch.nn.init.constant_(self.final.bias[1], 0.0)  # offset channel -> 0

    def forward(self, input, real=None):
        if real is not None and random.random() > 0.5:
            input = torch.cat([input, real], 1)
        else:
            input = torch.cat([input, input], 1)

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x5_0 = self.conv5_0(x4_0)
        x6_0 = self.conv6_0(x5_0)

        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x6_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_1)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))
        scale_field = self.final(x0_1)
        output = input[:,:1]*scale_field[:,:1] +scale_field[:,1:]
        return output, scale_field

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, relu='lrelu', dropout_prob=0.0):
        super().__init__()
        if relu=='lrelu':
            self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels,affine=True)
        #self.dropout1 = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels,affine=True)
        #self.dropout2 = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #out = self.dropout2(out)

        return out

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