import numpy as np
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        nb_filter = [32, 64, 128, 256, 512, 1024]
        self.nb_filter = nb_filter
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],stride=2)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],stride=2)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],stride=2)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],stride=2)
        self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[5], nb_filter[5],stride=2)

        self.conv4_1 = VGGBlock(nb_filter[4]+nb_filter[5], nb_filter[4], nb_filter[4])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)

    def forward(self, input):
        # Add shape checking
        assert input.size(1) == self.input_channels, f"Expected {self.input_channels} channels but got {input.size(1)}"

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x5_0 = self.conv5_0(x4_0)

        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_1)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))
 
        scaler_field = self.final(x0_1)

        return scaler_field * input, scaler_field


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, stride = 1, relu='lrelu'):
        super().__init__()
        if relu=='lrelu':
            self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.01)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1,stride=stride)
        self.bn1 = nn.InstanceNorm2d(middle_channels,affine=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels,affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    

# class ResnetBlock(nn.Module):
#     def __init__(self, dim):
#         super(ResnetBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(dim, dim, kernel_size=3),
#             nn.InstanceNorm2d(dim),
#             nn.ReLU(True),

#             nn.ReflectionPad2d(1),
#             nn.Conv2d(dim, dim, kernel_size=3),
#             nn.InstanceNorm2d(dim),
#         )

#     def forward(self, x):
#         return x + self.block(x)
    
# class GeneratorResNet(nn.Module):
#     def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=9):
#         super(GeneratorResNet, self).__init__()
        
#         # Initial convolution block
#         model = [
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(input_nc, 64, 7),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True)
#         ]
        
#         # Downsampling
#         curr_dim = 64
#         for _ in range(2):
#             model += [
#                 nn.Conv2d(curr_dim, curr_dim*2, 3, stride=2, padding=1),
#                 nn.InstanceNorm2d(curr_dim*2),
#                 nn.ReLU(inplace=True)
#             ]
#             curr_dim *= 2
        
#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResnetBlock(curr_dim)]
        
#         # Upsampling
#         for _ in range(2):
#             model += [
#                 nn.ConvTranspose2d(curr_dim, curr_dim//2, 3, stride=2, 
#                                    padding=1, output_padding=1),
#                 nn.InstanceNorm2d(curr_dim//2),
#                 nn.ReLU(inplace=True)
#             ]
#             curr_dim //= 2
        
#         # Output layer
#         model += [
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(curr_dim, output_nc, 7),
#             # nn.Sigmoid()  # Ensure output is positive for scaling
#         ]
        
#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         input_image = x 
#         scale_field = self.model(x)  # Output in [0,1]
#         # scale_field = 1.0 + 0.5 * (scale_field - 0.5)  # Map to [0.5,1.5]
#         # scale_field = torch.clamp(scale_field, 0.5, 1.5)  # Ensure stable range
#         return scale_field * input_image, scale_field


class Discriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64):
        """
        70×70 PatchGAN discriminator
        """
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # We reduce further for a 70x70 patch
        model += [
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Output 1 channel prediction map
        model += [nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
