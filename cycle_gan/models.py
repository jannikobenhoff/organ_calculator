import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import itertools


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim*2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim*2),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResnetBlock(curr_dim)]
        
        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim//2, 3, stride=2, 
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(curr_dim//2),
                nn.ReLU(inplace=True)
            ]
            curr_dim //= 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim, output_nc, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64):
        """
        A 70×70 PatchGAN discriminator (relatively standard).
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
