import torch.nn as nn


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


LAYER_LOOKUP = {
    "2d": dict(
        Conv = nn.Conv2d,
        Norm = nn.InstanceNorm2d,
    ),
    "3d": dict(
        Conv = nn.Conv3d,
        Norm = nn.InstanceNorm3d,
    ),
}


class Discriminator(nn.Module):
    """
    70×70 PatchGAN (2-D)  ➜  70×70×70 PatchGAN (3-D).

    Args
    ----
    dim         : "2d" or "3d"
    input_nc    : # input channels
    ndf         : base feature width
    """
    def __init__(self, dim: str = "2d", input_nc: int = 1, ndf: int = 64):
        super().__init__()
        assert dim in ("2d", "3d"), "dim must be '2d' or '3d'"

        Conv = LAYER_LOOKUP[dim]["Conv"]
        Norm = LAYER_LOOKUP[dim]["Norm"]

        def block(in_ch, out_ch, stride):
            """Conv → Norm → LeakyReLU (helper)."""
            return [
                Conv(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                Norm(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers = []
        # --------------------- depth 1 ---------------------
        layers += block(input_nc,  ndf,     stride=2)
        layers += block(ndf,       ndf,     stride=1)

        # --------------------- depth 2 ---------------------
        layers += block(ndf,       ndf*2,   stride=2)
        layers += block(ndf*2,     ndf*2,   stride=1)

        # --------------------- depth 3 ---------------------
        layers += block(ndf*2,     ndf*4,   stride=2)
        layers += block(ndf*4,     ndf*4,   stride=1)

        # --------------------- depth 4 ---------------------
        layers += block(ndf*4,     ndf*8,   stride=2)
        layers += block(ndf*8,     ndf*8,   stride=1)

        # --------------------- extra downsample -------------
        layers += block(ndf*8,     ndf*8,   stride=2)
        layers += block(ndf*8,     ndf*8,   stride=1)

        # --------------------- head -------------------------
        layers += [
            Conv(ndf*8, 1, kernel_size=3, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Input shape:
          2-D –  (B, C,  H,  W)
          3-D –  (B, C, D, H, W)
        """
        return self.model(x)

