import torchvision.transforms as T
import torch

contrast_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Lambda(lambda x: torch.clamp(x, -200, 500)),
        T.Normalize(mean=[-200], std=[700]),
    ])