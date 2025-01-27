import torch
import torchvision.utils as vutils
import os
from torch.utils.data import DataLoader

import nibabel as nib
import numpy as np
from PIL import Image

from models import GeneratorResNet
import glob
from dataset import SingleVolume2DDataset  # Make sure this dataset loads one .nii file

# --------------------------------------------------
# 0. File Paths & Hyperparameters
# --------------------------------------------------
CHECKPOINT_PATH = "checkpoints/cyclegan_epoch_004.pth"
CT_VOLUME_PATH = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTs/"
OUTPUT_SLICE_DIR = "fake_slices"
OUTPUT_VOLUME_PATH = "fake_mri_volume.nii.gz"

os.makedirs(OUTPUT_SLICE_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Load Generator
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_ct2mri = GeneratorResNet(input_nc=1, output_nc=1, n_residual_blocks=9).to(device)

# Load the checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
G_ct2mri.load_state_dict(checkpoint['G_ct2mri_state_dict'])

# Switch to evaluation mode
G_ct2mri.eval()

# --------------------------------------------------
# 2. Dataset & DataLoader
# --------------------------------------------------
# This dataset should handle a single volume (CT_VOLUME_FILE) and slice it.
CT_VOLUME_FILES = glob.glob(os.path.join(CT_VOLUME_PATH, "*.nii.gz"))
CT_VOLUME_FILE = CT_VOLUME_FILES[0] if CT_VOLUME_FILES else None

dataset = SingleVolume2DDataset(
    volume_path=CT_VOLUME_FILE,  # <-- must be a single NIfTI file
    transform=None,              # or your transforms
    slice_axis=2,                # or whichever axis you want
    min_max_normalize=True
)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# --------------------------------------------------
# 3. Inference Loop: Save Every 10th Slice
# --------------------------------------------------
with torch.no_grad():
    for i, (ct_slice,) in enumerate(data_loader):
        # Only save slices where i % 10 == 0
        if i % 10 != 0:
            continue
        
        # Move slice to GPU (if available)
        ct_slice = ct_slice.to(device)
        
        # Generate fake MRI
        fake_mri = G_ct2mri(ct_slice)  # shape [1, 1, H, W]
        
        # Save image
        slice_out_path = os.path.join(OUTPUT_SLICE_DIR, f"fakeMRI_{i:04d}.png")
        vutils.save_image(
            fake_mri,
            slice_out_path,
            normalize=True,
            range=(-1, 1)  # If you trained in [-1,1] range
        )
        
        print(f"[Slice {i}] Saved: {slice_out_path}")

print("Finished saving every 10th slice from the test volume.")