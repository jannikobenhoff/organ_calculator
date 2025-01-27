import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
import nibabel as nib
import numpy as np
from PIL import Image
from models import GeneratorResNet
from dataset import SingleVolume2DDataset  # Example dataset for 1 volume
from utils import reassemble_3D    # Your function from previous snippet

# 0. Paths
CHECKPOINT_PATH = "checkpoints/cyclegan_epoch_004.pth"
CT_VOLUME_PATH = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTs/"
OUTPUT_SLICE_DIR = "fake_slices"
OUTPUT_VOLUME_PATH = "fake_mri_volume.nii.gz"

os.makedirs(OUTPUT_SLICE_DIR, exist_ok=True)

# 1. Load generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_ct2mri = GeneratorResNet(input_nc=1, output_nc=1, n_residual_blocks=9).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
G_ct2mri.load_state_dict(checkpoint['G_ct2mri_state_dict'])
G_ct2mri.eval()

# 2. Prepare dataset/dataloader (e.g., SingleVolume2DDataset)
dataset = SingleVolume2DDataset(
    volume_path=CT_VOLUME_PATH,
    transform=None,         # or your transforms
    slice_axis=2,
    min_max_normalize=True
)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 3. Generate & save each slice
with torch.no_grad():
    for i, (ct_slice,) in enumerate(data_loader):
        ct_slice = ct_slice.to(device)
        fake_mri = G_ct2mri(ct_slice)  # shape [1, 1, H, W]
        
        slice_out_path = os.path.join(OUTPUT_SLICE_DIR, f"fakeMRI_{i:04d}.png")
        vutils.save_image(
            fake_mri,
            slice_out_path,
            normalize=True, range=(-1,1)  # If you trained with [-1,1] scaling
        )
        if i % 50 == 0:
            print(f"Saved slice {i} to {slice_out_path}")

# 4. Reassemble slices into a 3D volume
#    We must know the original volume shape or how many slices we have.
#    Suppose your dataset has `len(dataset) = number_of_slices`.
#    The shape might be (H, W, number_of_slices) if slice_axis=2.

# Retrieve the original shape from the nib file or from the dataset
vol_img = nib.load(CT_VOLUME_PATH)
vol_data = vol_img.get_fdata(dtype=np.float32)
original_shape = vol_data.shape   # e.g., (Hx, Hy, Hz)

# Now call reassemble_3D (adapt it to your shape & slice axis)
reassemble_3D(
    image_folder=OUTPUT_SLICE_DIR,
    volume_shape=original_shape,    # e.g. (Hx, Hy, Hz)
    slice_axis=2,                   # Must match how you sliced
    out_nii=OUTPUT_VOLUME_PATH
)

print(f"Reassembled 3D volume saved to {OUTPUT_VOLUME_PATH}")
