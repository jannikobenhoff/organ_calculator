import torch
import torchvision.utils as vutils
import os
from torch.utils.data import DataLoader

import nibabel as nib
import numpy as np
from PIL import Image

from models import GeneratorResNet
import glob
from dataset import SingleVolume2DDataset  # A dataset that loads 1 .nii file
from train import nifit_transform

if __name__ == "__main__": 
    print("Starting CycleGAN inference...", flush=True)
    
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
    # 2. Find a Single CT Volume File & Create Dataset
    # --------------------------------------------------
    CT_VOLUME_FILES = glob.glob(os.path.join(CT_VOLUME_PATH, "*.nii.gz"))
    if not CT_VOLUME_FILES:
        raise FileNotFoundError(f"No NIfTI files found in {CT_VOLUME_PATH}")

    CT_VOLUME_FILE = CT_VOLUME_FILES[0]  # Just take the first file

    # This dataset loads one NIfTI volume, slices along slice_axis=2
    dataset = SingleVolume2DDataset(
        volume_path=CT_VOLUME_FILE,
        transform=nifit_transform,   # Use the same transform as training
        slice_axis=2,
        min_max_normalize=True
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --------------------------------------------------
    # 3. Inference: Generate All Slices, Save PNGs for Every 10th
    # --------------------------------------------------

    all_fake_slices = []  # Will hold the entire set of generated 2D slices for 3D reassembly

    with torch.no_grad():
        for i, (ct_slice,) in enumerate(data_loader):
            # ct_slice shape: [1, 1, H, W] after transforms, or [1, C, H, W] if C>1
            ct_slice_gpu = ct_slice.to(device)

            # Generate fake MRI
            fake_mri = G_ct2mri(ct_slice_gpu)  # [1, 1, H, W]

            # Convert the fake MRI (which is a 4D tensor) to a 2D numpy array
            # so we can reconstruct a volume later
            # shape: [B=1, C=1, H, W] -> [H, W]
            fake_mri_2d = fake_mri[0, 0].cpu().numpy()  # Squeeze out batch & channel

            # Keep track of all slices in a Python list
            all_fake_slices.append(fake_mri_2d)

            # ------------------------------------
            # Save PNG for both CT & Fake MRI (every 10th slice)
            # ------------------------------------
            if i % 10 == 0:
                # 1) Save the fake MRI slice
                fake_mri_out_path = os.path.join(OUTPUT_SLICE_DIR, f"fakeMRI_{i:04d}.png")
                vutils.save_image(
                    fake_mri,  # still the 4D tensor [1,1,H,W]
                    fake_mri_out_path,
                    normalize=True
                    # range=(-1, 1) -> If you have a newer torchvision version & your data is in [-1,1]
                )

                # 2) Save the corresponding CT slice
                #    Note: we must do this before sending `ct_slice` to GPU or after we clone it,
                #    but here it's still on GPU in `ct_slice_gpu`. We'll use the original CPU tensor.
                ct_slice_out_path = os.path.join(OUTPUT_SLICE_DIR, f"CT_{i:04d}.png")
                vutils.save_image(
                    ct_slice,  # shape [1,1,H,W], on CPU (the DataLoader output)
                    ct_slice_out_path,
                    normalize=True
                )

                print(f"[Slice {i}] Saved fake MRI: {fake_mri_out_path} | CT: {ct_slice_out_path}")

    print(f"Finished inference on volume: {CT_VOLUME_FILE}")
    print(f"Saved sample slices (every 10th) to: {OUTPUT_SLICE_DIR}")

    # --------------------------------------------------
    # 4. Reconstruct Full 3D Volume as NIfTI from All Generated Slices
    # --------------------------------------------------

    # (A) Read original volume shape & affine from the input CT volume:
    original_nifti = nib.load(CT_VOLUME_FILE)
    original_affine = original_nifti.affine
    original_shape = original_nifti.shape  # e.g. (Dx, Dy, Dz)

    # In your SingleVolume2DDataset, we used slice_axis=2, so we have one slice per z-index
    num_slices = len(all_fake_slices)

    # If you used transforms that changed H and W, your reassembled volume won't match original_shape exactly.
    # We'll reassemble using the size from 'all_fake_slices[0]'.
    h, w = all_fake_slices[0].shape  # size after transforms

    # We'll create a 3D array [h, w, num_slices], which matches slicing axis=2
    fake_volume_3d = np.zeros((h, w, num_slices), dtype=np.float32)

    for z in range(num_slices):
        fake_volume_3d[:, :, z] = all_fake_slices[z]

    # (B) Save as a new NIfTI
    # The affine from the original volume might not perfectly match if shape changed,
    # but let's reuse it for convenience.
    fake_nifti = nib.Nifti1Image(fake_volume_3d, affine=original_affine)
    nib.save(fake_nifti, OUTPUT_VOLUME_PATH)

    print(f"Reassembled 3D volume saved to {OUTPUT_VOLUME_PATH}")
