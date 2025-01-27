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
import SimpleITK as sitk

def png_slices_to_nifti(png_folder, output_nifti):
    """
    Reads a sorted list of PNG files in `png_folder` and
    assembles them into a 3D volume, saved as `output_nifti`.
    """
    # 1. Gather PNG slice paths
    #    Make sure they sort in ascending Z order (e.g., slice_0000.png, slice_0001.png,...).
    slice_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
    if not slice_files:
        raise ValueError(f"No PNG files found in folder: {png_folder}")

    # 2. Use SimpleITK ImageSeriesReader to read as a 3D volume
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(slice_files)
    
    # 3. Execute reading => 3D volume
    volume = reader.Execute()
    
    # 4. Save as NIfTI
    sitk.WriteImage(volume, output_nifti)
    print(f"Saved 3D volume to: {output_nifti}")

if __name__ == "__main__": 
    print("Starting CycleGAN inference...", flush=True)

    # --------------------------------------------------
    # 0. File Paths & Hyperparameters
    # --------------------------------------------------
    CHECKPOINT_PATH = "checkpoints/cyclegan_epoch_004.pth"
    CT_VOLUME_PATH = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTs/"
    OUTPUT_SLICE_DIR = "fake_slices"
    CT_OUTPUT_SLICE_DIR = "ct_slices"
    OUTPUT_VOLUME_PATH = "fake_mri_volume.nii.gz"

    os.makedirs(OUTPUT_SLICE_DIR, exist_ok=True)
    os.makedirs(CT_OUTPUT_SLICE_DIR, exist_ok=True)

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
                ct_slice_out_path = os.path.join(CT_OUTPUT_SLICE_DIR, f"CT_{i:04d}.png")
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

    png_slices_to_nifti(OUTPUT_SLICE_DIR, OUTPUT_VOLUME_PATH)

    print(f"Reassembled 3D volume saved to {OUTPUT_VOLUME_PATH}")
