import shutil
from matplotlib import pyplot as plt
import torch
import torchvision.utils as vutils
import os
from torch.utils.data import DataLoader
from models import UNet
import glob
from dataset import Nifti2DDataset, SingleVolume2DDataset
from train import nifit_transform
import SimpleITK as sitk

def png_slices_to_nifti(png_folder, output_nifti):
    """
    Reads a sorted list of PNG files in `png_folder` and
    assembles them into a 3D volume, saved as `output_nifti`.
    """
    slice_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
    if not slice_files:
        raise ValueError(f"No PNG files found in folder: {png_folder}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(slice_files)
    
    volume = reader.Execute()
    sitk.WriteImage(volume, output_nifti)
    print(f"Saved 3D volume to: {output_nifti}")

if __name__ == "__main__": 
    print("Starting inference...", flush=True)

    # --------------------------------------------------
    # 0. File Paths & Hyperparameters
    # --------------------------------------------------
    CHECKPOINT_PATH = "checkpoints/cyclegan_epoch_001.pth"
    CT_VOLUME_PATH = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTs/"
    OUTPUT_SLICE_DIR = "fake_slices"
    CT_OUTPUT_SLICE_DIR = "ct_slices"
    SCALAR_FIELD_DIR = "scalar_fields"
    OUTPUT_VOLUME_PATH = "fake_mri_volume.nii.gz"
    CT_OUTPUT_VOLUME_PATH = "ct_volume.nii.gz"

    for directory in [OUTPUT_SLICE_DIR, CT_OUTPUT_SLICE_DIR, SCALAR_FIELD_DIR]:
        if os.path.exists(directory):
            shutil.rmtree(directory)

    # Then create fresh directories
    os.makedirs(OUTPUT_SLICE_DIR, exist_ok=True)
    os.makedirs(CT_OUTPUT_SLICE_DIR, exist_ok=True)
    os.makedirs(SCALAR_FIELD_DIR, exist_ok=True)

    # --------------------------------------------------
    # 1. Load Generator
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_ct2mri = UNet().to(device)

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

    CT_VOLUME_FILE = CT_VOLUME_FILES[0]  # Take the first file

    # Load dataset: extract slices from a NIfTI file
    # 2. Load dataset
    dataset = SingleVolume2DDataset(
        volume_path=CT_VOLUME_FILE,
        transform=nifit_transform,
        slice_axis=2
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --------------------------------------------------
    # 3. Inference: Generate All Slices
    # --------------------------------------------------
    with torch.no_grad():
        for i, (ct_slice,) in enumerate(data_loader):
            ct_slice_gpu = ct_slice.to(device)

            # **Generate the scalar field (output of generator)**
            fake_mri, scalar_field = G_ct2mri(ct_slice_gpu) 

            scalar_field_2d = scalar_field[0, 0].cpu().numpy()  

            # ------------------------------------
            # Save PNG for CT, Scalar Field, & Fake MRI
            # ------------------------------------
            # 1) Fake MRI slice
            fake_mri_np = fake_mri.cpu().numpy()[0, 0]  # [B,C,H,W] -> [H,W]
            fake_mri_out_path = os.path.join(OUTPUT_SLICE_DIR, f"fakeMRI_{i:04d}.png")
            # vutils.save_image(fake_mri, fake_mri_out_path, normalize=True)
            plt.imsave(
                os.path.join(OUTPUT_SLICE_DIR, f"fakeMRI_{i:03d}.png"),
                fake_mri_np,
                cmap='gray'
            )
            
            # 2) CT slice
            ct_slice_out_path = os.path.join(CT_OUTPUT_SLICE_DIR, f"CT_{i:04d}.png")
            vutils.save_image(ct_slice, ct_slice_out_path, normalize=True)

            # 3) Scalar Field slice
            scalar_field_out_path = os.path.join(SCALAR_FIELD_DIR, f"ScalarField_{i:04d}.png")
            vutils.save_image(scalar_field, scalar_field_out_path, normalize=True)

            print(f"[Slice {i}] Saved fake MRI: {fake_mri_out_path} | CT: {ct_slice_out_path} | Scalar Field: {scalar_field_out_path}")

    print(f"Finished inference on volume: {CT_VOLUME_FILE}")
    print(f"Saved sample slices to: {OUTPUT_SLICE_DIR}")

    # --------------------------------------------------
    # 4. Reconstruct Full 3D Volume as NIfTI
    # --------------------------------------------------
    png_slices_to_nifti(OUTPUT_SLICE_DIR, OUTPUT_VOLUME_PATH)
    png_slices_to_nifti(CT_OUTPUT_SLICE_DIR, CT_OUTPUT_VOLUME_PATH)
