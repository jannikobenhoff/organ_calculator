import torch
import torchvision.utils as vutils
import os
from torch.utils.data import DataLoader
from models import GeneratorResNet
import glob
from dataset import SingleVolume2DDataset  
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
    print("Starting CycleGAN inference...", flush=True)

    # --------------------------------------------------
    # 0. File Paths & Hyperparameters
    # --------------------------------------------------
    CHECKPOINT_PATH = "checkpoints/cyclegan_epoch_000.pth"
    CT_VOLUME_PATH = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTs/"
    OUTPUT_SLICE_DIR = "fake_slices"
    CT_OUTPUT_SLICE_DIR = "ct_slices"
    SCALAR_FIELD_DIR = "scalar_fields"
    OUTPUT_VOLUME_PATH = "fake_mri_volume.nii.gz"

    os.makedirs(OUTPUT_SLICE_DIR, exist_ok=True)
    os.makedirs(CT_OUTPUT_SLICE_DIR, exist_ok=True)
    os.makedirs(SCALAR_FIELD_DIR, exist_ok=True)

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
    all_fake_slices = []  

    with torch.no_grad():
        for i, (ct_slice,) in enumerate(data_loader):
            ct_slice_gpu = ct_slice.to(device)

            # **Generate the scalar field (output of generator)**
            scalar_field = G_ct2mri(ct_slice_gpu)  # Output range: ~[0.5, 1.5]

            # **Apply scalar field to the input CT slice**
            fake_mri = ct_slice_gpu * scalar_field  
            fake_mri = (fake_mri * 350) + 150  # Convert [-1,1] → [-200,500]
            fake_mri = torch.clamp(fake_mri, -200, 500)  # Prevent outliers

            # Convert tensors to numpy for saving
            fake_mri_2d = fake_mri[0, 0].cpu().numpy()  
            fake_mri_png = (fake_mri - (-200)) / (500 - (-200))  # Scale to [0,1]

            scalar_field_2d = scalar_field[0, 0].cpu().numpy()  

            # Store transformed slices for NIfTI reconstruction
            all_fake_slices.append(fake_mri_2d)

            # ------------------------------------
            # Save PNG for CT, Scalar Field, & Fake MRI
            # ------------------------------------
            # 1) Fake MRI slice
            fake_mri_out_path = os.path.join(OUTPUT_SLICE_DIR, f"fakeMRI_{i:04d}.png")
            vutils.save_image(fake_mri_png, fake_mri_out_path, normalize=True)
            
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

    print(f"Reassembled 3D volume saved to {OUTPUT_VOLUME_PATH}")
