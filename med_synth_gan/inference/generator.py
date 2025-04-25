import torch
from med_synth_gan.dataset.ct_mri_2d_dataset import CtMri2DDataset
from med_synth_gan.train.train import MedSynthGANModule

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from med_synth_gan.dataset.generator_dataset import Generator2DDataset
from med_synth_gan.models.models import UNet
import torchvision.utils as vutils
from med_synth_gan.inference.utils import png_slices_to_nifti


def generate_mri_from_ct(ct_dir, output_dir, checkpoint_path, batch_size=8):
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = UNet().to(device)

    # Load trained weights (generator only)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    dataset = Generator2DDataset(ct_dir, slice_axis=2)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create output directories
    fake_mri_dir = os.path.join(output_dir, "fake_mri_slices")
    os.makedirs(fake_mri_dir, exist_ok=True)

    # Generate slices
    with torch.no_grad():
        for batch_idx, ct_slices in enumerate(loader):
            ct_slices = ct_slices.to(device)
            fake_mri, _ = generator(ct_slices)

            # Save individual slices
            for i in range(fake_mri.shape[0]):
                slice_idx = batch_idx * batch_size + i
                vutils.save_image(
                    fake_mri[i],
                    os.path.join(fake_mri_dir, f"fakeMRI_{slice_idx:04d}.png"),
                    normalize=True
                )

    # Convert to NIfTI volume
    output_nifti = os.path.join(output_dir, "synthetic_mri.nii.gz")
    png_slices_to_nifti(fake_mri_dir, output_nifti)

    print(f"Generated synthetic MRI volume saved to {output_nifti}")



def main():
    generate_mri_from_ct(
        ct_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/",
        output_dir="synthesized_mri",
        checkpoint_path="../train/inference_mse_1e-05_5e-05_0/checkpoints/best_model.pth",
        batch_size=24
    )

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Starting MedSynthGAN generator", flush=True)
    #
    # train_dataset = CtMri2DDataset(
    #     ct_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/",
    #     mri_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/t2Axial/",
    #     slice_axis=2
    # )
    #
    # # Initialize module
    # best_model = MedSynthGANModule(
    #     loss_type="mse",
    #     lambda_grad=0,
    #     lr=1e-5,
    #     lr_d=5e-5,
    # ).to(device)
    #
    #
    # checkpoint = torch.load("/home/jao4016/organ_calculator/med_synth_gan/train/inference_mse_1e-05_5e-05_0/checkpoints/best_model.pth", map_location=device)
    # best_model.G_ct2mri.load_state_dict(checkpoint['generator_state_dict'])
    # best_model.eval()


if __name__ == "__main__":
    main()