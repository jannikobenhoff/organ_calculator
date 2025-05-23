import torch
from med_synth_gan.dataset.ct_mri_2d_dataset import CtMri2DDataset
from med_synth_gan.train.train import MedSynthGANModule
import glob
import os
import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader
from med_synth_gan.dataset.single_2d_dataset import SingleVolume2DDataset
from med_synth_gan.models.models import UNet
import torchvision.utils as vutils
from med_synth_gan.inference.utils import png_slices_to_nifti

def generate_mri_from_ct(ct_dir, output_dir, checkpoint_path, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = UNet().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print('Generator loaded', flush=True)

    ct_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))

    for ct_path in ct_paths[:10]:
        vol_name = os.path.basename(ct_path).split('.')[0]

        test_dataset = SingleVolume2DDataset(
            volume_path=ct_path,
            slice_axis=2,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )

        fake_mri_slices = []

        fake_mri_dir = os.path.join(output_dir, "fake_mri_slices")
        if os.path.exists(fake_mri_dir):
            shutil.rmtree(fake_mri_dir)
        os.makedirs(fake_mri_dir)

        with torch.no_grad():
            for i, (ct_slice,) in enumerate(test_loader):
                ct_slice = ct_slice.to(device)
                fake_mri, _ = generator(ct_slice)

                # Explicitly resize back to original dimensions
                fake_mri_resized = torch.nn.functional.interpolate(
                    fake_mri,
                    size=(test_dataset.orig_h, test_dataset.orig_w),
                    mode='bilinear',
                    align_corners=False
                )

                vutils.save_image(
                    fake_mri_resized,
                    os.path.join(fake_mri_dir, f"fakeMRI_{i:04d}.png"),
                    normalize=False
                )

                fake_mri_slices.append(fake_mri_resized.cpu())

        # Convert back to NIfTI with correct original dimensions
        output_nifti_path = os.path.join(output_dir, f"synth_{vol_name}.nii.gz")
        png_slices_to_nifti(fake_mri_dir, output_nifti_path, reference_nifti_path=ct_path)

        # Copy original CT metadata
        basename = os.path.basename(ct_path)
        dest_path = os.path.join(output_dir, basename)
        shutil.copy2(ct_path, dest_path)

        print(f"Generated {vol_name}")

    print(f"Completed processing {len(ct_paths)} CT volumes")

def main():
    print("Starting MedSynthGAN generator", flush=True)

    generate_mri_from_ct(
        ct_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/",
        output_dir="synthesized_mri",
        checkpoint_path="../train/inference_mse_1e-05_5e-05_0.0001/checkpoints/best_model.pth",
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