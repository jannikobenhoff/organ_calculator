import os
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from med_synth_gan.dataset.single_2d_dataset import SingleVolume2DDataset
from med_synth_gan.inference.utils import png_slices_to_nifti
import shutil


class VolumeInference:
    def __init__(self, test_volume_path, output_dir, device):
        self.test_volume_path = test_volume_path
        self.output_dir = output_dir
        self.device = device
        self.middle_slices = []
        os.makedirs(output_dir, exist_ok=True)

    def run_inference(self, model, epoch):
        # Create directories for this epoch
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        fake_mri_dir = os.path.join(epoch_dir, "fake_mri_slices")
        ct_dir = os.path.join(epoch_dir, "ct_slices")

        for d in [fake_mri_dir, ct_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        # Create dataset and dataloader
        test_dataset = SingleVolume2DDataset(
            volume_path=self.test_volume_path,
            slice_axis=2,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )

        print(f"Starting inference on {self.test_volume_path}")

        model.eval()
        fake_mri_slices = []
        ct_slices = []

        with torch.no_grad():
            for i, (ct_slice,) in enumerate(test_loader):
                ct_slice = ct_slice.to(self.device)
                fake_mri, scalar_field = model.G_ct2mri(ct_slice)

                # Save slices
                vutils.save_image(
                    fake_mri,
                    os.path.join(fake_mri_dir, f"fakeMRI_{i:04d}.png"),
                    normalize=True
                )
                vutils.save_image(
                    ct_slice,
                    os.path.join(ct_dir, f'CT_{i:04d}.png'),
                    normalize=True
                )

                fake_mri_slices.append(fake_mri)
                ct_slices.append(ct_slice)

        # Convert to NIfTI
        output_nifti_path = os.path.join(epoch_dir, f"fake_mri_epoch_{epoch}.nii.gz")
        ct_output_path = os.path.join(epoch_dir, f"ct_epoch_{epoch}.nii.gz")
        png_slices_to_nifti(fake_mri_dir, output_nifti_path)
        png_slices_to_nifti(ct_dir, ct_output_path)

        # Store middle slice
        if fake_mri_slices:
            middle_index = len(fake_mri_slices) // 3
            if not self.middle_slices:
                self.middle_slices.append(ct_slices[middle_index].squeeze(0).cpu())
            middle_slice = fake_mri_slices[middle_index].squeeze(0).cpu()
            self.middle_slices.append(middle_slice)

        print(f"Epoch {epoch}: Saved inference results to {epoch_dir}")
        model.train()

    def save_final_grid(self):
        if self.middle_slices:
            grid_image = vutils.make_grid(self.middle_slices, nrow=4, normalize=False)
            grid_path = os.path.join(self.output_dir, "middle_slices_grid.png")
            vutils.save_image(grid_image, grid_path)
            print(f"Saved middle slice grid to {grid_path}")
