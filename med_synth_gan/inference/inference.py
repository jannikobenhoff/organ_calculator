import os
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from med_synth_gan.dataset.single_2d_dataset import SingleVolume2DDataset
from med_synth_gan.inference.utils import png_slices_to_nifti
import shutil
import numpy as np
from PIL import Image


class VolumeInferenceCallback(Callback):
    def __init__(self, test_volume_path, output_dir):
        super().__init__()
        self.test_volume_path = test_volume_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.middle_slices = []

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 1 == 0:
            # Create directories for this epoch
            epoch_dir = os.path.join(self.output_dir, f"epoch_{trainer.current_epoch}")
            fake_mri_dir = os.path.join(epoch_dir, "fake_mri_slices")
            ct_dir = os.path.join(epoch_dir, "ct_slices")
            # scalar_field_dir = os.path.join(epoch_dir, "scalar_field_slices")

            for d in [fake_mri_dir, ct_dir]: #, scalar_field_dir]:
                # Delete directory and all its contents if it exists
                if os.path.exists(d):
                    shutil.rmtree(d)
                # Create new directory
                os.makedirs(d)

            # Create dataset and dataloader for the test volume
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
            # Perform inference
            pl_module.eval()
            fake_mri_slices = []
            ct_slices = []
            with torch.no_grad():
                for i, (ct_slice,) in enumerate(test_loader):
                    ct_slice = ct_slice.to(pl_module.device)

                    # Generate fake MRI and scalar field
                    fake_mri, scalar_field = pl_module.G_ct2mri(ct_slice)
                    fake_mri_slices.append(fake_mri)
                    ct_slices.append(ct_slice)

                    # Save slices as PNG
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

            # Convert PNG slices to NIfTI
            output_nifti_path = os.path.join(epoch_dir, f"fake_mri_epoch_{trainer.current_epoch}.nii.gz")
            ct_output_path = os.path.join(epoch_dir, f"ct_epoch_{trainer.current_epoch}.nii.gz")

            png_slices_to_nifti(fake_mri_dir, output_nifti_path)
            png_slices_to_nifti(ct_dir, ct_output_path)

            print(f"Epoch {trainer.current_epoch}: Saved inference results to {epoch_dir}")

            # Store the middle slice of the current epoch's fake MRI
            middle_index = len(fake_mri_slices) // 3
            if len(self.middle_slices) == 0:
                self.middle_slices.append(ct_slices[middle_index])
            if fake_mri_slices:
                middle_slice = fake_mri_slices[middle_index].squeeze(0)  # Remove batch dimension
                self.middle_slices.append(middle_slice)

    def on_train_end(self, trainer, pl_module):
        if self.middle_slices:
            grid_image = vutils.make_grid(self.middle_slices, nrow=4, normalize=True)
            grid_path = os.path.join(self.output_dir, "middle_slices_grid.png")
            vutils.save_image(grid_image, grid_path)
            print(f"Saved middle slice grid to {grid_path}")


# import os
# import torch
# from pytorch_lightning.callbacks import Callback
# from torch.utils.data import DataLoader
# import torchvision.utils as vutils
# from med_synth_gan.dataset.single_2d_dataset import SingleVolume2DDataset
# from med_synth_gan.inference.utils import png_slices_to_nifti
# import shutil
#
#
# class VolumeInferenceCallback(Callback):
#     def __init__(self, test_volume_path, output_dir):
#         super().__init__()
#         self.test_volume_path = test_volume_path
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)
#
#     def on_train_epoch_end(self, trainer, pl_module):
#         if (trainer.current_epoch + 1) % 1 == 0:
#             # Create directories for this epoch
#             epoch_dir = os.path.join(self.output_dir, f"epoch_{trainer.current_epoch}")
#             fake_mri_dir = os.path.join(epoch_dir, "fake_mri_slices")
#             ct_dir = os.path.join(epoch_dir, "ct_slices")
#             scalar_field_dir = os.path.join(epoch_dir, "scalar_field_slices")
#
#             for d in [fake_mri_dir, ct_dir, scalar_field_dir]:
#                 # Delete directory and all its contents if it exists
#                 if os.path.exists(d):
#                     shutil.rmtree(d)
#                 # Create new directory
#                 os.makedirs(d)
#
#             # Create dataset and dataloader for the test volume
#             test_dataset = SingleVolume2DDataset(
#                 volume_path=self.test_volume_path,
#                 slice_axis=2,
#             )
#
#             test_loader = DataLoader(
#                 test_dataset,
#                 batch_size=1,
#                 shuffle=False
#             )
#
#             print(f"Starting inference on {self.test_volume_path}")
#             # Perform inference
#             pl_module.eval()
#             with torch.no_grad():
#                 for i, (ct_slice,) in enumerate(test_loader):
#                     ct_slice = ct_slice.to(pl_module.device)
#
#                     # Generate fake MRI and scalar field
#                     fake_mri, scalar_field = pl_module.G_ct2mri(ct_slice)
#
#                     # Save slices as PNG
#                     vutils.save_image(
#                         fake_mri,
#                         os.path.join(fake_mri_dir, f"fakeMRI_{i:04d}.png"),
#                         normalize=True
#                     )
#
#                     vutils.save_image(
#                         ct_slice,
#                         os.path.join(ct_dir, f'CT_{i:04d}.png'),
#                         normalize=True
#                     )
#
#                     # vutils.save_image(
#                     #     scalar_field,
#                     #     os.path.join(scalar_field_dir, f"ScalarField_{i:04d}.png"),
#                     #     normalize=True
#                     # )
#
#             # Convert PNG slices to NIfTI
#             output_nifti_path = os.path.join(epoch_dir, f"fake_mri_epoch_{trainer.current_epoch}.nii.gz")
#             ct_output_path = os.path.join(epoch_dir, f"ct_epoch_{trainer.current_epoch}.nii.gz")
#
#             # Assuming you have these utility functions
#             png_slices_to_nifti(fake_mri_dir, output_nifti_path)
#             png_slices_to_nifti(ct_dir, ct_output_path)
#
#             print(f"Epoch {trainer.current_epoch}: Saved inference results to {epoch_dir}")
