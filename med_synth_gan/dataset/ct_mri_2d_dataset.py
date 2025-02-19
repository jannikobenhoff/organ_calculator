import os
import glob
import nibabel as nib
import numpy as np
from tifffile import enumarg
from torch.utils.data import Dataset
from PIL import Image
from med_synth_gan.dataset.utils import contrast_transform_ct, contrast_transform_mri
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from med_synth_gan.inference.utils import png_slices_to_nifti


class CtMri2DDataset(Dataset):
    def __init__(self, ct_dir, mri_dir, slice_axis=2):
        super().__init__()
        self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
        self.mri_vol_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))

        # Reduce ct data set sice to balance both
        self.ct_vol_paths = self.ct_vol_paths[:50]

        self.slice_axis = slice_axis

        # Use memory mapping for NIfTI files
        self.ct_volumes = [nib.load(path).get_fdata(dtype=np.float32, caching='unchanged')
                           for path in self.ct_vol_paths]
        self.mri_volumes = [nib.load(path).get_fdata(dtype=np.float32, caching='unchanged')
                            for path in self.mri_vol_paths]

        # Pre-calculate indices and p99 values
        self.ct_slice_indices = self._calculate_slice_indices(self.ct_volumes)
        self.mri_slice_indices = self._calculate_slice_indices(self.mri_volumes)

        # Calculate both min and max values for better normalization
        self.mri_stats = {slice_i: {
            'p99': np.percentile(vol, 99),
            'p1': np.percentile(vol, 1)
        } for slice_i, vol in enumerate(self.mri_volumes)}

        self.dataset_len = max(len(self.ct_slice_indices), len(self.mri_slice_indices))

        # Modified MRI transform with improved normalization
        self.contrast_transform_mri = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def _calculate_slice_indices(self, volumes):
        indices = []
        for vol_idx, vol in enumerate(volumes):
            num_slices = vol.shape[self.slice_axis]
            indices.extend([(vol_idx, s) for s in range(num_slices)])
        return indices

    def _get_slice(self, volume, slice_idx):
        if self.slice_axis == 0:
            return volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            return volume[:, slice_idx, :]
        return volume[:, :, slice_idx]

    def __getitem__(self, idx):
        # CT processing remains the same
        ct_vol_idx, ct_slice_idx = self.ct_slice_indices[idx % len(self.ct_slice_indices)]
        ct_slice = self._get_slice(self.ct_volumes[ct_vol_idx], ct_slice_idx)
        ct_img = Image.fromarray(ct_slice, mode='F')
        ct_tensor = contrast_transform_ct(ct_img)


        # Modified MRI processing
        mri_vol_idx, mri_slice_idx = self.mri_slice_indices[idx % len(self.mri_slice_indices)]
        mri_slice = self._get_slice(self.mri_volumes[mri_vol_idx], mri_slice_idx)
        mri_img = Image.fromarray(mri_slice, mode='F')
        mri_tensor = contrast_transform_mri(mri_img)

        return ct_tensor, mri_tensor

        # Apply MRI transformations with improved normalization
        mri_tensor = self.contrast_transform_mri(mri_img)

        # Normalize using both lower and upper bounds
        p99 = self.mri_stats[mri_vol_idx]['p99']
        p1 = self.mri_stats[mri_vol_idx]['p1']

        # Clip and normalize
        mri_tensor = torch.clamp(mri_tensor, min=p1, max=p99)
        mri_tensor = (mri_tensor - p1) / (p99 - p1)  # Normalize to [0, 1]

        if mri_tensor.mean() < 0.01:
            return None

        return ct_tensor, mri_tensor

    def __len__(self):
        return self.dataset_len

# class CtMri2DDataset(Dataset):
#     def __init__(self, ct_dir, mri_dir, slice_axis=2, cache_dir=None):
#         super().__init__()
#         self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
#         self.mri_vol_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))
#         self.slice_axis = slice_axis
#         self.cache_dir = cache_dir
#
#         # Use memory mapping for NIfTI files
#         self.ct_volumes = [nib.load(path).get_fdata(dtype=np.float32, caching='unchanged')
#                            for path in self.ct_vol_paths]
#         self.mri_volumes = [nib.load(path).get_fdata(dtype=np.float32, caching='unchanged')
#                             for path in self.mri_vol_paths]
#
#         # Pre-calculate indices and p99 values
#         self.ct_slice_indices = self._calculate_slice_indices(self.ct_volumes)
#         self.mri_slice_indices = self._calculate_slice_indices(self.mri_volumes)
#         self.mri_p99_values = {slice_i: np.percentile(vol, 90) for slice_i, vol in enumerate(self.mri_volumes)}
#
#         self.dataset_len = max(len(self.ct_slice_indices), len(self.mri_slice_indices))
#
#         # Create transforms once
#         self.contrast_transform_mri = T.Compose([
#             T.Resize((256, 256)),
#             T.ToTensor(),
#             T.Lambda(lambda x: torch.clamp(x, min=0)),
#         ])
#
#     def _calculate_slice_indices(self, volumes):
#         indices = []
#         for vol_idx, vol in enumerate(volumes):
#             num_slices = vol.shape[self.slice_axis]
#             indices.extend([(vol_idx, s) for s in range(num_slices)])
#         return indices
#
#     def _get_slice(self, volume, slice_idx):
#         if self.slice_axis == 0:
#             return volume[slice_idx, :, :]
#         elif self.slice_axis == 1:
#             return volume[:, slice_idx, :]
#         return volume[:, :, slice_idx]
#
#     def __len__(self):
#         return self.dataset_len
#
#     def __getitem__(self, idx):
#         # Get CT slice
#         ct_vol_idx, ct_slice_idx = self.ct_slice_indices[idx % len(self.ct_slice_indices)]
#         ct_slice = self._get_slice(self.ct_volumes[ct_vol_idx], ct_slice_idx)
#         ct_img = Image.fromarray(ct_slice, mode='F')
#         ct_tensor = contrast_transform_ct(ct_img)
#
#         # Get MRI slice
#         mri_vol_idx, mri_slice_idx = self.mri_slice_indices[idx % len(self.mri_slice_indices)]
#         mri_slice = self._get_slice(self.mri_volumes[mri_vol_idx], mri_slice_idx)
#         mri_img = Image.fromarray(mri_slice, mode='F')
#
#         # Apply MRI transformations
#         mri_tensor = self.contrast_transform_mri(mri_img)
#         mri_tensor = mri_tensor / self.mri_p99_values[mri_vol_idx]
#
#         if mri_tensor.mean() < 0.01:
#             return None
#
#         return ct_tensor, mri_tensor
#
#
# class CtMri2DDataset3(Dataset):
#     """
#     Loads a list of 3D NIfTI volumes (CT and MRI), extracts 2D slices,
#     and returns them as single images for training.
#     """
#
#     def __init__(
#             self,
#             ct_dir,  # Directory containing CT NIfTI files
#             mri_dir,  # Directory containing MRI NIfTI files
#             slice_axis=2,  # 0 -> axial slices across x, 1 -> across y, 2 -> across z
#     ):
#         super().__init__()
#
#         self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
#         self.mri_vol_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))
#         self.transform_ct = contrast_transform_ct
#         self.slice_axis = slice_axis
#
#         # Preload volumes and create a list of slices for CT
#         self.ct_slices = self._load_slices(self.ct_vol_paths, is_ct=True)
#
#         self.mri_p99_dict = {}  # Store p99 values for each MRI file
#         self.slice_to_vol_idx = {}  # Map slice index to volume index
#         self.mri_slices = []
#
#         for vol_idx, path in enumerate(self.mri_vol_paths):
#             vol = nib.load(path).get_fdata(dtype=np.float32)
#             self.mri_p99_dict[vol_idx] = np.percentile(vol, 99)
#
#             # Load slices and keep track of which volume they came from
#             num_slices = vol.shape[self.slice_axis]
#             start_idx = len(self.mri_slices)
#             for s in range(num_slices):
#                 slice_data = self._get_slice(vol, s)
#                 self.mri_slices.append(slice_data)
#                 self.slice_to_vol_idx[start_idx + s] = vol_idx
#
#         # Pre-transform MRI slices
#         self.mri_slices = self._preprocess_mri_slices()
#
#         # Ensure datasets match in length for unpaired training
#         self.dataset_len = max(len(self.ct_slices), len(self.mri_slices))
#
#     def _load_slices(self, volume_paths, is_ct=False):
#         """
#         Loads all slices from the provided NIfTI volumes and applies transformation if it is a CT scan.
#         """
#         slices = []
#         for path in volume_paths:
#             vol = nib.load(path).get_fdata(dtype=np.float32)
#
#             num_slices = vol.shape[self.slice_axis]
#             for s in range(num_slices):
#                 slice_data = self._get_slice(vol, s)
#                 img = Image.fromarray(slice_data, mode='F')
#
#                 # Apply CT transform
#                 if is_ct:
#                     slice_data = self.transform_ct(img)
#                 else:
#                     slice_data = img
#                 slices.append(slice_data)
#         return slices
#
#     def _get_slice(self, volume, slice_idx):
#         """
#         Extract a 2D slice from a 3D volume given slice_axis and slice_idx.
#         """
#         if self.slice_axis == 0:
#             slice_2d = volume[slice_idx, :, :]
#         elif self.slice_axis == 1:
#             slice_2d = volume[:, slice_idx, :]
#         else:  # default: 2
#             slice_2d = volume[:, :, slice_idx]
#
#         return slice_2d
#
#     def _preprocess_mri_slices(self):
#         """
#         Pre-transforms all MRI slices and stores them.
#         """
#         transformed_slices = []
#         for idx, mri_slice in enumerate(self.mri_slices):
#             vol_idx = self.slice_to_vol_idx[idx]
#             vol_p99 = self.mri_p99_dict[vol_idx]
#
#             # Convert to PIL for compatibility with torchvision transforms
#             mri_img = Image.fromarray(mri_slice, mode='F')
#
#             contrast_transform_mri = T.Compose([
#                 T.Resize((256, 256)),
#                 T.ToTensor(),
#                 T.Lambda(lambda x: torch.clamp(
#                     input=x,
#                     min=torch.tensor(0, dtype=x.dtype, device=x.device),
#                     max=torch.tensor(vol_p99, dtype=x.dtype, device=x.device)
#                 )),
#                 T.Lambda(lambda x: x / vol_p99),
#             ])
#
#             mri_img = contrast_transform_mri(mri_img)
#             if mri_img.mean() > 0.01:
#                 transformed_slices.append(mri_img)
#         return transformed_slices
#
#     def __len__(self):
#         return self.dataset_len
#
#     def __getitem__(self, idx):
#         ct_slice = self.ct_slices[idx % len(self.ct_slices)]
#         mri_slice = self.mri_slices[idx % len(self.mri_slices)]
#
#         if mri_slice.mean() < 0.01:  # 1e-8
#             # Sort out bad training data
#             return None
#
#         return ct_slice, mri_slice
#
#
# class CtMri2DDataset2(Dataset):
#     """
#     Loads a list of 3D NIfTI volumes (CT and MRI), extracts 2D slices,
#     and returns them as single images for training.
#     """
#     def __init__(
#         self,
#         ct_dir,          # Directory containing CT NIfTI files
#         mri_dir,         # Directory containing MRI NIfTI files
#         slice_axis=2,    # 0 -> axial slices across x, 1 -> across y, 2 -> across z
#     ):
#         super().__init__()
#
#         self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
#
#         self.mri_vol_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))
#
#         self.transform_ct = contrast_transform_ct
#
#         self.slice_axis = slice_axis
#
#         # Preload volumes and create a list of slices for CT
#         self.ct_slices = self._load_slices(self.ct_vol_paths)
#         self.mri_slices = self._load_slices(self.mri_vol_paths)
#
#         # Ensure datasets match in length for unpaired training
#         self.dataset_len = max(len(self.ct_slices), len(self.mri_slices))
#
#     def _load_slices(self, volume_paths):
#         """
#         Loads all slices from the provided NIfTI volumes and applies transformation if it is a CT scan.
#         """
#         slices = []
#         for path in volume_paths:
#             vol = nib.load(path).get_fdata(dtype=np.float32)
#
#             num_slices = vol.shape[self.slice_axis]
#             for s in range(num_slices):
#                 slice_data = self._get_slice(vol, s)
#                 slices.append(slice_data)
#         return slices
#
#     def __len__(self):
#         return self.dataset_len
#
#     def _get_slice(self, volume, slice_idx):
#         """
#         Extract a 2D slice from a 3D volume given slice_axis and slice_idx.
#         """
#         if self.slice_axis == 0:
#             slice_2d = volume[slice_idx, :, :]
#         elif self.slice_axis == 1:
#             slice_2d = volume[:, slice_idx, :]
#         else:  # default: 2
#             slice_2d = volume[:, :, slice_idx]
#
#         return slice_2d
#
#     def __getitem__(self, idx):
#         ct_slice = self.ct_slices[idx % len(self.ct_slices)]
#         mri_slice = self.mri_slices[idx % len(self.mri_slices)]
#
#         # Convert to PIL for compatibility with torchvision transforms
#         ct_img = Image.fromarray(ct_slice, mode='F')
#         mri_img = Image.fromarray(mri_slice, mode='F')
#
#         p99 = np.percentile(mri_slice, 99)
#
#         contrast_transform_mri = T.Compose([
#             T.Resize((256, 256)),
#             T.ToTensor(),
#             T.Lambda(lambda x: torch.clamp(
#                 input=x,
#                 min=torch.tensor(0, dtype=x.dtype, device=x.device),
#                 max=torch.tensor(p99, dtype=x.dtype, device=x.device)
#             )),
#             T.Lambda(lambda x: x / p99),
#         ])
#
#         # Apply transforms
#         ct_img = self.transform_ct(ct_img)
#         mri_img = contrast_transform_mri(mri_img)
#
#         if mri_img.mean() < 0.01:  # 1e-8
#             # Sort out bad training data
#             return None
#
#         return ct_img, mri_img

if __name__ == '__main__':
    def compute_histogram(tensor, bins=11, range_min=0, range_max=1.1):
        """Compute histogram of tensor values for visualization."""
        tensor = tensor.detach().cpu().numpy().flatten()
        hist, bin_edges = np.histogram(tensor, bins=bins, range=(range_min, range_max))
        return hist, bin_edges

    train_dataset = CtMri2DDataset(
        ct_dir="../files/ct/",
        mri_dir="../files/mri/",
        slice_axis=2
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )

    for i, (real_ct, real_mri) in enumerate(train_dataloader):
        # print(f"Batch {i}")
        # print(f"CT shape: {real_ct.shape}, range: [{real_ct.min():.3f}, {real_ct.max():.3f}], mean: {real_ct.mean():.3f}")
        # print(f"MRI shape: {real_mri.shape}, range: [{real_mri.min():.3f}, {real_mri.max():.3f}], mean: {real_mri.mean():.3f}")
        #
        # # Print every mean for every item in batch mri
        # for mm, m in enumerate(real_mri):
        #     print(f"{mm}: {m.mean():.3f}")

        # print(real_mri.max(), real_mri.min(), len(real_mri), real_mri.mean())
        # hist, bin_edges = compute_histogram(real_mri, bins=11, range_min=0, range_max=1.1)
        # all_count = sum(hist)
        # for ii, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
        #     print(f"Bin {ii}: {edge:.2f} → {count / all_count:.2f}")
        #
        # # vutils.save_image(
        # #     real_mri,
        # #     f"mri_train_slice{i}.png",
        # #     normalize=True
        # # )
        #
        # if i > 3:  # Check first 5 batches only
        #     break

        # Save slices as PNG
        vutils.save_image(
            real_mri,
            os.path.join("fake", f"fakeMRI_{i:04d}.png"),
            normalize=True
        )
    # Assuming you have these utility functions
    png_slices_to_nifti("fake", "nif")
