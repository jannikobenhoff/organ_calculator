import os
import glob
import nibabel as nib
import numpy as np
from tifffile import enumarg
from torch.utils.data import Dataset
from PIL import Image
from med_synth_gan.dataset.utils import contrast_transform_ct, normalize_mri
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import glob
import os
import numpy as np
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch


class CtMri2DDataset2(Dataset):
    """
    Loads a list of 3D NIfTI volumes (CT and MRI), extracts 2D slices,
    and returns them as single images for training.
    """

    def __init__(
            self,
            ct_dir,  # Directory containing CT NIfTI files
            mri_dir,  # Directory containing MRI NIfTI files
            slice_axis=2,  # 0 -> axial slices across x, 1 -> across y, 2 -> across z
    ):
        super().__init__()

        self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
        self.mri_vol_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))
        self.transform_ct = contrast_transform_ct
        self.slice_axis = slice_axis

        # Preload volumes and create a list of slices for CT
        self.ct_slices = self._load_slices(self.ct_vol_paths, is_ct=True)

        self.mri_p99_dict = {}  # Store p99 values for each MRI file
        self.slice_to_vol_idx = {}  # Map slice index to volume index
        self.mri_slices = []

        for vol_idx, path in enumerate(self.mri_vol_paths):
            vol = nib.load(path).get_fdata(dtype=np.float32)
            self.mri_p99_dict[vol_idx] = np.percentile(vol, 99)

            # Load slices and keep track of which volume they came from
            num_slices = vol.shape[self.slice_axis]
            start_idx = len(self.mri_slices)
            for s in range(num_slices):
                slice_data = self._get_slice(vol, s)
                self.mri_slices.append(slice_data)
                self.slice_to_vol_idx[start_idx + s] = vol_idx

        # Pre-transform MRI slices
        self.mri_slices = self._preprocess_mri_slices()

        # Ensure datasets match in length for unpaired training
        self.dataset_len = max(len(self.ct_slices), len(self.mri_slices))

    def _load_slices(self, volume_paths, is_ct=False):
        """
        Loads all slices from the provided NIfTI volumes and applies transformation if it is a CT scan.
        """
        slices = []
        for path in volume_paths:
            vol = nib.load(path).get_fdata(dtype=np.float32)

            num_slices = vol.shape[self.slice_axis]
            for s in range(num_slices):
                slice_data = self._get_slice(vol, s)
                img = Image.fromarray(slice_data, mode='F')

                # Apply CT transform
                if is_ct:
                    slice_data = self.transform_ct(img)
                else:
                    slice_data = img
                slices.append(slice_data)
        return slices

    def _get_slice(self, volume, slice_idx):
        """
        Extract a 2D slice from a 3D volume given slice_axis and slice_idx.
        """
        if self.slice_axis == 0:
            slice_2d = volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_2d = volume[:, slice_idx, :]
        else:  # default: 2
            slice_2d = volume[:, :, slice_idx]

        return slice_2d

    def _preprocess_mri_slices(self):
        """
        Pre-transforms all MRI slices and stores them.
        """
        transformed_slices = []
        for idx, mri_slice in enumerate(self.mri_slices):
            vol_idx = self.slice_to_vol_idx[idx]
            vol_p99 = self.mri_p99_dict[vol_idx]

            # Convert to PIL for compatibility with torchvision transforms
            mri_img = Image.fromarray(mri_slice, mode='F')

            contrast_transform_mri = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Lambda(lambda x: torch.clamp(
                    input=x,
                    min=torch.tensor(0, dtype=x.dtype, device=x.device),
                    max=torch.tensor(vol_p99, dtype=x.dtype, device=x.device)
                )),
                T.Lambda(lambda x: x / vol_p99),
            ])

            mri_img = contrast_transform_mri(mri_img)
            if mri_img.mean() > 0.01:
                transformed_slices.append(mri_img)
        return transformed_slices

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        ct_slice = self.ct_slices[idx % len(self.ct_slices)]
        mri_slice = self.mri_slices[idx % len(self.mri_slices)]

        if mri_slice.mean() < 0.01:  # 1e-8
            # Sort out bad training data
            return None

        return ct_slice, mri_slice


class CtMri2DDataset(Dataset):
    """
    Loads a list of 3D NIfTI volumes (CT and MRI), extracts 2D slices,
    and returns them as single images for training.
    """
    def __init__(
        self,
        ct_dir,          # Directory containing CT NIfTI files
        mri_dir,         # Directory containing MRI NIfTI files
        slice_axis=2,    # 0 -> axial slices across x, 1 -> across y, 2 -> across z
    ):
        super().__init__()

        self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))

        self.mri_vol_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))

        self.transform_ct = contrast_transform_ct
        # self.transform_mri = normalize_mri

        self.slice_axis = slice_axis

        # Preload volumes and create a list of slices for CT
        self.ct_slices = self._load_slices(self.ct_vol_paths)
        self.mri_slices = self._load_slices(self.mri_vol_paths)

        # Ensure datasets match in length for unpaired training
        self.dataset_len = max(len(self.ct_slices), len(self.mri_slices))

    def _load_slices(self, volume_paths, is_ct=False):
        """
        Loads all slices from the provided NIfTI volumes and applies transformation if it is a CT scan.
        """
        slices = []
        for path in volume_paths:
            vol = nib.load(path).get_fdata(dtype=np.float32)

            num_slices = vol.shape[self.slice_axis]
            for s in range(num_slices):
                slice_data = self._get_slice(vol, s)
                # img = Image.fromarray(slice_data, mode='F')
                #
                # # Apply CT transform
                # if is_ct:
                #     slice_data = self.transform_ct(img)
                # else:
                #     slice_data = img
                slices.append(slice_data)
        return slices

    def __len__(self):
        return self.dataset_len

    def _get_slice(self, volume, slice_idx):
        """
        Extract a 2D slice from a 3D volume given slice_axis and slice_idx.
        """
        if self.slice_axis == 0:
            slice_2d = volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_2d = volume[:, slice_idx, :]
        else:  # default: 2
            slice_2d = volume[:, :, slice_idx]

        return slice_2d

    def __getitem__(self, idx):
        ct_slice = self.ct_slices[idx % len(self.ct_slices)]
        mri_slice = self.mri_slices[idx % len(self.mri_slices)]

        # Convert to PIL for compatibility with torchvision transforms
        ct_img = Image.fromarray(ct_slice, mode='F')
        mri_img = Image.fromarray(mri_slice, mode='F')

        p99 = np.percentile(mri_slice, 99)

        contrast_transform_mri = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Lambda(lambda x: torch.clamp(
                input=x,
                min=torch.tensor(0, dtype=x.dtype, device=x.device),
                max=torch.tensor(p99, dtype=x.dtype, device=x.device)
            )),
            T.Lambda(lambda x: x / p99),
        ])

        # Apply transforms
        ct_img = self.transform_ct(ct_img)
        mri_img = contrast_transform_mri(mri_img)

        if mri_img.mean() < 0.01:  # 1e-8
            # Sort out bad training data
            return None

        return ct_img, mri_img

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
        batch_size=8,
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

        print(real_mri.max(), real_mri.min(), len(real_mri), real_mri.mean())
        hist, bin_edges = compute_histogram(real_mri, bins=11, range_min=0, range_max=1.1)
        all_count = sum(hist)
        for ii, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
            print(f"Bin {ii}: {edge:.2f} → {count / all_count:.2f}")

        # vutils.save_image(
        #     real_mri,
        #     f"mri_train_slice{i}.png",
        #     normalize=True
        # )

        if i > 3:  # Check first 5 batches only
            break
