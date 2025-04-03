import os
import glob
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from med_synth_gan.dataset.utils import contrast_transform_ct
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import random


class CtMri2DDataset(Dataset):
    def __init__(self, ct_dir, mri_dir, slice_axis=2):
        super().__init__()
        self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
        self.mri_vol_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))

        # Reduce ct data set sice to balance both
        self.ct_vol_paths = self.ct_vol_paths[:50]

        self.slice_axis = slice_axis

        # Use memory mapping for NIfTI files
        self.ct_volumes = [nib.load(path).get_fdata(dtype=np.float32) #, caching='unchanged')
                           for path in self.ct_vol_paths]
        self.mri_volumes = [nib.load(path).get_fdata(dtype=np.float32) #, caching='unchanged')
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

    def _get_slice(self, volume, slice_idx, mri=False):
        if self.slice_axis == 0:
            slice_ = volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_ = volume[:, slice_idx, :]
        else:
            slice_ = volume[:, :, slice_idx]

        rotated =  np.rot90(slice_)

        if mri:
            return np.fliplr(rotated)
        else:
            return rotated

    def __getitem__(self, idx):
        # random CT slice
        ct_vol_idx, ct_slice_idx = random.choice(self.ct_slice_indices)
        ct_slice = self._get_slice(self.ct_volumes[ct_vol_idx], ct_slice_idx, mri=False)
        ct_img = Image.fromarray(ct_slice, mode='F')
        ct_tensor = contrast_transform_ct(ct_img)

        # random MRI slice
        mri_vol_idx, mri_slice_idx = random.choice(self.mri_slice_indices)
        mri_slice = self._get_slice(self.mri_volumes[mri_vol_idx], mri_slice_idx, mri=True)
        mri_img = Image.fromarray(mri_slice, mode='F')

        # Resize MRI
        mri_tensor = self.contrast_transform_mri(mri_img)

        # Normalize
        p99 = self.mri_stats[mri_vol_idx]['p99']
        # p1 = self.mri_stats[mri_vol_idx]['p1']
        mri_tensor = torch.clamp(mri_tensor, min=0, max=p99)
        mri_tensor = (mri_tensor - 0) / (p99 - 0)

        if mri_tensor.mean() < 0.01:
            return None  # Optionally skip blank slices

        return ct_tensor, mri_tensor

    def __len__(self):
        return self.dataset_len

if __name__ == '__main__':
    def compute_histogram(tensor, bins=11, range_min=0, range_max=1.1):
        """Compute histogram of tensor values for visualization."""
        tensor = tensor.detach().cpu().numpy().flatten()
        hist, bin_edges = np.histogram(tensor, bins=bins, range=(range_min, range_max))
        return hist, bin_edges

    train_dataset = CtMri2DDataset(
        ct_dir="../files/ct/",
        #ct_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/",
        #mri_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/t2Axial/",
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
            real_ct,
            f"CT_{i:04d}.png",
            normalize=True
        )
        break
    # Assuming you have these utility functions
    #png_slices_to_nifti("fake", "nif")
