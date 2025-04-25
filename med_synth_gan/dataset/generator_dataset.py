import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import os

from med_synth_gan.dataset.utils import contrast_transform_ct

class Generator2DDataset(Dataset):
    def __init__(self, ct_dir, slice_axis=2):
        super().__init__()
        self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
        self.slice_axis = slice_axis

        # Load all CT volumes with memory mapping
        self.ct_volumes = [nib.load(path).get_fdata(dtype=np.float32)
                          for path in self.ct_vol_paths]

        # Pre-calculate slice indices for all volumes
        self.ct_slice_indices = []
        for vol_idx, vol in enumerate(self.ct_volumes):
            num_slices = vol.shape[self.slice_axis]
            self.ct_slice_indices.extend([(vol_idx, s) for s in range(num_slices)])

    def _get_slice(self, volume, slice_idx):
        if self.slice_axis == 0:
            slice_ = volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_ = volume[:, slice_idx, :]
        else:
            slice_ = volume[:, :, slice_idx]
        return np.rot90(slice_)

    def __getitem__(self, idx):
        # Get volume and slice indices
        vol_idx, slice_idx = self.ct_slice_indices[idx]

        # Extract and preprocess slice
        ct_slice = self._get_slice(self.ct_volumes[vol_idx], slice_idx)

        # Apply same transforms as training
        ct_img = Image.fromarray(ct_slice, mode='F')
        ct_tensor = contrast_transform_ct(ct_img)

        # Return volume index and slice index for reconstruction
        return ct_tensor, vol_idx, slice_idx

    def __len__(self):
        return len(self.ct_slice_indices)
