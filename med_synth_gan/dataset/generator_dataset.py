import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from med_synth_gan.dataset.utils import contrast_transform_ct

class Generator2DDataset(Dataset):
    """
    Loads multiple 3D NIfTI volumes, extracts 2D slices along `slice_axis`,
    and returns them with volume indices for reconstruction.
    """

    def __init__(self, volume_paths, slice_axis=2):
        """
        :param volume_paths: List of paths to NIfTI files
        :param slice_axis: Axis along which to extract slices (0=X, 1=Y, 2=Z)
        """
        super().__init__()
        self.transform = contrast_transform_ct
        self.slice_axis = slice_axis
        self.slices = []
        self.volume_indices = []
        self.volume_shapes = []

        # Load all volumes and extract slices
        for vol_idx, path in enumerate(volume_paths):
            volume_nifti = nib.load(path)
            volume_data = volume_nifti.get_fdata(dtype=np.float32)
            self.volume_shapes.append(volume_data.shape)

            num_slices = volume_data.shape[self.slice_axis]
            for slice_idx in range(num_slices):
                self.slices.append(self._get_slice(volume_data, slice_idx))
                self.volume_indices.append(vol_idx)

    def _get_slice(self, volume, slice_idx):
        """Extract and preprocess single slice from 3D volume"""
        if self.slice_axis == 0:
            slice_2d = volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_2d = volume[:, slice_idx, :]
        else:
            slice_2d = volume[:, :, slice_idx]

        return np.rot90(slice_2d)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (transformed_slice, volume_index, slice_index_in_volume)
        """
        slice_2d = self.slices[idx]
        vol_idx = self.volume_indices[idx]

        # Convert to PIL Image and apply transforms
        slice_img = Image.fromarray(slice_2d, mode='F')
        if self.transform:
            slice_img = self.transform(slice_img)

        # Get original slice index within volume
        vol_start = self.volume_indices.index(vol_idx)
        slice_in_vol = idx - vol_start

        return slice_img, vol_idx, slice_in_vol
