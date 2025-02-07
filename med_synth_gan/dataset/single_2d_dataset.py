import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from med_synth_gan.dataset.utils import contrast_transform

class SingleVolume2DDataset(Dataset):
    """
    Loads a single 3D NIfTI volume, extracts 2D slices along `slice_axis`,
    and returns them as individual 2D images for inference.
    """
    def __init__(self, volume_path, slice_axis=2):
        """
        :param volume_path: Path to the NIfTI file.
        :param transform: Optional torchvision transforms to apply to each 2D slice.
        :param slice_axis: Axis along which to extract slices (0=X, 1=Y, 2=Z).
        :param apply_contrast_norm: If True, applies contrast normalization [-200, 500] → [-1, 1].
        """
        super().__init__()
        self.transform = contrast_transform
        self.slice_axis = slice_axis

        # Load 3D volume
        volume_nifti = nib.load(volume_path)
        self.volume = volume_nifti.get_fdata(dtype=np.float32)  # Shape: (X, Y, Z) or (H, W, D)

        # Extract slices
        self.slices = [self._get_slice(i) for i in range(self.volume.shape[self.slice_axis])]

    def _get_slice(self, slice_idx):
        """
        Extracts a single 2D slice from the 3D volume along the specified axis.
        """
        if self.slice_axis == 0:
            slice_2d = self.volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_2d = self.volume[:, slice_idx, :]
        else:
            slice_2d = self.volume[:, :, slice_idx]  # Default: axial (Z-axis)

        return slice_2d

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """
        Returns a single 2D slice from the volume, transformed for PyTorch inference.
        """
        slice_2d = self.slices[idx]

        # Convert to PIL Image (required for torchvision transforms)
        slice_img = Image.fromarray(slice_2d, mode='F')

        # Apply transforms
        if self.transform:
            slice_img = self.transform(slice_img)

        return (slice_img,)