import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import itertools
import os
import glob
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import torch
import torchvision.utils as vutils

class Nifti2DDataset(Dataset):
    """
    Loads a list of 3D NIfTI volumes (CT or MRI), extracts 2D slices,
    and returns them as single images for CycleGAN training.
    """
    def __init__(
        self,
        ct_dir,          # Directory containing CT NIfTI files
        mri_dir,         # Directory containing MRI NIfTI files
        transform=None,
        slice_axis=2,    # 0 -> axial slices across x, 1 -> across y, 2 -> across z
        normalize=True
    ):
        super().__init__()
        
        self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
        self.mri_vol_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))
        
        self.transform = transform
        self.slice_axis = slice_axis
        self.normalize = normalize
        
        # Pre-load volumes and create a list of slices for CT
        self.ct_slices = self._load_slices(self.ct_vol_paths)
        self.mri_slices = self._load_slices(self.mri_vol_paths)

        # Ensure datasets match in length for unpaired training
        self.dataset_len = max(len(self.ct_slices), len(self.mri_slices))
    
    def _load_slices(self, volume_paths):
        """
        Loads all slices from the provided NIfTI volumes.
        """
        slices = []
        for path in volume_paths:
            vol = nib.load(path).get_fdata(dtype=np.float32)  
            
            num_slices = vol.shape[self.slice_axis]
            for s in range(num_slices):
                slice_data = self._get_slice(vol, s)
                # Normalize
                if self.normalize:
                    slice_data = self._contrast_normalization(slice_data)
                
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
    
    def _contrast_normalization(self, arr):
        """
        Maps intensities from [-200, 500] to [-1,1].
        """
        arr = np.clip(arr, -200, 500)  # Clamp to the valid range
        arr = (arr - 150.0) / 350.0  # Normalize to [-1, 1]
        return arr.astype(np.float32)
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        ct_slice = self.ct_slices[idx % len(self.ct_slices)]
        mri_slice = self.mri_slices[idx % len(self.mri_slices)]

        # Convert to PIL for compatibility with torchvision transforms
        ct_img = Image.fromarray(ct_slice, mode='F')  
        mri_img = Image.fromarray(mri_slice, mode='F')

        if self.transform:
            ct_img = self.transform(ct_img)  
            mri_img = self.transform(mri_img)

        return ct_img, mri_img
    
    def export_ct_slice_as_png(self, slice_index, output_path="ct_slice.png"):
        """
        Exports a single CT slice as a PNG file.
        :param slice_index: Index of the slice to export
        :param output_path: File path where the PNG will be saved
        """
        if slice_index >= len(self.ct_slices):
            print(f"Error: slice_index {slice_index} out of range. Max: {len(self.ct_slices) - 1}")
            return
        
        # Get the raw slice
        ct_slice = self.ct_slices[slice_index]
        
        # Convert [-1,1] to [0,1] for saving
        image_for_png = 0.5 * (torch.tensor(ct_slice) + 1.0)
        image_for_png = image_for_png.unsqueeze(0)  # [1, H, W] for torchvision

        # Save using torchvision
        vutils.save_image(image_for_png, output_path)
        print(f"Saved CT slice {slice_index} to {output_path}")

# class SingleVolume2DDataset(Dataset):
#     """
#     Loads a single 3D NIfTI volume, extracts 2D slices along `slice_axis`,
#     and returns them as individual items (one per slice).
#     """
#     def __init__(
#         self,
#         volume_path,         # Path to a single .nii or .nii.gz file
#         transform=None,
#         slice_axis=2,        # 0-> x-plane slices, 1-> y-plane, 2-> z-plane
#         min_max_normalize=True
#     ):
#         super().__init__()
#         self.transform = transform
#         self.slice_axis = slice_axis
#         self.min_max_normalize = min_max_normalize
        
#         # Load the 3D volume
#         volume_nifti = nib.load(volume_path)
#         volume = volume_nifti.get_fdata(dtype=np.float32)  # shape [Dx, Dy, Dz], etc.
        
#         # Extract slices
#         num_slices = volume.shape[self.slice_axis]
#         self.slices_data = []
#         for s in range(num_slices):
#             slice_2d = self._get_slice(volume, s)
#             self.slices_data.append(slice_2d)
    
#     def _get_slice(self, volume, slice_idx):
#         """
#         Extract a 2D slice from a 3D volume given `slice_axis` and slice index.
#         """
#         if self.slice_axis == 0:
#             return volume[slice_idx, :, :]
#         elif self.slice_axis == 1:
#             return volume[:, slice_idx, :]
#         else:  # default: 2
#             return volume[:, :, slice_idx]
    
#     def __len__(self):
#         return len(self.slices_data)
    
#     def __getitem__(self, idx):
#         slice_2d = self.slices_data[idx]
        
#         # (Optionally) min-max normalize this slice
#         if self.min_max_normalize:
#             slice_2d = self._min_max_norm(slice_2d)
        
#         # Convert to PIL Image in floating mode (32-bit)
#         # so that torchvision transforms can be applied
#         slice_img = Image.fromarray(slice_2d, mode='F')
        
#         # Apply any transform (e.g. Resize, ToTensor, Normalize, etc.)
#         if self.transform is not None:
#             slice_img = self.transform(slice_img)
        
#         # Return it as a tuple (for consistency with usage: `for ct_slice, in loader`)
#         return (slice_img, )
    
#     def _min_max_norm(self, arr):
#         min_val = np.min(arr)
#         max_val = np.max(arr)
#         if max_val - min_val < 1e-5:
#             # Avoid division by zero if slice is uniform
#             return np.zeros_like(arr, dtype=np.float32)
#         return (arr - min_val) / (max_val - min_val)  # scale to [0, 1]