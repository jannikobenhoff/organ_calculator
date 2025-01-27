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
        min_max_normalize=True
    ):
        super().__init__()
        
        self.ct_vol_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
        self.mri_vol_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))
        
        self.transform = transform
        self.slice_axis = slice_axis
        self.min_max_normalize = min_max_normalize
        
        # Pre-load volumes and create a list of slices for CT
        self.ct_slices = []   # Will hold (slice_array, volume_index, slice_index)
        for ct_path in self.ct_vol_paths:
            vol = nib.load(ct_path).get_fdata(dtype=np.float32)  # [Dx, Dy, Dz]
            # Optional: pre-clean or clamp intensities, if needed (CT can have outliers)
            
            # Collect slices along chosen axis
            num_slices = vol.shape[self.slice_axis]
            for s in range(num_slices):
                # Extract the slice
                slice_data = self._get_slice(vol, s)
                
                # You can skip empty or nearly empty slices if you want
                # if np.mean(slice_data) < some_threshold:
                #     continue
                
                self.ct_slices.append(slice_data)
        
        # Similarly for MRI
        self.mri_slices = []
        for mri_path in self.mri_vol_paths:
            vol = nib.load(mri_path).get_fdata(dtype=np.float32)
            num_slices = vol.shape[self.slice_axis]
            for s in range(num_slices):
                slice_data = self._get_slice(vol, s)
                self.mri_slices.append(slice_data)
        
        # For unpaired training, we typically cycle over the larger set
        self.dataset_len = max(len(self.ct_slices), len(self.mri_slices))
        
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
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        # Repeat (loop) over slices if idx exceeds the smaller domain
        ct_slice = self.ct_slices[idx % len(self.ct_slices)]
        mri_slice = self.mri_slices[idx % len(self.mri_slices)]
        
        # Optional: min-max normalization or z-score
        if self.min_max_normalize:
            ct_slice = self._min_max_norm(ct_slice)
            mri_slice = self._min_max_norm(mri_slice)
        
        # Convert to PIL Images to use standard torchvision transforms
        # (Or you can handle them as torch Tensors directly)
        ct_img = Image.fromarray(ct_slice, mode='F')  # 'F' = 32-bit floating point
        mri_img = Image.fromarray(mri_slice, mode='F')
        
        # Apply transforms
        if self.transform is not None:
            ct_img = self.transform(ct_img)   # -> Tensor [C, H, W]
            mri_img = self.transform(mri_img) # -> Tensor [C, H, W]
        
        return ct_img, mri_img
    
    def _min_max_norm(self, arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-5:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - min_val) / (max_val - min_val)  # scale to [0, 1]

class SingleVolume2DDataset(Dataset):
    """
    Loads a single 3D NIfTI volume, extracts 2D slices along `slice_axis`,
    and returns them as individual items (one per slice).
    """
    def __init__(
        self,
        volume_path,         # Path to a single .nii or .nii.gz file
        transform=None,
        slice_axis=2,        # 0-> x-plane slices, 1-> y-plane, 2-> z-plane
        min_max_normalize=True
    ):
        super().__init__()
        self.transform = transform
        self.slice_axis = slice_axis
        self.min_max_normalize = min_max_normalize
        
        # Load the 3D volume
        volume_nifti = nib.load(volume_path)
        volume = volume_nifti.get_fdata(dtype=np.float32)  # shape [Dx, Dy, Dz], etc.
        
        # Extract slices
        num_slices = volume.shape[self.slice_axis]
        self.slices_data = []
        for s in range(num_slices):
            slice_2d = self._get_slice(volume, s)
            self.slices_data.append(slice_2d)
    
    def _get_slice(self, volume, slice_idx):
        """
        Extract a 2D slice from a 3D volume given `slice_axis` and slice index.
        """
        if self.slice_axis == 0:
            return volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            return volume[:, slice_idx, :]
        else:  # default: 2
            return volume[:, :, slice_idx]
    
    def __len__(self):
        return len(self.slices_data)
    
    def __getitem__(self, idx):
        slice_2d = self.slices_data[idx]
        
        # (Optionally) min-max normalize this slice
        if self.min_max_normalize:
            slice_2d = self._min_max_norm(slice_2d)
        
        # Convert to PIL Image in floating mode (32-bit)
        # so that torchvision transforms can be applied
        slice_img = Image.fromarray(slice_2d, mode='F')
        
        # Apply any transform (e.g. Resize, ToTensor, Normalize, etc.)
        if self.transform is not None:
            slice_img = self.transform(slice_img)
        
        # Return it as a tuple (for consistency with usage: `for ct_slice, in loader`)
        return (slice_img, )
    
    def _min_max_norm(self, arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-5:
            # Avoid division by zero if slice is uniform
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - min_val) / (max_val - min_val)  # scale to [0, 1]