import os
import glob
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from med_synth_gan.dataset.utils import contrast_transform


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
        
        self.transform = contrast_transform
        self.slice_axis = slice_axis
        
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
    
    # def export_ct_slice_as_png(self, slice_index, output_path="ct_slice.png"):
    #     """
    #     Exports a single CT slice as a PNG file.
    #     :param slice_index: Index of the slice to export
    #     :param output_path: File path where the PNG will be saved
    #     """
    #     if slice_index >= len(self.ct_slices):
    #         print(f"Error: slice_index {slice_index} out of range. Max: {len(self.ct_slices) - 1}")
    #         return
        
    #     # Get the raw slice
    #     ct_slice = self.ct_slices[slice_index]
        
    #     # Convert [-1,1] to [0,1] for saving
    #     image_for_png = 0.5 * (torch.tensor(ct_slice) + 1.0)
    #     image_for_png = image_for_png.unsqueeze(0)  # [1, H, W] for torchvision

    #     # Save using torchvision
    #     vutils.save_image(image_for_png, output_path)
    #     print(f"Saved CT slice {slice_index} to {output_path}")
