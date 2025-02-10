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

        self.mri_p99_dict = {}  # Store p99 values for each MRI file
        self.slice_to_vol_idx = {}  # Map slice index to volume index
        self.mri_slices = []

        for vol_idx, path in enumerate(self.mri_vol_paths):
            vol = nib.load(path).get_fdata(dtype=np.float32)
            self.mri_p99_dict[vol_idx] = np.percentile(vol, 95)

            # Load slices and keep track of which volume they came from
            num_slices = vol.shape[self.slice_axis]
            start_idx = len(self.mri_slices)
            for s in range(num_slices):
                slice_data = self._get_slice(vol, s)
                self.mri_slices.append(slice_data)
                self.slice_to_vol_idx[start_idx + s] = vol_idx

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
        mri_idx = idx % len(self.mri_slices)
        mri_slice = self.mri_slices[mri_idx]

        vol_idx = self.slice_to_vol_idx[mri_idx]
        vol_p99 = self.mri_p99_dict[vol_idx]

        # Convert to PIL for compatibility with torchvision transforms
        ct_img = Image.fromarray(ct_slice, mode='F')
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

        # Apply transforms
        ct_img = self.transform_ct(ct_img)
        mri_img = contrast_transform_mri(mri_img)

        if mri_img.mean() < 1e-8:  # Adjust threshold as needed
            return None

        return ct_img, mri_img

if __name__ == '__main__':
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
        print(f"Batch {i}")
        print(f"CT shape: {real_ct.shape}, range: [{real_ct.min():.3f}, {real_ct.max():.3f}], mean: {real_ct.mean():.3f}")
        print(f"MRI shape: {real_mri.shape}, range: [{real_mri.min():.3f}, {real_mri.max():.3f}], mean: {real_mri.mean():.3f}")

        # Print every mean for every item in batch mri
        for mm, m in enumerate(real_mri):
            print(f"{mm}: {m.mean():.3f}")

        vutils.save_image(
            real_mri,
            f"mri_train_slice{i}.png",
            normalize=True
        )

        if i > 5:  # Check first 5 batches only
            break
