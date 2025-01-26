import nibabel as nib
import numpy as np
import os
from PIL import Image

def reassemble_3D(image_folder, volume_shape, slice_axis=2, out_nii="fake_mri_3D.nii.gz"):
    """
    Reconstruct a 3D volume from the saved PNG slices.
    volume_shape = (Dx, Dy, Dz) or (H, W, #slices)
    slice_axis: 2 for axial, etc.
    """
    # 1. Gather slice file paths in sorted order
    slice_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    
    # 2. Create empty array
    vol_3d = np.zeros(volume_shape, dtype=np.float32)
    
    for i, sf in enumerate(slice_files):
        path = os.path.join(image_folder, sf)
        # Load PNG
        slice_img = Image.open(path).convert('F')  # 32-bit float
        slice_arr = np.array(slice_img)
        
        # Insert into vol_3d
        if slice_axis == 2:
            vol_3d[:, :, i] = slice_arr
        elif slice_axis == 1:
            vol_3d[:, i, :] = slice_arr
        else:  # slice_axis == 0
            vol_3d[i, :, :] = slice_arr
    
    # 3. Save as NIfTI
    nifti_img = nib.Nifti1Image(vol_3d, affine=np.eye(4))
    nib.save(nifti_img, out_nii)
    print(f"Saved 3D volume to {out_nii}")
