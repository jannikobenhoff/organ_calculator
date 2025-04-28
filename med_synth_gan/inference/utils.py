import os
import glob
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from PIL import Image

# def png_slices_to_nifti(png_folder, output_nifti):
#     """
#     Reads a sorted list of PNG files in `png_folder` and
#     assembles them into a 3D volume, saved as `output_nifti`.
#     """
#     slice_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
#     if not slice_files:
#         raise ValueError(f"No PNG files found in folder: {png_folder}")
#
#     reader = sitk.ImageSeriesReader()
#     reader.SetFileNames(slice_files)
#
#     volume = reader.Execute()
#     sitk.WriteImage(volume, output_nifti)
#     print(f"Saved 3D volume to: {output_nifti}")

def png_slices_to_nifti(png_folder, output_nifti, reference_nifti_path):
    slice_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
    if not slice_files:
        raise ValueError(f"No PNG files found in folder: {png_folder}")

    slices = [np.array(Image.open(s).convert('F')) for s in slice_files]

    # Stack slices into a 3D numpy array
    volume_array = np.stack(slices, axis=-1).astype(np.float32)

    # Rotate axial slices 90 degrees clockwise
    volume_array = np.rot90(volume_array, k=-1, axes=(0, 1))

    # Switch coronal and sagittal axes
    volume_array = np.transpose(volume_array, (1, 0, 2))

    # Load original affine/header from CT reference
    original_nifti = nib.load(reference_nifti_path)
    affine = original_nifti.affine
    header = original_nifti.header.copy()
    header.set_data_shape(volume_array.shape)

    nib.save(nib.Nifti1Image(volume_array, affine, header), output_nifti)
    print(f"Saved rotated and adjusted NIfTI volume to: {output_nifti}")

