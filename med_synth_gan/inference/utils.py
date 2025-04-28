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
import numpy as np
import nibabel as nib
import glob
import os
from PIL import Image

def png_slices_to_nifti(png_folder, output_nifti, reference_nifti_path):
    """
    Manually read sorted PNG slices into a 3D volume and save with original affine/header.
    """
    # Read sorted PNG slices manually
    slice_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
    if not slice_files:
        raise ValueError(f"No PNG files found in folder: {png_folder}")

    slices = [np.array(Image.open(s).convert('F')) for s in slice_files]

    # Stack slices along the 3rd dimension (correct for axial)
    volume_array = np.stack(slices, axis=-1).astype(np.float32)

    # Load original affine and header from CT
    original_nifti = nib.load(reference_nifti_path)
    affine = original_nifti.affine
    header = original_nifti.header.copy()

    # Ensure header dimensions match new volume
    header.set_data_shape(volume_array.shape)

    # Save as new NIfTI file with original affine
    nib.save(nib.Nifti1Image(volume_array, affine, header), output_nifti)
    print(f"Saved 3D NIfTI volume to: {output_nifti}")
