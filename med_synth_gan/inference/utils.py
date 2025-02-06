import os
import glob
import SimpleITK as sitk


def png_slices_to_nifti(png_folder, output_nifti):
    """
    Reads a sorted list of PNG files in `png_folder` and
    assembles them into a 3D volume, saved as `output_nifti`.
    """
    slice_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
    if not slice_files:
        raise ValueError(f"No PNG files found in folder: {png_folder}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(slice_files)

    volume = reader.Execute()
    sitk.WriteImage(volume, output_nifti)
    print(f"Saved 3D volume to: {output_nifti}")