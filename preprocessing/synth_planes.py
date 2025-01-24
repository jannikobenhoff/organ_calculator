import nibabel as nib
import numpy as np

# Load the NIfTI file
img = nib.load('/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/imagesTr/AMOS_MR_2022_000001_0000.nii.gz')
data = img.get_fdata()

# Get the shape of your volume
print(f"Volume shape: {data.shape}")
# This will show you (X, Y, Z) dimensions

# For example, if shape is (256, 256, 180):
# X dimension (L): 256 slices for sagittal
# Y dimension (I): 256 slices for axial
# Z dimension (P): 180 slices for coronal

# You can choose middle slices as default:
sagittal_middle = data.shape[0] // 2  # Middle slice for L dimension
axial_middle = data.shape[1] // 2     # Middle slice for I dimension
coronal_middle = data.shape[2] // 2   # Middle slice for P dimension

# Extract middle slices
axial_slice = data[:, axial_middle, :]      # Middle axial slice
coronal_slice = data[:, :, coronal_middle]  # Middle coronal slice
sagittal_slice = data[sagittal_middle, :, :] # Middle sagittal slice

print(f"Shape of axial slice: {axial_slice.shape}")
print(f"Shape of coronal slice: {coronal_slice.shape}")
print(f"Shape of sagittal slice: {sagittal_slice.shape}")