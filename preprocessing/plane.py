import nibabel as nib

# Load the NIfTI file
img = nib.load('/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/imagesTr/AMOS_MR_2022_000001_0000.nii.gz')
# Get orientation information
orientation = nib.orientations.aff2axcodes(img.affine)
