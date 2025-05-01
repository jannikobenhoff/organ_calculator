import torchvision.utils as vutils
import os
import glob
import nibabel as nib
import numpy as np

def save_debug_images(real_ct, real_mri, step, dim):
    """
    real_ct / real_mri : 4-D (2-D mode) or 5-D (3-D mode) tensors
    """
    if dim == "2d":
        vutils.save_image(real_mri,
                          f"mri_train_slice{step}.png",
                          normalize=True)
        vutils.save_image(real_ct,
                          f"ct_train_slice{step}.png",
                          normalize=True)
    else:                                     # 3-D → pick middle slice
        # mid = real_ct.shape[4] // 2  # middle slice along axis 4 (W)
        # ct_mid = real_ct[0, :, :, :, mid]  # 1 × H × D   → still C×H×W after squeeze
        # mri_mid = real_mri[0, :, :, :, mid]
        #
        # # ct_mid = torch.rot90(ct_mid, k=1, dims=[1, 2])
        # # mri_mid = torch.rot90(mri_mid, k=1, dims=[1, 2])
        #
        # vutils.save_image(mri_mid,
        #                   f"mri_3dtrain_slice{step}.png",
        #                   normalize=True)
        # vutils.save_image(ct_mid,
        #                   f"ct_3dtrain_slice{step}.png",
        #                   normalize=True)

        # New
        mid = real_ct.shape[-1] // 2  # W axis centre

        # slice →   1 × C × D × H
        ct_mid = real_ct[0, :, :, :, mid]
        mri_mid = real_mri[0, :, :, :, mid]

        # permute to C × H × W  (W = D here)
        #ct_mid = ct_mid.permute(0, 2, 1)
        #mri_mid = mri_mid.permute(0, 2, 1)

        vutils.save_image(mri_mid,
                          f"mri_3dtrain_slice{step}.png",
                          normalize=True)
        vutils.save_image(ct_mid,
                          f"ct_3dtrain_slice{step}.png",
                          normalize=True)


if __name__ == "__main__":
    ct_dir = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/"
    mri_dir = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/t2Axial/"

    ct_paths = sorted(glob.glob(os.path.join(ct_dir, '*.nii*')))
    mri_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))

    ct_imgs = [nib.load(p) for p in ct_paths[:5]]
    mri_imgs = [nib.load(p) for p in mri_paths[:5]]

    print("CT:", flush=True)
    for ct in ct_imgs:
        volume = ct.get_fdata(dtype=np.float32)

        original_shape = volume.shape
        print("Shape:", original_shape, flush=True)
        print("Orientation:", ct.affine, flush=True)

    print("MRI:", flush=True)
    for mri in mri_imgs:
        volume = mri.get_fdata(dtype=np.float32)
        original_shape = volume.shape
        print("Shape:", original_shape, flush=True)
        print("Orientation:", mri.affine, flush=True)