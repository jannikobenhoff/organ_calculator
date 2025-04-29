import torchvision.utils as vutils

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
        mid = real_ct.shape[2] // 2  # D // 2
        ct_mid = real_ct[0, :, mid, :, :]  # 1 × H × W
        mri_mid = real_mri[0, :, mid, :, :]  # 1 × H × W

        vutils.save_image(mri_mid,
                          f"mri_train_slice{step}.png",
                          normalize=True)
        vutils.save_image(ct_mid,
                          f"ct_train_slice{step}.png",
                          normalize=True)