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