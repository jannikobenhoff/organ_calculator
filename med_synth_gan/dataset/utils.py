import torchvision.transforms as T
import torch
import numpy as np
import torch
import torchvision.transforms as T
import nibabel as nib
import torchvision.utils as vutils
from PIL import Image
import torch.nn.functional as F

contrast_transform_ct = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Lambda(lambda x: torch.clamp(x, -200, 500)),
        T.Lambda(lambda x: (x+200) / 700),
])

contrast_transform_mri = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Lambda(lambda x: torch.clamp(x, 0, 500)),
        T.Lambda(lambda x: x / 500),
])


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Return a *new* NIfTI re‑oriented to RAS (Right‑Anterior‑Superior).
    Data array is copied; affine is updated accordingly.
    """
    ornt_current = nib.orientations.io_orientation(img.affine)
    ornt_ras     = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform    = nib.orientations.ornt_transform(ornt_current, ornt_ras)

    data_ras     = nib.orientations.apply_orientation(img.get_fdata(), transform)
    affine_ras   = img.affine @ nib.orientations.inv_ornt_aff(transform, img.shape)

    return nib.Nifti1Image(data_ras, affine_ras, header=img.header.copy())

def load_and_resample(nib_img, size, order=1):
    """Read full volume → resample to `size` with trilinear (order=1)."""
    vol = nib_img.get_fdata(dtype=np.float32)  # (Z,Y,X)
    # bring to (1, D, H, W) for F.interpolate
    vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # 1×1×Z×Y×X
    vol = F.interpolate(vol, size=size, mode="trilinear",
                        align_corners=False, antialias=False)
    return vol[0]  # drop batch dim → 1×D×H×W

def orient_ct(vol: torch.Tensor) -> torch.Tensor:
    """
    vol: 1×D×H×W  → rotate 90° CCW on each axial slice.
    Returns 1×D×W×H (H and W swapped – fine for the network).
    """
    return torch.rot90(vol, k=1, dims=[-2, -1])


def orient_mri(vol: torch.Tensor) -> torch.Tensor:
    """
    Same as orient_ct plus left-right flip ⇒ match 2-D rule.
    Output shape: 1×D×W×H
    """
    vol = torch.rot90(vol, k=1, dims=[-2, -1])
    return torch.flip(vol, dims=[-1])

def contrast_transform_ct_3d(x: torch.Tensor, out_size=(256,256,96)):
    """
    Args
    ----
    x : 1×D×H×W tensor (float32, HU)
    Returns
    -------
    1×D×H×W tensor in [0,1]
    """
    # resize (tri-linear) if requested
    if x.shape[1:] != out_size:
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=out_size,
            mode="trilinear", align_corners=False
        )[0]

    x = torch.clamp(x, -200, 500)          # window
    x = (x + 200) / 700                    # scale to 0-1
    return x

def normalize_mri(input_image):
        numpydata = np.asarray(input_image)
        p1, p99 = np.percentile(numpydata, [1, 95])

        contrast_transform_mri = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Lambda(lambda x: torch.clamp(x, 0, p99)),
                T.Lambda(lambda x: x / p99),
        ])

        return contrast_transform_mri(input_image)


# def histogram_normalization_mri(image_tensor, min_percentile=1, max_percentile=99):
#     """
#     Normalizes MRI images using histogram-based normalization.
#     - Clips intensities to a dynamic range based on percentiles.
#     - Scales values to [0,1].
#     """
#     image_np = image_tensor.numpy()
#
#     # Compute percentiles
#     p1, p99 = np.percentile(image_np, [min_percentile, max_percentile])
#
#     # Clip intensities dynamically
#     image_np = np.clip(image_np, p1, p99)
#
#     # Min-max normalization to [0,1]
#     image_np = (image_np - p1) / (p99 - p1)
#
#     return torch.tensor(image_np, dtype=torch.float32)
#
#
# # Define a transform pipeline
# contrast_transform_mri = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor(),
#     T.Lambda(lambda x: histogram_normalization_mri(x)),
# ])


if __name__ == '__main__':
        volume_nifti = nib.load("../files/mri/AMOS_MR_2022_000011_0000.nii.gz").get_fdata(dtype=np.float32)

        num_slices = volume_nifti.shape[2]
        for s in range(50, num_slices):
                slice_data = volume_nifti[:, :, s]
                mri_img = Image.fromarray(slice_data, mode='F')
                numpydata = np.asarray(mri_img)

                print(np.mean(numpydata))
                print(np.min(numpydata))
                print(np.max(numpydata))
                p1, p99 = np.percentile(numpydata, [1, 99])
                print(p1, p99)
                transformed_slice = normalize_mri(mri_img)

                vutils.save_image(
                        transformed_slice,
                        f"slice.png",
                        normalize=True
                )


                break
