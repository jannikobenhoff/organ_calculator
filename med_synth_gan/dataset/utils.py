import torchvision.transforms as T
import torch
import numpy as np
import torch
import torchvision.transforms as T
import nibabel as nib
import torchvision.utils as vutils
from PIL import Image

contrast_transform_ct = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Lambda(lambda x: torch.clamp(x, -200, 500)),
        T.Lambda(lambda x: (x+200) / 700),
])



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
