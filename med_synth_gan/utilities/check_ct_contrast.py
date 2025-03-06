import os
import torch
import torchvision.transforms as T
import SimpleITK as sitk
import numpy as np
from PIL import Image


if __name__ == "__main__":
    ct_dir = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/"

    output_dir = "ct_scans_export/"  # Replace with your output folder

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define transformation pipeline
    contrast_transform_ct = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Lambda(lambda x: torch.clamp(x, -200, 500)),  # Clamp to HU range
        T.Lambda(lambda x: (x + 200) / 700),  # Normalize to [0,1]
    ])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each NIfTI file in ct_dir
    for filename in os.listdir(ct_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            file_path = os.path.join(ct_dir, filename)

            # Read NIfTI file
            img = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(img)  # Shape: (slices, H, W)

            # Ensure there are at least 3 slices
            if array.shape[0] < 3:
                print(f"Skipping {filename}: not enough slices.")
                continue

            # Get the 3rd slice (index 2)
            slice_3rd = array[35]

            # Convert to PIL Image for transformations
            slice_pil = Image.fromarray(slice_3rd.astype(np.float32))

            # Apply transformations
            transformed_slice = contrast_transform_ct(slice_pil)

            # Convert tensor to PIL image
            transformed_pil = T.ToPILImage()(transformed_slice)

            # Save transformed image
            output_path = os.path.join(output_dir, f"{filename}.png")
            transformed_pil.save(output_path)

            print(f"Processed and saved: {output_path}")

    print("Processing complete!")
