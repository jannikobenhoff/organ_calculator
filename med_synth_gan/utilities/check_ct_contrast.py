import os
import torch
import torchvision.transforms as T
import SimpleITK as sitk
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

    def create_grid(images, grid_size=(3, 3)):
        """
        Arrange a list of images into a grid.
        images: List of PIL images
        grid_size: Tuple (rows, cols)
        """
        rows, cols = grid_size
        w, h = images[0].size  # Get size of single image

        grid_img = Image.new("L", (cols * w, rows * h), color=0)  # Create blank grid

        for idx, img in enumerate(images):
            if idx >= rows * cols:  # Stop if we exceed grid capacity
                break
            x = (idx % cols) * w
            y = (idx // cols) * h
            grid_img.paste(img, (x, y))

        return grid_img


    def plot_histogram(image_tensor, filename):
        """
        Plots and saves histogram of intensity values.
        """
        image_array = image_tensor.numpy().flatten()  # Flatten to 1D array

        plt.figure(figsize=(8, 5))
        plt.hist(image_array, bins=50, range=(0, 1), color='blue', alpha=0.7)
        plt.xlabel("Normalized Intensity")
        plt.ylabel("Frequency")
        plt.title(f"Histogram: {filename}")
        plt.grid(True)

        histogram_path = os.path.join(output_dir, f"{filename}_hist.png")
        plt.savefig(histogram_path)
        plt.close()

    # Process each NIfTI file in ct_dir
    for filename in os.listdir(ct_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            file_path = os.path.join(ct_dir, filename)

            # Read NIfTI file
            img = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(img)  # Shape: (slices, H, W)

            # Select every 10th slice
            selected_slices = [array[i] for i in range(0, array.shape[0], 10)]

            # Convert slices to images and apply transformations
            transformed_images = []
            transformed_tensors = []  # Store tensors for histogram
            for slice_data in selected_slices:
                slice_pil = Image.fromarray(slice_data.astype(np.float32))
                transformed_tensor = contrast_transform_ct(slice_pil)
                transformed_pil = T.ToPILImage()(transformed_tensor)

                transformed_images.append(transformed_pil)
                transformed_tensors.append(transformed_tensor)

            if not transformed_images:
                print(f"Skipping {filename}: Not enough slices.")
                continue

            # Determine grid size (e.g., 3x3, 4x4)
            num_slices = len(transformed_images)
            grid_size = (3, 3) if num_slices >= 9 else (2, 2) if num_slices >= 4 else (1, num_slices)

            # Create image grid
            grid_image = create_grid(transformed_images, grid_size)

            # Save grid image
            grid_output_path = os.path.join(output_dir, f"{filename}.png")
            grid_image.save(grid_output_path)

            # Flatten all transformed slices into one array for histogram
            full_tensor = torch.cat([t.flatten() for t in transformed_tensors])
            plot_histogram(full_tensor, filename)

    print("Processing complete!", flush=True)
