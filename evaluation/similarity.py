import nibabel as nib
import numpy as np
import os
from scipy.ndimage import zoom
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


def calculate_similarity(image_path1, image_path2):
    # Load the NIfTI images
    image1 = nib.load(image_path1)
    image2 = nib.load(image_path2)

    # Get the image data as numpy arrays
    data1 = image1.get_fdata()
    data2 = image2.get_fdata()

    # Check if the shapes of the images are the same
    if data1.shape != data2.shape:
        return 0

    # Calculate the similarity metric (e.g., mean squared error)
    similarity = np.mean((data1 - data2) ** 2)

    return similarity
def compare_medical_images(data1, data2):
    """
    Compare two 3D medical images of different shapes
    Returns multiple similarity metrics
    """
    # Resize second image to match first image's dimensions
    zoom_factors = [data1.shape[i] / data2.shape[i] for i in range(3)]
    data2_resized = zoom(data2, zoom_factors, order=3)  # order=3 for cubic interpolation
    
    # Normalize intensity values to 0-1 range
    data1_norm = (data1 - data1.min()) / (data1.max() - data1.min())
    data2_norm = (data2_resized - data2_resized.min()) / (data2_resized.max() - data2_resized.min())
    
    # Calculate different similarity metrics
    metrics = {}
    
    # 1. Mean Squared Error
    metrics['mse'] = np.mean((data1_norm - data2_norm) ** 2)
    
    # 2. Normalized Cross Correlation
    metrics['ncc'] = np.corrcoef(data1_norm.flatten(), data2_norm.flatten())[0, 1]
    
    # 3. Mutual Information
    # metrics['mi'] = calculate_mutual_information(data1_norm, data2_norm)
    
    # 4. Structural Similarity (for each slice)
    metrics['ssim'] = calculate_3d_ssim(data1_norm, data2_norm)
    
    return metrics

def calculate_mutual_information(data1, data2, bins=50):
    """
    Calculate mutual information between two 3D images
    """
    hist_2d, _, _ = np.histogram2d(data1.flatten(), data2.flatten(), bins=bins)
    
    # Calculate marginal histograms
    p_x = np.sum(hist_2d, axis=1)
    p_y = np.sum(hist_2d, axis=0)
    
    # Calculate entropy
    p_xy = hist_2d / float(np.sum(hist_2d))
    p_x = p_x / float(np.sum(p_x))
    p_y = p_y / float(np.sum(p_y))
    
    # Remove zero elements
    nonzero_mask = p_xy > 0
    
    # Calculate mutual information
    mutual_info = np.sum(p_xy[nonzero_mask] * 
                        np.log2(p_xy[nonzero_mask] / 
                        (p_x.reshape(-1, 1) * p_y)))
    
    return mutual_info

def calculate_3d_ssim(data1, data2, window_size=7):
    """
    Calculate SSIM for 3D medical images
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Calculate SSIM for each slice in z-axis
    ssim_values = []
    for z in range(data1.shape[2]):
        slice_ssim = ssim(data1[:,:,z], 
                         data2[:,:,z], 
                         data_range=1.0,
                         win_size=window_size)
        ssim_values.append(slice_ssim)
    
    return np.mean(ssim_values)

# Usage example
def compare_nifti_images(image_path1, image_path2):
    # Load images
    image1 = nib.load(image_path1)
    image2 = nib.load(image_path2)
    
    # Get data
    data1 = image1.get_fdata()
    data2 = image2.get_fdata()
    
    # Compare images
    similarity_metrics = compare_medical_images(data1, data2)
    
    return similarity_metrics



if __name__ == "__main__":
    # Directory containing the images
    image_directory = '../data/totalsegmentator_combined/imagesTr'

    # Path to the inference input image
    inference_input = '../data/inference_input/Abdomen_000_0000.nii.gz'

    # Iterate through the images in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith('.nii.gz'):
            image_path = os.path.join(image_directory, filename)
            similarity_score = compare_nifti_images(image_path, inference_input)
            if similarity_score['ssim'] > 0.85:
                print(f"Similarity score for {filename}: {similarity_score}")