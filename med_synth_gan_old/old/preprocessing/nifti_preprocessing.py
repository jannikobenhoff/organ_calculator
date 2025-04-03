import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from utils import create_itksnap_label_file, save_multilabel_nifti


def combine_segmentations(base_dir, output_dir, label_mapping=None):
    """
    Combines individual segmentation files into a single multi-label NIfTI file.
    
    Args:
        base_dir (str): Base directory containing subject folders
        output_dir (str): Directory to save combined segmentations
        label_mapping (dict, optional): Dictionary mapping anatomical parts to label numbers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If no label mapping provided, create one from the first subject
    if label_mapping is None:
        first_subject = next(Path(base_dir).glob('s*/segmentations'))
        unique_parts = set()
        for file in first_subject.glob('*.nii.gz'):
            part_name = file.stem.replace('.nii', '')
            unique_parts.add(part_name)
        
        # Create mapping with background as 0
        label_mapping = {part: idx + 1 for idx, part in enumerate(sorted(unique_parts))}
    
    for subject_dir in tqdm(list(Path(base_dir).glob('s*/segmentations'))):
        subject_id = subject_dir.parent.name
        print(f"\nProcessing {subject_id}...")

        main_ct_img = nib.load(subject_dir.parent / 'ct.nii.gz')

        # Get reference image from first segmentation to get dimensions and affine
        first_seg = next(subject_dir.glob('*.nii.gz'))
        ref_img = nib.load(str(first_seg))
        combined_seg = np.zeros(ref_img.shape, dtype=np.uint16)
        
        for seg_file in subject_dir.glob('*.nii.gz'):
            part_name = seg_file.stem.replace('.nii', '')
            if part_name not in label_mapping:
                print(f"Warning: {part_name} not in label mapping, skipping...")
                continue
                
            label_num = label_mapping[part_name]
            seg_data = nib.load(str(seg_file)).get_fdata()
            
            # Add to the combined image
            combined_seg[seg_data > 0] = label_num
        
        combined_nifti = nib.Nifti1Image(combined_seg, ref_img.affine)
        inv_label_mapping = {v: k for k, v in label_mapping.items()}
        
        # Save labels
        output_path = os.path.join(output_dir, f"labelsTr/totalsegmentator_{subject_id}.nii.gz")
        save_multilabel_nifti(combined_nifti, output_path, inv_label_mapping)

        # Save image
        output_path_image = os.path.join(output_dir, f"imagesTr/totalsegmentator_{subject_id}.nii.gz")
        nib.save(main_ct_img, output_path_image)

        print(f"Saved combined segmentation to {output_path}")
    
    return label_mapping

def main():
    base_dir = 'Totalsegmentator_dataset_v201'
    output_dir = 'totalsegmentator_combined'
    
    try:
        label_mapping = combine_segmentations(base_dir, output_dir)
        
       # Save label mapping
        mapping_file = os.path.join(output_dir, 'label_mapping.txt')
        with open(mapping_file, 'w') as f:
            for part, label in sorted(label_mapping.items(), key=lambda x: x[1]):
                f.write(f"{label}: {part}\n")
        
        # Create and save ITK-SnAP label description file
        itksnap_file = os.path.join(output_dir, 'itksnap_label_description.txt')
        create_itksnap_label_file(label_mapping, itksnap_file)
                
        print(f"\nLabel mapping saved to {mapping_file}")
        print(f"ITK-SnAP label description saved to {itksnap_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()