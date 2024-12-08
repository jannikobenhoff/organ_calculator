import nibabel as nib
import torch
import numpy as np

def calculate_ratio(path, target):
    nifti_img = nib.load(path)
    data = nifti_img.get_fdata()
    
    organ_left = data == list(target.items())[0][1]
    organ_right = data == list(target.items())[1][1]

    ratio = np.sum(organ_left) / np.sum(organ_right)
    return ratio


if __name__ =="__main__":
    targets = {
        "autochthon_left": 5,
        "autochthon_right": 6
    }

    selection = ["000", "001", "002", "003"]
    
    ratios = []
    for s in selection:
        pred_path = f"../data/inference_output/Abdomen_{s}.nii.gz"
        ratio = calculate_ratio(pred_path, targets)
        print(f"Ratio for {s}: {ratio}")

        ratios.append(ratio)
