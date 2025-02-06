
from helper import prepare_evaluation_tensors, softmax_helper_dim1, AllGatherGrad, get_tp_fp_fn_tn
import torch
from torch import nn, distributed
import SimpleITK as sitk

from typing import Any, List, Optional, Tuple, Callable, Union
import torch
import torch.nn as nn
from typing import Callable
import numpy as np

import nibabel as nib
import matplotlib.pyplot as plt

def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_metrics(reference_file: str, prediction_file: str,  
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    # Load images
    nifti_img = nib.load(prediction_file)
    seg_pred = nifti_img.get_fdata()
    pred_spacing = nifti_img.header.get_zooms()[:3]  # Get voxel spacing (x, y, z)

    nifti_img = nib.load(reference_file)
    seg_ref = nifti_img.get_fdata()
    ref_spacing = nifti_img.header.get_zooms()[:3]  # Get voxel spacing (x, y, z)
    print("Voxel Spacing:", ref_spacing)
    # Ensure the spacing matches between reference and prediction
    if ref_spacing != pred_spacing:
        raise ValueError("Voxel spacing mismatch between reference and prediction images.")

    print(np.unique(seg_ref))
    seg_ref[seg_ref == 2] = 42  # kidney left
    seg_ref[seg_ref == 1] = 43  # kidney right
    seg_ref[seg_ref == 3] = 44  # liver
    seg_ref[seg_ref == 4] = 84  # spleen

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp

        mask_ref_itk = sitk.GetImageFromArray(mask_ref.astype(np.uint8))
        mask_pred_itk = sitk.GetImageFromArray(mask_pred.astype(np.uint8))
        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(mask_ref_itk, mask_pred_itk)
    
        # Get the computed Hausdorff distance in voxels
        hausdorff_distance_voxels = hausdorff_filter.GetHausdorffDistance()

        # Convert Hausdorff distance to millimeters
        voxel_spacing = np.array(ref_spacing)  # Ensure spacing is in (x, y, z)
        hausdorff_distance_mm = hausdorff_distance_voxels * np.linalg.norm(voxel_spacing)

        results['metrics'][r]['HausdorffDistanceVoxels'] = hausdorff_distance_voxels
        results['metrics'][r]['HausdorffDistanceMM'] = hausdorff_distance_mm

    return results

if __name__ == "__main__":
    selection = ["000", "001", "002", "003"]
    organs = {
        42: "Kidney Left",
        43: "Kidney Right",
        44: "Liver",
        84: "Spline"
    }

    dice_dict = {}
    iou_dict = {}
    hd_dict = {}
    for s in selection:
        pred_path = f"../data/inference_output/Abdomen_{s}.nii.gz"
        ref_path = f"../data/inference_ref/{s}_segmented.nii.gz"

        selected_organs = [42, 43, 44, 84]  # organ IDs you want to evaluate

        # Prepare tensors for evaluation
        result = compute_metrics(ref_path, pred_path, selected_organs)

        for key, value in result.items():
            if key == 'metrics':
                for k, v in value.items():
                    print(f"Organ: {organs[k]}")
                    print("Dice:", v["Dice"])
                    print("IuO:", v["IoU"])
                    print("HausdorffDistanceMM:", v["HausdorffDistanceMM"])

                    if k not in dice_dict:
                        dice_dict[k] = []
                        iou_dict[k] = []
                        hd_dict[k] = [] 
                    dice_dict[k].append(v["Dice"])
                    iou_dict[k].append(v["IoU"])
                    hd_dict[k].append(v["HausdorffDistanceMM"])
            else:
                print(key, value)

    
    print("Mean Dice:", {organs[k]: np.mean(v) for k, v in dice_dict.items()})
    print("Mean IoU:", {organs[k]: np.mean(v) for k, v in iou_dict.items()})
    print("Mean HausdorffDistanceMM:", {organs[k]: np.mean(v) for k, v in hd_dict.items()})

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for k, v in dice_dict.items():
        ax[0].plot(selection, v, label=organs[k])
    ax[0].set_title("Dice")
    ax[0].legend()

    for k, v in iou_dict.items():
        ax[1].plot(selection, v, label=organs[k])
    ax[1].set_title("IoU")
    ax[1].legend()

    for k, v in hd_dict.items():
        ax[2].plot(selection, v, label=organs[k])
    ax[2].set_title("HausdorffDistanceMM")
    ax[2].legend()

    plt.show()
