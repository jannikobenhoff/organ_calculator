import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff
from typing import Tuple, List, Optional

class ContourDistanceMetric(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 compute_hausdorff: bool = True,
                 ignore_index: Optional[int] = None,
                 spacing: Tuple[float, float] = (1.0, 1.0)):
        """
        Compute mean contour distance and optionally Hausdorff distance between prediction and ground truth.
        
        Args:
            num_classes (int): Number of classes in segmentation
            compute_hausdorff (bool): Whether to compute Hausdorff distance
            ignore_index (int, optional): Class index to ignore
            spacing (tuple): Pixel spacing in (x, y) directions
        """
        super().__init__()
        self.num_classes = num_classes
        self.compute_hausdorff = compute_hausdorff
        self.ignore_index = ignore_index
        self.spacing = spacing

    def extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract contours from binary mask using OpenCV."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def compute_contour_distances(self, 
                                pred_contour: np.ndarray, 
                                target_contour: np.ndarray) -> Tuple[float, float]:
        """
        Compute average and Hausdorff distances between two contours.
        """
        # Convert contours to point sets
        pred_points = pred_contour.squeeze().reshape(-1, 2)
        target_points = target_contour.squeeze().reshape(-1, 2)

        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf'), float('inf')

        # Apply spacing
        pred_points = pred_points * np.array(self.spacing)
        target_points = target_points * np.array(self.spacing)

        # Compute average surface distance
        distances_pred_to_target = np.min([np.linalg.norm(p - target_points, axis=1) 
                                         for p in pred_points], axis=1)
        distances_target_to_pred = np.min([np.linalg.norm(p - pred_points, axis=1) 
                                         for p in target_points], axis=1)
        
        avg_distance = (np.mean(distances_pred_to_target) + 
                       np.mean(distances_target_to_pred)) / 2

        # Compute Hausdorff distance if requested
        if self.compute_hausdorff:
            hausdorff_distance = max(
                directed_hausdorff(pred_points, target_points)[0],
                directed_hausdorff(target_points, pred_points)[0]
            )
        else:
            hausdorff_distance = None

        return avg_distance, hausdorff_distance

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Calculate mean contour distance and Hausdorff distance.
        
        Args:
            pred (torch.Tensor): Predictions (B, C, H, W) or (B, H, W)
            target (torch.Tensor): Ground truth (B, H, W)
            
        Returns:
            dict: Dictionary containing average and Hausdorff distances for each class
        """
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)

        # Move tensors to CPU and convert to numpy
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        batch_size = pred.shape[0]
        results = {
            'average_distance': np.zeros((batch_size, self.num_classes)),
            'hausdorff_distance': np.zeros((batch_size, self.num_classes)) 
            if self.compute_hausdorff else None
        }

        # Compute distances for each batch and class
        for batch_idx in range(batch_size):
            for class_idx in range(self.num_classes):
                if class_idx == self.ignore_index:
                    continue

                # Create binary masks for current class
                pred_mask = (pred[batch_idx] == class_idx)
                target_mask = (target[batch_idx] == class_idx)

                # Skip if class is not present
                if not np.any(pred_mask) and not np.any(target_mask):
                    continue

                # Extract contours
                pred_contours = self.extract_contours(pred_mask)
                target_contours = self.extract_contours(target_mask)

                # Skip if no contours found
                if not pred_contours or not target_contours:
                    continue

                # Compute distances for largest contours
                pred_contour = max(pred_contours, key=cv2.contourArea)
                target_contour = max(target_contours, key=cv2.contourArea)

                avg_dist, hausdorff_dist = self.compute_contour_distances(
                    pred_contour, target_contour)

                results['average_distance'][batch_idx, class_idx] = avg_dist
                if self.compute_hausdorff:
                    results['hausdorff_distance'][batch_idx, class_idx] = hausdorff_dist

        # Compute mean across batch
        results['average_distance_mean'] = np.mean(results['average_distance'])
        if self.compute_hausdorff:
            results['hausdorff_distance_mean'] = np.mean(results['hausdorff_distance'])

        return results


if __name__ == "__main__":
    # Initialize the metric
    contour_metric = ContourDistanceMetric(
        num_classes=3,
        compute_hausdorff=True,
        spacing=(1.0, 1.0)  # pixel spacing in mm
    )

    # Example data
    pred = torch.randn(4, 3, 256, 256)  # (batch_size, num_classes, height, width)
    target = torch.randint(0, 3, (4, 256, 256))  # (batch_size, height, width)

    # Calculate distances
    distances = contour_metric(pred, target)

    print(f"Mean Average Distance: {distances['average_distance_mean']:.2f}")
    print(f"Mean Hausdorff Distance: {distances['hausdorff_distance_mean']:.2f}")

    # Per-class distances
    print("\nPer-class Average Distances:")
    for class_idx in range(contour_metric.num_classes):
        print(f"Class {class_idx}: {np.mean(distances['average_distance'][:, class_idx]):.2f}")