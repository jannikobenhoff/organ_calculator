import torch
import torch.nn as nn
from typing import Optional

class JaccardIndex(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 ignore_index: Optional[int] = None,
                 absent_score: float = 1.0,
                 threshold: float = 0.5,
                 reduction: str = 'mean'):
        """
        Compute Jaccard Index (IoU) for segmentation evaluation.
        
        Args:
            num_classes (int): Number of classes
            ignore_index (int, optional): Index to ignore in calculation
            absent_score (float): Score to use for classes absent in both prediction and target
            threshold (float): Threshold for converting soft predictions to binary
            reduction (str): 'mean', 'sum', or 'none' for per-class scores
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.threshold = threshold
        self.reduction = reduction
        
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Jaccard Index.
        
        Args:
            pred (torch.Tensor): Predictions (B, C, H, W) or (B, H, W)
            target (torch.Tensor): Ground truth (B, H, W)
            
        Returns:
            torch.Tensor: Jaccard Index score(s)
        """
        if pred.dim() == 4:  # (B, C, H, W)
            if pred.size(1) != self.num_classes:
                raise ValueError(f"Expected {self.num_classes} channels in pred, got {pred.size(1)}")
            # Convert probabilities to class predictions
            pred = torch.argmax(pred, dim=1)
        
        if pred.dim() != 3 or target.dim() != 3:
            raise ValueError("Both pred and target should be 3D tensors after processing")

        # Initialize scores tensor
        scores = torch.zeros(self.num_classes, device=pred.device)
        
        # Calculate IoU for each class
        for class_idx in range(self.num_classes):
            if class_idx == self.ignore_index:
                continue
                
            # Create binary masks for current class
            pred_mask = (pred == class_idx)
            target_mask = (target == class_idx)
            
            # Skip if ignore_index is present
            if self.ignore_index is not None:
                valid_mask = target != self.ignore_index
                pred_mask = pred_mask & valid_mask
                target_mask = target_mask & valid_mask
            
            # Calculate intersection and union
            intersection = torch.logical_and(pred_mask, target_mask).sum()
            union = torch.logical_or(pred_mask, target_mask).sum()
            
            # Handle cases where class is absent in both pred and target
            if union == 0:
                scores[class_idx] = self.absent_score
            else:
                scores[class_idx] = intersection.float() / union.float()
        
        # Apply reduction
        if self.reduction == 'mean':
            return scores.mean()
        elif self.reduction == 'sum':
            return scores.sum()
        else:  # 'none'
            return scores

    def __str__(self):
        return (f"JaccardIndex(num_classes={self.num_classes}, "
                f"ignore_index={self.ignore_index}, "
                f"absent_score={self.absent_score}, "
                f"threshold={self.threshold}, "
                f"reduction={self.reduction})")


if __name__ == "__main__":
    # Initialize the metric
    jaccard = JaccardIndex(
        num_classes=3,
        ignore_index=None,
        absent_score=1.0,
        reduction='mean'
    )

    # Example data
    pred = torch.randn(4, 3, 256, 256)  # (batch_size, num_classes, height, width)
    target = torch.randint(0, 3, (4, 256, 256))  # (batch_size, height, width)

    # Calculate Jaccard Index
    score = jaccard(pred, target)
    print(f"Jaccard Index: {score.item()}")

    # For per-class scores
    jaccard_per_class = JaccardIndex(
        num_classes=3,
        reduction='none'
    )
    class_scores = jaccard_per_class(pred, target)
    print("Per-class Jaccard Index:", class_scores)