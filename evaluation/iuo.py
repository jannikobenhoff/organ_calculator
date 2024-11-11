
from helper import softmax_helper_dim1, AllGatherGrad, get_tp_fp_fn_tn
import torch
from torch import nn, distributed

from typing import Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
from typing import Callable

class IoULoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_iou: bool = False, 
                 do_bg: bool = True, smooth: float = 1.):
        """
        Implementation of IoU loss for semantic segmentation.
        
        Args:
            apply_nonlin (Callable): Activation function to be applied (e.g. softmax)
            batch_iou (bool): Whether to compute IoU over batch dimension
            do_bg (bool): Whether to include background class in loss computation
            smooth (float): Smoothing factor to avoid division by zero
        """
        super(IoULoss, self).__init__()

        self.do_bg = do_bg
        self.batch_iou = batch_iou
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Flatten the tensors
        batch_size = x.size(0)
        num_classes = x.size(1)
        
        if self.batch_iou:
            x = x.view(batch_size * num_classes, -1)
            y = y.view(batch_size * num_classes, -1)
        else:
            x = x.view(batch_size, num_classes, -1)
            y = y.view(batch_size, num_classes, -1)

        # Calculate intersection and union
        intersection = (x * y).sum(dim=-1)
        union = x.sum(dim=-1) + y.sum(dim=-1) - intersection

        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)

        # Handle background class
        if not self.do_bg:
            if self.batch_iou:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]

        # Average IoU across classes
        iou = iou.mean()

        return 1 - iou  # Return 1 - IoU for minimization

if __name__ == "__main__":
    # Initialize the loss function
    iou_loss = IoULoss(
        apply_nonlin=torch.nn.Softmax(dim=1),
        batch_iou=True,
        do_bg=True,
        smooth=1e-5
    )

    # Example tensors
    # pred shape: [batch_size, num_classes, height, width]
    # ref shape: [batch_size, num_classes, height, width]
    pred = torch.randn(4, 3, 256, 256)  # Example predictions
    ref = torch.randint(0, 2, (4, 3, 256, 256)).float()  # Example reference masks

    # Calculate loss
    loss = iou_loss(pred, ref)
    print(f"IoU Loss: {loss.item()}")