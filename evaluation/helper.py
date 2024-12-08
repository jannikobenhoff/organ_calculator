import torch
from torch import nn, distributed

from typing import Any, Optional, Tuple, Callable

import nibabel as nib
import torch
import numpy as np

def load_and_process_nifti(nifti_path, selected_organs=[1, 2, 3, 4]):
    # Load nifti file
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    
    # Create binary masks for each selected organ
    processed = torch.zeros((len(selected_organs),) + data.shape)
    for idx, organ_id in enumerate(selected_organs):
        print(f"Processing organ {organ_id}")
        processed[idx] = torch.from_numpy((data == organ_id).astype(np.float32))
        processed[idx][processed[idx] == 1] = organ_id 
        print(processed[idx].unique())
    return processed

def load_nfiti(nifti_path, selected_organs):
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    data[data == 1] = 42 # kidney left
    data[data == 2] = 43 # kidney right
    data[data == 3] = 44 # liver
    data[data == 4] = 84 # spline
    # Create binary masks for each selected organ
    processed = torch.zeros((len(selected_organs),) + data.shape)
    for idx, organ_id in enumerate(selected_organs):
        print(f"Processing organ {organ_id}")
        processed[idx] = torch.from_numpy((data == organ_id).astype(np.float32))
        processed[idx][processed[idx] == 1] = organ_id 
        print(processed[idx].unique())
    return processed


def prepare_evaluation_tensors(pred_path, ref_path, selected_organs=[1, 2, 3, 4]):
    # Load predictions and reference
    pred_data = load_and_process_nifti(pred_path, selected_organs)
    ref_data = load_nfiti(ref_path, selected_organs)
    print(pred_data.unique())
    print(ref_data.unique())

    # Add batch dimension if not present
    if pred_data.dim() == 4:
        pred_data = pred_data.unsqueeze(0)
    if ref_data.dim() == 4:
        ref_data = ref_data.unsqueeze(0)
    
    # Convert reference to expected format (B, H, W, D)
    ref_data = torch.argmax(ref_data, dim=1)
    
    return pred_data, ref_data

def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)



def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


class AllGatherGrad(torch.autograd.Function):
    # stolen from pytorch lightning
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> torch.Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None


if __name__ == "__main__":
    pred_path = "../data/inference_output/Abdomen_000.nii.gz"
    ref_path = "../data/inference_ref/000_segmented.nii.gz"

    selected_organs = [42, 43, 44, 84]  # organ IDs you want to evaluate

    # Prepare tensors for evaluation
    pred_tensor, ref_tensor = prepare_evaluation_tensors(pred_path, ref_path, selected_organs)
    print(pred_tensor.shape, ref_tensor.shape)
    print(pred_tensor.count_nonzero())
    print(ref_tensor.count_nonzero())
