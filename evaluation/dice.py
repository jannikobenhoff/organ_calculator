from models.nnUNet.nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss
from models.nnUNet.nnunetv2.utilities.helpers import softmax_helper_dim1
import torch

if __name__ == '__main__':
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)