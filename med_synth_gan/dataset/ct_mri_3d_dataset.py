import os, glob, random
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from med_synth_gan.dataset.utils import contrast_transform_ct_3d, load_and_resample, reorient_to_ras

class CtMri3DDataset(Dataset):
    """
    Returns *aligned* CT / MRI volumes (shape: C×D×H×W, C=1).
    The two volumes are *independent* draws (same logic you used in 2-D).
    """

    def __init__(
        self,
        ct_dir:  str,
        mri_dir: str,
        out_size: tuple[int, int, int] = (256,256,96),
        ct_limit: int = 100,          # keep your “balance” heuristic
    ):
        super().__init__()
        self.ct_paths  = sorted(glob.glob(os.path.join(ct_dir,  '*.nii*')))
        self.mri_paths = sorted(glob.glob(os.path.join(mri_dir, '*.nii*')))

        self.ct_paths = self.ct_paths[:ct_limit]

        # lazy-load – keep headers only, load ndarray when actually needed
        # self.ct_imgs  = [nib.load(p) for p in self.ct_paths]
        # self.mri_imgs = [nib.load(p) for p in self.mri_paths]
        self.ct_imgs = [reorient_to_ras(nib.load(p)) for p in self.ct_paths]
        self.mri_imgs = [reorient_to_ras(nib.load(p)) for p in self.mri_paths]

        self.out_size = out_size  # final (D,H,W); keep power-of-2 for UNet

        # pre-compute statistics for MRI normalisation
        self.mri_stats = [
            dict(
                p1  = np.percentile(img.get_fdata(dtype=np.float32), 1 ),
                p99 = np.percentile(img.get_fdata(dtype=np.float32), 99),
            ) for img in self.mri_imgs
        ]

    def __len__(self):
        return max(len(self.ct_paths), len(self.mri_paths))

    def __getitem__(self, idx):
        # ----------- CT ----------
        ct_idx = idx % len(self.ct_paths)
        ct_vol = load_and_resample(self.ct_imgs[ct_idx], self.out_size)
        ct_vol = self._ct_contrast(ct_vol)

        # ----------- MRI (random) ----------
        mri_idx = random.randint(0, len(self.mri_paths) - 1)
        mri_vol = load_and_resample(self.mri_imgs[mri_idx], self.out_size)
        mri_vol = self._mri_contrast(mri_vol, self.mri_stats[mri_idx])

        return ct_vol, mri_vol

    @staticmethod
    def _ct_contrast(x: torch.Tensor) -> torch.Tensor:
        x = contrast_transform_ct_3d(x)
        return x

    @staticmethod
    def _mri_contrast(x: torch.Tensor, stats) -> torch.Tensor:
        p1, p99 = stats["p1"], stats["p99"]
        x = torch.clamp(x, 0, p99)
        x = (x - 0) / (p99 - 0 + 1e-8)
        return x
