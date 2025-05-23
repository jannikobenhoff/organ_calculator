import numpy as np
from med_synth_gan.inference.utils import png_slices_to_nifti
import os, shutil, nibabel as nib, torch, torchvision.utils as vutils
from torch.utils.data import DataLoader
from med_synth_gan.dataset.single_2d_dataset import SingleVolume2DDataset
from med_synth_gan.dataset.utils import contrast_transform_ct_3d, load_and_resample

class VolumeInference:
    """
    * dim="2d": run slice-by-slice like before.
    * dim="3d": feed the whole volume through the generator once,
                save a NIfTI file and a middle slice PNG for quick check.
    """
    def __init__(self, test_volume_path, output_dir, device, dim="2d"):
        self.path   = test_volume_path
        self.outdir = output_dir
        self.device = device
        self.dim    = dim
        self.middle_slices = []
        os.makedirs(output_dir, exist_ok=True)

    def _save_png(self, tensor, fname):
        vutils.save_image(tensor.cpu(), fname, normalize=True)

    def run_inference(self, model, epoch):
        epoch_dir = os.path.join(self.outdir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        if self.dim == "2d":
            self._run_2d(model, epoch_dir, epoch)
        else:
            self._run_3d(model, epoch_dir, epoch)

    def _run_2d(self, model, epoch_dir, epoch):
        fake_dir = os.path.join(epoch_dir, "fake_mri_slices")
        ct_dir   = os.path.join(epoch_dir, "ct_slices")
        for d in (fake_dir, ct_dir):
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d)

        test_ds = SingleVolume2DDataset(self.path, slice_axis=2)
        loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

        model.eval()
        fake_stack, ct_stack = [], []
        with torch.no_grad():
            for i, (ct_slice,) in enumerate(loader):
                ct_slice = ct_slice.to(self.device)
                fake_mri = model.G_ct2mri(ct_slice)[0]

                self._save_png(fake_mri, os.path.join(fake_dir, f"fake_{i:04d}.png"))
                self._save_png(ct_slice,  os.path.join(ct_dir,   f"ct_{i:04d}.png"))

                fake_stack.append(fake_mri)
                ct_stack.append(ct_slice)

        # convert PNG stack → NIfTI
        png_slices_to_nifti(fake_dir, os.path.join(epoch_dir, f"fake_mri_{epoch}.nii.gz"))
        png_slices_to_nifti(ct_dir,   os.path.join(epoch_dir, f"ct_{epoch}.nii.gz"))

        # grid update (middle third slice of stack)
        if fake_stack:
            mid = len(fake_stack) // 3
            # self.middle_slices.append(ct_stack[mid][0].cpu())   # GT CT
            self.middle_slices.append(fake_stack[mid][0].cpu()) # Fake MRI
        model.train()

    def _run_3d(self, model, epoch_dir, epoch):
        vol = load_and_resample(nib.load(self.path), size=(256, 256, 96))
        vol = contrast_transform_ct_3d(vol).to(self.device)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            fake_vol = model.G_ct2mri(vol.unsqueeze(0))[0]  # 1×1×D×H×W
        if was_training:
            model.train()

        fake_vol = fake_vol.squeeze(0).squeeze(0)
        fake_arr = fake_vol.cpu().numpy().astype("float32")

        nib.save(nib.Nifti1Image(fake_arr, np.eye(4, dtype="float32")),
                 os.path.join(epoch_dir, f"fake_mri_{epoch}.nii.gz"))

        # ---------- save middle axial slice ----------
        mid = fake_vol.shape[-1] // 2  # middle of W axis
        ct_mid = vol[0, :, :, mid].unsqueeze(0)  # 1×H×D
        mri_mid = fake_vol[:, :, mid].unsqueeze(0)  # 1×H×D
        # mid = fake_vol.shape[0] // 2  # middle slice along D
        # ct_mid = vol[0, mid, :, :].unsqueeze(0)  # (1, H, W)
        # mri_mid = fake_vol[mid, :, :].unsqueeze(0)  # (1, H, W)

        ct_mid = ct_mid.permute(0, 2, 1)  # 1×H×W
        mri_mid = mri_mid.permute(0, 2, 1)

        self._save_png(mri_mid.cpu(), os.path.join(epoch_dir, "fake_mid.png"))
        self._save_png(ct_mid.cpu(), os.path.join(epoch_dir, "ct_mid.png"))

        self.middle_slices.append(mri_mid.cpu())

    def save_final_grid(self):
        if self.middle_slices:
            grid = vutils.make_grid(self.middle_slices, nrow=4, normalize=True)
            vutils.save_image(grid, os.path.join(self.outdir, "middle_slices_grid.png"))

