import argparse
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from med_synth_gan.dataset.ct_mri_2d_dataset import CtMri2DDataset
from med_synth_gan.models.models import UNet
from med_synth_gan.models.cycle_gan import Discriminator
from med_synth_gan.models.losses import Grad
from med_synth_gan.inference.inference import VolumeInferenceCallback
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torchvision.utils as vutils
import os
import numpy as np
import torch.nn.functional as F


class MedSynthGANModule(pl.LightningModule):
    def __init__(self, lr, lr_d, lambda_grad):
        super().__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.lr_d = lr_d
        self.lambda_grad = lambda_grad

        self.save_hyperparameters()

        # Models
        self.G_ct2mri = UNet()
        self.G_mri2ct = UNet()
        self.D_mri = Discriminator()
        self.D_ct = Discriminator()

        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_grad = Grad(penalty='l1')

    def forward(self, ct_image):
        return self.G_ct2mri(ct_image)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        # Get optimizers
        opt_g, opt_d = self.optimizers()
        real_ct, real_mri = batch

        # Train Generator
        opt_g.zero_grad()
        fake_mri, scale_field_ct2mri = self.G_ct2mri(real_ct)

        # loss_histogram = self.histogram_loss(fake_mri, real_mri) * 0.1

        pred_fake_mri = self.D_mri(fake_mri)
        loss_GAN_ct2mri = self.criterion_GAN(pred_fake_mri, torch.ones_like(pred_fake_mri))
        loss_grad_ct2mri = self.criterion_grad.loss(None, scale_field_ct2mri) * self.lambda_grad
        loss_G = loss_GAN_ct2mri + loss_grad_ct2mri
        self.manual_backward(loss_G)
        #self.clip_gradients(opt_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt_g.step()

        # Train Discriminator
        opt_d.zero_grad()
        fake_mri, _ = self.G_ct2mri(real_ct)
        pred_real_mri = self.D_mri(real_mri)
        loss_D_real = self.criterion_GAN(pred_real_mri, torch.ones_like(pred_real_mri))
        pred_fake_mri = self.D_mri(fake_mri.detach())
        loss_D_fake = self.criterion_GAN(pred_fake_mri, torch.zeros_like(pred_fake_mri))
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.manual_backward(loss_D)
        #self.clip_gradients(opt_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt_d.step()

        if batch_idx % 100 == 0:
            self.log('loss_G', loss_G, prog_bar=True)
            self.log('loss_D', loss_D, prog_bar=True)
            self.log('scalar_field_mean', scale_field_ct2mri.mean(), prog_bar=True)
            self.log('scalar_field_min', scale_field_ct2mri.min(), prog_bar=True)
            self.log('scalar_field_max', scale_field_ct2mri.max(), prog_bar=True)
            self.log('tv_loss', loss_grad_ct2mri, prog_bar=True)
            # self.log('hist_loss', loss_histogram, prog_bar=True)

            vutils.save_image(
                real_mri,
                f"mri_train_slice{batch_idx}.png",
                normalize=True
            )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.G_ct2mri.parameters()) + list(self.G_mri2ct.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        opt_d = torch.optim.Adam(self.D_mri.parameters(), lr=self.lr_d, betas=(0.5, 0.999))

        return [opt_g, opt_d], []

    def histogram_loss(self, generated_image, real_image):
        """Computes KL divergence between generated and expected histogram."""
        gen_hist, _ = self.compute_normalized_histogram(generated_image)
        expected_hist, _ = self.compute_normalized_histogram(real_image)

        # Re-normalize histograms
        gen_hist = gen_hist / gen_hist.sum()
        expected_hist = expected_hist / expected_hist.sum()

        # Avoid log(0) issues
        gen_hist = torch.clamp(gen_hist, min=1e-8)
        expected_hist = torch.clamp(expected_hist, min=1e-8)

        # Compute KL divergence
        loss = F.kl_div(gen_hist.log(), expected_hist, reduction="batchmean")  # Expect p log(p/q)

        return loss

    def compute_normalized_histogram(self, tensor, bins=11, range_min=0, range_max=1):
        """
        Compute a normalized histogram from a tensor.
        - Bins: Number of bins to use (default 11)
        - range_min, range_max: Value range (default [0,1] for normalized MRI)
        """
        device = tensor.device  # Get the device before converting to NumPy
        tensor = tensor.detach().cpu().numpy().flatten()  # Convert to NumPy

        # Compute histogram
        hist, bin_edges = np.histogram(tensor, bins=bins, range=(range_min, range_max))

        # Normalize to sum to 1 (avoid division by zero)
        hist = hist.astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum  # Normalize

        return torch.tensor(hist, dtype=torch.float32, device=device), bin_edges  # Assign back to the correct device


class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__(refresh_rate=100, leave=True, process_position=0)  # Update every 10 batches

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description("Training")
        return bar

    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)
        return {
            "epoch": items.get("epoch", ""),
            "batch": items.get("step", ""),
            "loss_G": items.get("loss_G", ""),
            "loss_D": items.get("loss_D", ""),
            "sf_mean": items.get("scalar_field_mean", ""),
            "sf_min": items.get("scalar_field_min", ""),
            "sf_max": items.get("scalar_field_max", ""),
            "tv_loss": items.get("tv_loss", ""),
            #"hist_loss": items.get("hist_loss", ""),
        }

def collate_fn(batch):
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="MedSynthGAN training script.")

    parser.add_argument(
        "-b",
        "--batch-size",
        default=24,
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lambda_grad",
        "--lambda-grad",
        default=1e-6,  #1e-6
        type=float,
        help="Weight for total-variation (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=5e-6,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-lr_d",
        "--learning-rate-discriminator",
        default=1e-6,
        type=float,
        help="Learning rate (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    print("Starting MedSynthGAN training with args: {}".format(args), flush=True)

    # Dataset and DataLoader
    train_dataset = CtMri2DDataset(
        ct_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/",
        mri_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/imagesTr/",
        slice_axis=2
    )

    print("Finished loading {} training samples".format(len(train_dataset)), flush=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize model and trainer
    model = MedSynthGANModule(lr=args.learning_rate, lr_d=args.learning_rate_discriminator, lambda_grad=args.lambda_grad)

    # Inference
    inference_callback = VolumeInferenceCallback(
        test_volume_path="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTs/AMOS_CT_2022_000009_0000.nii.gz",
        output_dir="inference_results"
    )

    trainer = pl.Trainer(
        default_root_dir='checkpoints',
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[
            CustomProgressBar(),
            inference_callback
        ],
    )

    # trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    # trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Train the model
    trainer.fit(model, train_dataloader)

    return model


if __name__ == "__main__":
    main(sys.argv[1:])
