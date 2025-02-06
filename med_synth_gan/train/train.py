import argparse
import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset.ct_mri_2d_dataset import CtMri2DDataset
from models.models import Discriminator, UNet
from models.losses import Grad
from torch.cuda.amp import GradScaler, autocast
from inference.inference import VolumeInferenceCallback


class MedSynthGANModule(pl.LightningModule):
    def __init__(self, lr=1e-4, lr_d=3e-4, lambda_grad=0):
        super().__init__()
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

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_ct, real_mri = batch

        # Train Generator
        if optimizer_idx == 0:
            fake_mri, scale_field_ct2mri = self.G_ct2mri(real_ct)

            pred_fake_mri = self.D_mri(fake_mri)
            loss_GAN_ct2mri = self.criterion_GAN(pred_fake_mri, torch.ones_like(pred_fake_mri))

            loss_grad_ct2mri = self.criterion_grad.loss(None, scale_field_ct2mri) * self.hparams.lambda_grad

            loss_G = loss_GAN_ct2mri + loss_grad_ct2mri

            self.log('loss_G', loss_G)
            return loss_G

        # Train Discriminator
        if optimizer_idx == 1:
            fake_mri, _ = self.G_ct2mri(real_ct)

            pred_real_mri = self.D_mri(real_mri)
            loss_D_real = self.criterion_GAN(pred_real_mri, torch.ones_like(pred_real_mri))

            pred_fake_mri = self.D_mri(fake_mri.detach())
            loss_D_fake = self.criterion_GAN(pred_fake_mri, torch.zeros_like(pred_fake_mri))

            loss_D = (loss_D_real + loss_D_fake) * 0.5

            self.log('loss_D', loss_D)
            return loss_D

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.G_ct2mri.parameters()) + list(self.G_mri2ct.parameters()),
            lr=self.hparams.lr
        )
        opt_d = torch.optim.Adam(self.D_mri.parameters(), lr=self.hparams.lr_d)

        return [opt_g, opt_d], []


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
        "-lambda",
        "--lambda-grad",
        default=1e-6,
        type=float,
        help="Weight for total-variation (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-lr_d",
        "--learning-rate-discriminator",
        default=3e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    # Dataset and DataLoader
    train_dataset = CtMri2DDataset(
        ct_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/",
        mri_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/imagesTr/",
        slice_axis=2
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True
    )

    # Initialize model and trainer
    model = MedSynthGANModule(lr=args.learning_rate)

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
            ModelCheckpoint(save_top_k=2, monitor="loss_G"),
            LearningRateMonitor("epoch"),
            inference_callback
        ],
    )

    # Train the model
    trainer.fit(model, train_dataloader)

    return model


if __name__ == "__main__":
    main(sys.argv[1:])
