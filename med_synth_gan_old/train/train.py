import argparse
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sympy import false
from torch.utils.data import DataLoader
from med_synth_gan.dataset.ct_mri_2d_dataset import CtMri2DDataset
from med_synth_gan.models.models import UNet
from med_synth_gan.models.cycle_gan import Discriminator
from med_synth_gan.models.losses import Grad
from med_synth_gan.inference.inference import VolumeInferenceCallback
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import math
import numpy as np
import cv2
import random


class MedSynthGANModule(pl.LightningModule):
    def __init__(self, lr, lr_d, lambda_grad, loss_type="mse"):
        super().__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.lr_d = lr_d
        self.lambda_grad = lambda_grad
        self.loss_type = loss_type
        #self.step = 0
        self.save_hyperparameters()

        # Models
        self.G_ct2mri = UNet()
        # self.G_mri2ct = UNet()
        self.D_mri = Discriminator()
        # self.D_ct = Discriminator()

        # Loss functions
        if loss_type == "bce":
            self.criterion_GAN = nn.BCEWithLogitsLoss()
        elif loss_type == "mse":
            self.criterion_GAN = nn.MSELoss()
        elif loss_type == "hinge":
            self.criterion_GAN = nn.HingeEmbeddingLoss(margin=1)
        self.criterion_grad = Grad(penalty='l1')


    def forward(self, ct_image):
        return self.G_ct2mri(ct_image)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        # Get optimizers
        opt_g, opt_d = self.optimizers()
        real_ct, real_mri = batch
        real_mri = real_mri[:4]

        opt_g.zero_grad()
        fake_mri, scale_field_ct2mri = self.G_ct2mri(real_ct)

        pred_fake_mri = self.D_mri(fake_mri)
        if self.loss_type == "hinge":
            loss_GAN_ct2mri = self.criterion_GAN(pred_fake_mri, torch.full_like(pred_fake_mri, 1))  # Ensure +1 target
        else:
            loss_GAN_ct2mri = self.criterion_GAN(pred_fake_mri, torch.ones_like(pred_fake_mri))

        loss_grad_ct2mri = self.criterion_grad.loss(None, scale_field_ct2mri) * self.lambda_grad
        loss_G = loss_GAN_ct2mri + loss_grad_ct2mri
        self.manual_backward(loss_G)

        # torch.nn.utils.clip_grad_norm_(self.G_ct2mri.parameters(), 0.1)
        opt_g.step()

        #if self.step % 2 == 0:

        # Train Discriminator
        opt_d.zero_grad()

        fake_mri, _ = self.G_ct2mri(real_ct)

        real_mri_aug = self.augment_for_discriminator(real_mri)
        fake_mri_aug = self.augment_for_discriminator(fake_mri.detach())

        pred_real_mri = self.D_mri(real_mri_aug)
        pred_fake_mri = self.D_mri(fake_mri_aug)

        if self.loss_type == "hinge":
            real_labels = torch.full_like(pred_real_mri, 1)
            fake_labels = torch.full_like(pred_fake_mri, -1)
            loss_D_real = self.criterion_GAN(pred_real_mri, real_labels)
            loss_D_fake = self.criterion_GAN(pred_fake_mri, fake_labels)
        elif self.loss_type == "bce":
            real_labels = torch.full_like(pred_real_mri, 0.9)  # label smoothing
            fake_labels = torch.zeros_like(pred_fake_mri)
            loss_D_real = self.criterion_GAN(pred_real_mri, real_labels)
            loss_D_fake = self.criterion_GAN(pred_fake_mri, fake_labels)
        else:  # MSE
            loss_D_real = self.criterion_GAN(pred_real_mri, torch.ones_like(pred_real_mri))
            loss_D_fake = self.criterion_GAN(pred_fake_mri, torch.zeros_like(pred_fake_mri))

        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        self.manual_backward(loss_D)
        opt_d.step()

        if batch_idx % 10 == 0 : #and self.step % 2 == 0:
            self.log('loss_G', loss_G, prog_bar=True)
            self.log('loss_D', loss_D, prog_bar=True)
            self.log('scalar_field_mean', scale_field_ct2mri.mean(), prog_bar=True)
            self.log('scalar_field_min', scale_field_ct2mri.min(), prog_bar=True)
            self.log('scalar_field_max', scale_field_ct2mri.max(), prog_bar=True)
            #self.log('tv_loss', loss_grad_ct2mri, prog_bar=True)
            self.log('lr_d', self.lr_d, prog_bar=True)
            self.log('lr_g', self.lr, prog_bar=True)

        # if batch_idx % 100 == 0:
        #     vutils.save_image(
        #         real_mri,
        #         f"mri_train_slice{batch_idx}.png",
        #         normalize=True
        #     )
        #     vutils.save_image(
        #         real_ct,
        #         f"ct_train_slice{batch_idx}.png",
        #         normalize=True
        #     )
        #self.step += 1

    # def elastic_deformation(self, img, alpha=40, sigma=6):
    #     """Apply elastic deformation (2D) similar to B-spline warping."""
    #     device = img.device
    #     img_np = img.squeeze().cpu().numpy()
    #     shape = img_np.shape
    #
    #     dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    #     dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    #
    #     x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    #     map_x = (x + dx).astype(np.float32)
    #     map_y = (y + dy).astype(np.float32)
    #
    #     distorted = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    #     return torch.tensor(distorted).unsqueeze(0).to(device)

    def augment_for_discriminator(self, image, crop_size=224):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)

        augmented = []
        for img in image:
            # Random rotation (-30° to +30°)
            angle = random.uniform(-30, 30)
            img = TF.rotate(img, angle)

            # Random flip
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.vflip(img)

            # Random brightness and contrast
            # brightness_factor = random.uniform(0.8, 1.2)
            # contrast_factor = random.uniform(0.8, 1.2)
            # img = TF.adjust_brightness(img, brightness_factor)
            # img = TF.adjust_contrast(img, contrast_factor)

            # Gaussian noise
            # if random.random() > 0.5:
            #     noise = torch.randn_like(img) * 0.1
            #     img = img + noise
            #     img = torch.clamp(img, 0, 1)

            # Random crop to crop_size
            # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(crop_size, crop_size))
            # img = TF.crop(img, i, j, h, w)

            augmented.append(img)

        return torch.stack(augmented)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.G_ct2mri.parameters(),
            lr=self.lr, betas=(0.9, 0.95), weight_decay=0.001  # 0.01
        )

        opt_d = torch.optim.AdamW(self.D_mri.parameters(), lr=self.lr_d, betas=(0.9, 0.95), weight_decay=0.001)  # 0.01

        # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.98)
        # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.98)

        return (
            [opt_g, opt_d]
            # ,
            # [
            #     {"scheduler": scheduler_g, "interval": "epoch", "frequency": 1, "name": "lr_g"},
            #     {"scheduler": scheduler_d, "interval": "epoch", "frequency": 1, "name": "lr_d"},
            # ]
        )

class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__(refresh_rate=50, leave=True, process_position=0)  # Update every 10 batches

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description("Training")
        return bar

    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)
        return {
            "loss_G": items.get("loss_G", ""),
            "loss_D": items.get("loss_D", ""),
            "sf_mean": items.get("scalar_field_mean", ""),
            "sf_min": items.get("scalar_field_min", ""),
            "sf_max": items.get("scalar_field_max", ""),
            #"lr_g": items.get("lr_g", ""),
            #"lr_d": items.get("lr_d", ""),
            # "tv_loss": items.get("tv_loss", ""),
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
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lambda_grad",
        "--lambda-grad",
        default=0,
        type=float,
        help="Weight for total-variation (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=5e-5, #5e-5
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-lr_d",
        "--learning-rate-discriminator",
        default=4e-5, # should be larger than Generator for MSE 1e-4
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-loss_type",
        "--loss_type",
        default="mse",  # bce, mse, hinge
        type=str,
        help="Loss (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    print("Starting MedSynthGAN training with args: {}".format(args), flush=True)

    train_dataset = CtMri2DDataset(
        ct_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/",
        mri_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/t2Axial/",
        slice_axis=2
    )

    print("Finished loading {} training samples".format(len(train_dataset)), flush=True)

    # Create the DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )

    # Initialize model and trainer
    model = MedSynthGANModule(lr=args.learning_rate, lr_d=args.learning_rate_discriminator,
                              lambda_grad=args.lambda_grad, loss_type=args.loss_type)

    # Inference
    inference_callback = VolumeInferenceCallback(
        test_volume_path="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTs/AMOS_CT_2022_000001_0000.nii.gz",
        output_dir="inference_{}_{}_{}_{}".format(args.loss_type, args.learning_rate, args.learning_rate_discriminator, args.lambda_grad),
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

    # Train the model
    trainer.fit(model, train_dataloader)

    return model


if __name__ == "__main__":
    main(sys.argv[1:])
