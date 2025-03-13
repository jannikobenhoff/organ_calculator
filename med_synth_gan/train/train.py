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


class MedSynthGANModule(pl.LightningModule):
    def __init__(self, lr, lr_d, lambda_grad, loss_type="mse"):
        super().__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.lr_d = lr_d
        self.lambda_grad = lambda_grad
        self.loss_type = loss_type

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

        if self.loss_type == "hinge":
            # Train Discriminator
            opt_d.zero_grad()
            fake_mri, _ = self.G_ct2mri(real_ct)

            # Real MRI classification
            pred_real_mri = self.D_mri(real_mri)
            real_labels = torch.ones_like(pred_real_mri)  # Previously 1.0
            real_labels[real_labels == 1] = 1  # Ensure positive class is +1

            loss_D_real = self.criterion_GAN(pred_real_mri, real_labels)

            # Fake MRI classification
            pred_fake_mri = self.D_mri(fake_mri.detach())
            fake_labels = torch.zeros_like(pred_fake_mri)  # Previously 0.0
            fake_labels[fake_labels == 0] = -1  # Ensure negative class is -1

            loss_D_fake = self.criterion_GAN(pred_fake_mri, fake_labels)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            self.manual_backward(loss_D)
            opt_d.step()
        else:
            # Train Discriminator
            opt_d.zero_grad()
            fake_mri, _ = self.G_ct2mri(real_ct)
            pred_real_mri = self.D_mri(real_mri)
            if self.loss_type == "bce":
                real_labels_smooth = torch.full_like(pred_real_mri, 0.9)  # instead of 1.0
                loss_D_real = self.criterion_GAN(pred_real_mri, real_labels_smooth)
            else:
                # MSE
                loss_D_real = self.criterion_GAN(pred_real_mri, torch.ones_like(pred_real_mri))


            pred_fake_mri = self.D_mri(fake_mri.detach())
            loss_D_fake = self.criterion_GAN(pred_fake_mri, torch.zeros_like(pred_fake_mri))

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            self.manual_backward(loss_D)

            #torch.nn.utils.clip_grad_norm_(self.D_mri.parameters(), 0.1)
            opt_d.step()


        if batch_idx % 100 == 0:
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

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.G_ct2mri.parameters(),
            lr=self.lr, betas=(0.9, 0.95), weight_decay=0.001  # 0.01
        )

        opt_d = torch.optim.AdamW(self.D_mri.parameters(), lr=self.lr_d, betas=(0.9, 0.95), weight_decay=0.001)  # 0.01


        # opt_g = torch.optim.Adam(
        #     self.G_ct2mri.parameters(),
        #     lr=self.lr, betas=(0.5, 0.999)
        # )
        # opt_d = torch.optim.Adam(self.D_mri.parameters(), lr=self.lr_d, betas=(0.5, 0.999))

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.98)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.98)

        # 3) Return them in the correct Lightning format
        return (
            [opt_g, opt_d],
            [
                {"scheduler": scheduler_g, "interval": "epoch", "frequency": 1, "name": "lr_g"},
                {"scheduler": scheduler_d, "interval": "epoch", "frequency": 1, "name": "lr_d"},
            ]
        )

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
        default=0, # 1e-5,
        type=float,
        help="Weight for total-variation (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=4e-5, #5e-5
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-lr_d",
        "--learning-rate-discriminator",
        default=5e-5, # should be larger than Generator for MSE 1e-4
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-loss_type",
        "--loss_type",
        default="bce",  # bce, mse, hinge
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
        output_dir="inference_{}_{}_{}".format(args.loss_type, args.learning_rate, args.learning_rate_discriminator),
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
