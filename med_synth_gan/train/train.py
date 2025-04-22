import argparse
import sys
import torch
import torch.nn as nn
# import pytorch_lightning as pl
from torch.utils.data import DataLoader
from med_synth_gan.dataset.ct_mri_2d_dataset import CtMri2DDataset
from med_synth_gan.models.models import UNet
from med_synth_gan.models.cycle_gan import Discriminator
from med_synth_gan.models.losses import Grad
from med_synth_gan.inference.inference import VolumeInference
# from pytorch_lightning.callbacks.progress import TQDMProgressBar
import random
import kornia.augmentation as K
from tqdm import tqdm

def random_flip_rot_crop(batch,):
    B, C, H, W = batch.shape

    augmented_images = []
    for i in range(B):
        img = batch[i]

        # Random horizontal flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])  # Flip width dimension

        # Random vertical flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])  # Flip height dimension

        # Random rotation by 0, 90, 180, or 270 degrees
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, dims=(1, 2))

        # After rotation, update image dimensions
        _, h_img, w_img = img.shape

        augmented_images.append(img)

    augmented_batch = torch.stack(augmented_images)
    return augmented_batch

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class MedSynthGANModule(nn.Module):
    def __init__(self, lr, lr_d, lambda_grad, loss_type="mse"):
        super().__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.lr_d = lr_d
        self.lambda_grad = lambda_grad
        self.loss_type = loss_type
        #self.disc_step = 0
        #self.save_hyperparameters()

        # Models
        self.G_ct2mri = UNet()
        self.D_mri = Discriminator()

        self.opt_g = torch.optim.AdamW(self.G_ct2mri.parameters(), lr=lr)
        self.opt_d = torch.optim.AdamW(self.D_mri.parameters(), lr=lr_d)

        # Loss functions
        if loss_type == "bce":
            self.criterion_GAN = nn.BCEWithLogitsLoss()
        elif loss_type == "mse":
            self.criterion_GAN = nn.MSELoss()
        elif loss_type == "hinge":
            self.criterion_GAN = nn.HingeEmbeddingLoss(margin=1)
        self.criterion_grad = Grad(penalty='l1')
        self.grad_scaler = torch.amp.GradScaler('cuda')

        H,W = 256,256
        self.aug = torch.nn.Sequential(
            K.RandomHorizontalFlip(p=0.3),
            K.RandomVerticalFlip(p=0.3),
            K.RandomRotation(degrees=180.0,p=0.3),
            K.RandomResizedCrop(size=(H, W), scale=(0.8, 1.0),p=0.3),
            K.RandomPerspective(distortion_scale=0.5, p=0.3),
        ).cuda()

    def forward(self, ct_image):
        return self.G_ct2mri(ct_image)

    def generator_step(self, real_ct):
        with torch.autocast('cuda', enabled=True):
            fake_mri, scale_field = self.G_ct2mri(real_ct)
            pred_fake = self.D_mri(fake_mri)
            loss_gan = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_grad = self.criterion_grad.loss(None, scale_field) * self.lambda_grad
            loss_G = loss_gan + loss_grad

        self.opt_g.zero_grad()
        self.grad_scaler.scale(loss_G).backward()
        self.grad_scaler.step(self.opt_g)
        self.grad_scaler.update()
        return loss_G.item()

    def discriminator_step(self, real_ct, real_mri):
        with torch.autocast('cuda', enabled=True):
            fake_mri = self.G_ct2mri(real_ct[:4])[0].detach()
            fake_mri = self.aug(fake_mri)
            real_mri = self.aug(real_mri)

            pred_real = self.D_mri(real_mri)
            loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = self.D_mri(fake_mri)
            loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_real + loss_fake) * 0.5

        self.opt_d.zero_grad()
        self.grad_scaler.scale(loss_D).backward()
        self.grad_scaler.step(self.opt_d)
        self.grad_scaler.update()
        return loss_D.item()
    #
    # def training_step(self, batch, batch_idx):
    #     if batch is None:
    #         return None
    #
    #     # Get optimizers
    #     opt_g, opt_d = self.optimizers()
    #     real_ct, real_mri = batch
    #     # set_requires_grad(self.D_mri, False)
    #     with (torch.autocast('cuda', enabled=True)):
    #         fake_mri, scale_field_ct2mri = self.G_ct2mri(real_ct)
    #         pred_fake_mri = self.D_mri(fake_mri)
    #         loss_GAN_ct2mri = self.criterion_GAN(pred_fake_mri, torch.ones_like(pred_fake_mri))
    #
    #         loss_grad_ct2mri = self.criterion_grad.loss(None, scale_field_ct2mri) * self.lambda_grad
    #         loss_G = loss_GAN_ct2mri + loss_grad_ct2mri
    #     opt_g.optimizer.zero_grad(set_to_none=True)
    #     self.grad_scaler.scale(loss_G).backward()
    #     self.grad_scaler.unscale_(opt_g.optimizer)
    #     # network_norm = torch.nn.utils.clip_grad_norm_(self.G_ct2mri.parameters(), 1)
    #     self.grad_scaler.step(opt_g.optimizer)
    #     self.grad_scaler.update()
    #     opt_g.optimizer.zero_grad(set_to_none=True)
    #
    #     # if self.disc_step % 2 == 0:
    #     # Train Discriminator
    #     # set_requires_grad(self.D_mri, True)
    #     with (torch.autocast('cuda', enabled=True)):
    #         fake_mri, _ = self.G_ct2mri(real_ct[:4])
    #
    #         fake_mri = self.aug(fake_mri)
    #         real_mri = self.aug(real_mri)
    #
    #         pred_real_mri = self.D_mri(real_mri)
    #         loss_D_real = self.criterion_GAN(pred_real_mri, torch.ones_like(pred_real_mri))
    #         pred_fake_mri = self.D_mri(fake_mri.detach())
    #         loss_D_fake = self.criterion_GAN(pred_fake_mri, torch.zeros_like(pred_fake_mri))
    #         loss_D = (loss_D_real + loss_D_fake) * 0.5
    #
    #     opt_d.optimizer.zero_grad(set_to_none=True)
    #     self.grad_scaler.scale(loss_D).backward()
    #     self.grad_scaler.unscale_(opt_d.optimizer)
    #     network_norm = torch.nn.utils.clip_grad_norm_(self.D_mri.parameters(), 1)
    #     self.grad_scaler.step(opt_d.optimizer)
    #     self.grad_scaler.update()
    #     opt_d.optimizer.zero_grad(set_to_none=True)
    #
    #     if batch_idx % 10 == 0:
    #         self.log('loss_G', loss_G, prog_bar=True)
    #         self.log('loss_D', loss_D, prog_bar=True)
    #         self.log('scalar_field_mean', scale_field_ct2mri.mean(), prog_bar=True)
    #         self.log('scalar_field_min', scale_field_ct2mri.min(), prog_bar=True)
    #         self.log('scalar_field_max', scale_field_ct2mri.max(), prog_bar=True)
    #         #self.log('tv_loss', loss_grad_ct2mri, prog_bar=True)
    #         self.log('lr_d', self.lr_d, prog_bar=True)
    #         self.log('lr_g', self.lr, prog_bar=True)
    #
    #     self.disc_step += 1
    #     self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += 1

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.G_ct2mri.parameters(),
            lr=self.lr, betas=(0.5, 0.999), weight_decay=0  # 0.01
        )

        opt_d = torch.optim.AdamW(self.D_mri.parameters(), lr=self.lr_d, betas=(0.5, 0.999), weight_decay=0)  # 0.01

        return (
            [opt_g, opt_d]
        )

# class CustomProgressBar(TQDMProgressBar):
#     def __init__(self):
#         super().__init__(refresh_rate=10, leave=True, process_position=0)  # Update every 10 batches
#
#     def init_train_tqdm(self):
#         bar = super().init_train_tqdm()
#         bar.set_description("Training")
#         return bar
#
#     def get_metrics(self, *args, **kwargs):
#         items = super().get_metrics(*args, **kwargs)
#         return {
#             "loss_G": items.get("loss_G", ""),
#             "loss_D": items.get("loss_D", ""),
#             "sf_mean": items.get("scalar_field_mean", ""),
#             "sf_min": items.get("scalar_field_min", ""),
#             "sf_max": items.get("scalar_field_max", ""),
#             #"lr_g": items.get("lr_g", ""),
#             #"lr_d": items.get("lr_d", ""),
#             # "tv_loss": items.get("tv_loss", ""),
#         }

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
        default=50,
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
        default=5e-5 , #5e-5
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-lr_d",
        "--learning-rate-discriminator",
        default= 3e-5,  # should be larger than Generator for MSE 1e-4
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


# def main(argv):
#     args = parse_args(argv)
#
#     print("Starting MedSynthGAN training with args: {}".format(args), flush=True)
#
#     train_dataset = CtMri2DDataset(
#         ct_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/",
#         mri_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/t2Axial/",
#         slice_axis=2
#     )
#
#     print("Finished loading {} training samples".format(len(train_dataset)), flush=True)
#
#     # Create the DataLoader
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         num_workers=0,
#         shuffle=True,
#         collate_fn=collate_fn,
#         pin_memory=True,
#         # prefetch_factor=2
#     )
#
#     # Initialize model and trainer
#     model = MedSynthGANModule(lr=args.learning_rate, lr_d=args.learning_rate_discriminator,
#                               lambda_grad=args.lambda_grad, loss_type=args.loss_type)
#
#     # Inference
#     inference_callback = VolumeInferenceCallback(
#         test_volume_path="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTs/AMOS_CT_2022_000001_0000.nii.gz",
#         output_dir="inference_{}_{}_{}_{}".format(args.loss_type, args.learning_rate, args.learning_rate_discriminator, args.lambda_grad),
#     )
#
#     trainer = pl.Trainer(
#         default_root_dir='checkpoints',
#         max_epochs=args.epochs,
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         devices=1,
#         precision="32",
#         callbacks=[
#             CustomProgressBar(),
#             inference_callback
#         ],
#     )
#
#     # Train the model
#     trainer.fit(model, train_dataloader)
#
#     return model

def main(argv):
    args = parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Starting MedSynthGAN training with args: {}".format(args), flush=True)

    train_dataset = CtMri2DDataset(
        ct_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/",
        mri_dir="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/t2Axial/",
        slice_axis=2
    )

    # Initialize module
    model = MedSynthGANModule(
        loss_type=args.loss_type,
        lambda_grad=args.lambda_grad,
        lr=args.learning_rate,
        lr_d=args.learning_rate_discriminator,
    ).to(device)

    inferencer = VolumeInference(
        test_volume_path="/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTs/AMOS_CT_2022_000001_0000.nii.gz",
        output_dir="inference_{}_{}_{}_{}".format(args.loss_type, args.learning_rate, args.learning_rate_discriminator, args.lambda_grad),
        device=device
    )
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            # prefetch_factor=2
        )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for batch_idx, (real_ct, real_mri) in enumerate(progress_bar):
            real_ct = real_ct.to(device)
            real_mri = real_mri.to(device)

            # Generator update
            loss_G = model.generator_step(real_ct)

            # Discriminator update
            if batch_idx % 2 == 0:  # Update discriminator less frequently
                loss_D = model.discriminator_step(real_ct, real_mri)

            # Update progress bar
            progress_bar.set_postfix({
                'loss_G': f"{loss_G:.4f}",
                'loss_D': f"{loss_D:.4f}"
            })

        # Save checkpoint and run inference
        inferencer.run_inference(model, epoch)

    # Save final grid
    inferencer.save_final_grid()


if __name__ == "__main__":
    main(sys.argv[1:])
