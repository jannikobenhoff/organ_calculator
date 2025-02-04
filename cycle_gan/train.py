import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import torchvision.transforms as T
import itertools
from dataset import  Nifti2DDataset
from models import Discriminator, GeneratorResNet
from losses import Grad 


IMG_SIZE = 256
nifit_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Lambda(lambda x: torch.clamp(x, -200, 500)),  # Ensure no extreme outliers
    T.Normalize(mean=[150], std=[350]),  # Normalize [-200,500] → [-1,1]
])


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

if __name__ == "__main__":
    print("Starting CycleGAN training...", flush=True)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_grad = Grad(penalty='l1')

    # Hyperparameters
    batch_size = 2
    n_epochs = 10
    lambda_cycle = 7.5  # Weight for cycle loss
    lambda_identity = 2.5 # Weight for identity loss
    lambda_grad = 0.05  # Weight for gradient loss
    lr_d = 2e-3  # Discriminator learning rate
    lr = 5e-5  # Optimizer learning rate

    # Dataloaders
    root_ct_train = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/"
    root_mri_train = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/imagesTr/"
    checkpoint_directory = "checkpoints"

    os.makedirs(checkpoint_directory, exist_ok=True)

    train_dataset = Nifti2DDataset(
        ct_dir=root_ct_train,
        mri_dir=root_mri_train,
        transform=nifit_transform,
        slice_axis=2,
        normalize=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    print("Number of training samples:", len(train_dataset), flush=True)
    for i, (ct_slice, mri_slice) in enumerate(train_loader):
        print("CT slice shape:", ct_slice.shape)   # [B, 1, H, W]
        print("MRI slice shape:", mri_slice.shape) # [B, 1, H, W]
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    G_ct2mri = GeneratorResNet().to(device)
    G_mri2ct = GeneratorResNet().to(device)
    D_mri = Discriminator().to(device)
    D_ct = Discriminator().to(device)

    # Apply weight initialization
    G_ct2mri.apply(weights_init_normal)
    G_mri2ct.apply(weights_init_normal)
    D_mri.apply(weights_init_normal)
    D_ct.apply(weights_init_normal)

    # Optimizers
    optimizer_G = optim.Adam(
        itertools.chain(G_ct2mri.parameters(), G_mri2ct.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    
    optimizer_D_mri = optim.Adam(D_mri.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizer_D_ct = optim.Adam(D_ct.parameters(), lr=lr_d, betas=(0.5, 0.999))

    for epoch in range(n_epochs):
        for i, (real_ct, real_mri) in enumerate(train_loader):
            real_ct = real_ct.to(device)
            real_mri = real_mri.to(device)
            
            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()
            
            # Identity loss (G_ct2mri should return 1 for MRI and vice versa)
            identity_mri_field = G_ct2mri(real_mri)  
            identity_mri = real_mri * identity_mri_field  
            loss_id_mri = criterion_identity(identity_mri, real_mri) * lambda_identity
            
            identity_ct_field = G_mri2ct(real_ct)
            identity_ct = real_ct * identity_ct_field  
            loss_id_ct = criterion_identity(identity_ct, real_ct) * lambda_identity

            # Forward pass: Generate scalar fields and transformed images
            scale_field_ct2mri = G_ct2mri(real_ct)  # CT → MRI field
            scale_field_mri2ct = G_mri2ct(real_mri)  # MRI → CT field

            fake_mri = (real_ct * scale_field_ct2mri)  # CT × field → synthetic MRI
            fake_ct = (real_mri * scale_field_mri2ct)  # MRI × field → synthetic CT

            # GAN loss
            pred_fake_mri = D_mri((fake_mri + 1) / 2)  # Normalize to [0,1]
            loss_GAN_ct2mri = criterion_GAN(pred_fake_mri, torch.ones_like(pred_fake_mri))

            pred_fake_ct = D_ct((fake_ct + 1) / 2)
            loss_GAN_mri2ct = criterion_GAN(pred_fake_ct, torch.ones_like(pred_fake_ct))

            # Cycle consistency loss
            rec_ct = G_mri2ct(fake_mri) * fake_mri  # Recovered CT = MRI2CT Field × fake MRI
            loss_cycle_ct = criterion_cycle(rec_ct, real_ct) * lambda_cycle

            rec_mri = G_ct2mri(fake_ct) * fake_ct  # Recovered MRI = CT2MRI Field × fake CT
            loss_cycle_mri = criterion_cycle(rec_mri, real_mri) * lambda_cycle

            # Compute Grad2D Loss (encourages smooth transformation fields)
            loss_grad_ct2mri = criterion_grad.loss(None, scale_field_ct2mri) * lambda_grad
            loss_grad_mri2ct = criterion_grad.loss(None, scale_field_mri2ct) * lambda_grad

            # Total generator loss (Including Grad regularization)
            loss_G = (loss_GAN_ct2mri + loss_GAN_mri2ct + 
                    loss_cycle_ct + loss_cycle_mri + 
                    loss_id_mri + loss_id_ct + 
                    loss_grad_ct2mri + loss_grad_mri2ct)
            loss_G.backward()
            optimizer_G.step()
            
            # -----------------------
            #  Train Discriminator
            # -----------------------
            optimizer_D_mri.zero_grad()
            pred_real_mri = D_mri((real_mri + 1) / 2)  # Normalize to [0,1]
            loss_D_real_mri = criterion_GAN(pred_real_mri, torch.ones_like(pred_real_mri))
            loss_D_fake_mri = criterion_GAN(D_mri((fake_mri.detach() + 1) / 2), torch.zeros_like(pred_real_mri))
            loss_D_mri = (loss_D_real_mri + loss_D_fake_mri) * 0.5
            loss_D_mri.backward()
            optimizer_D_mri.step()

            optimizer_D_ct.zero_grad()
            pred_real_ct = D_ct((real_ct + 1) / 2)
            loss_D_real_ct = criterion_GAN(pred_real_ct, torch.ones_like(pred_real_ct))
            loss_D_fake_ct = criterion_GAN(D_ct((fake_ct.detach() + 1) / 2), torch.zeros_like(pred_real_ct))
            loss_D_ct = (loss_D_real_ct + loss_D_fake_ct) * 0.5
            loss_D_ct.backward()
            optimizer_D_ct.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] "
                    f"[D_mri: {loss_D_mri.item():.6f}, D_ct: {loss_D_ct.item():.6f}] "
                    f"[G: {loss_G.item():.4f}, Grad_CT2MRI: {loss_grad_ct2mri.item():.4f}, Grad_MRI2CT: {loss_grad_mri2ct.item():.4f}]")

                print(f"Mean Scalar Field CT->MRI: {scale_field_ct2mri.mean().item():.4f}, MRI->CT: {scale_field_mri2ct.mean().item():.4f}")

                # Log min/max values
                print(f"Min Scalar Field CT->MRI: {scale_field_ct2mri.min().item():.4f}, Max: {scale_field_ct2mri.max().item():.4f}")
                print(f"Min Scalar Field MRI->CT: {scale_field_mri2ct.min().item():.4f}, Max: {scale_field_mri2ct.max().item():.4f}")
   
            # if i % 200 == 0 and :  # e.g., save every 200 batches
            #     # Suppose fake_mri is your generated MRI: shape [B, 1, H, W]
            #     # Convert it to a grid and save as a PNG
            #     out_path = f"epoch_{epoch}_batch_{i}_fakeMRI.png"
                
            #     # Denormalize if you used T.Normalize(mean=[0.5], std=[0.5]) etc.
            #     # A quick denormalization can be done in code or you can just accept that
            #     # the images are in [-1, 1].
                
            #     vutils.save_image(fake_mri, out_path, normalize=True, range=(-1, 1))
            #     print(f"Saved synthesized MRI sample to {out_path}")
            

        checkpoint_path = f"{checkpoint_directory}/cyclegan_epoch_{epoch:03d}.pth"
        torch.save({
            'epoch': epoch,
            'G_ct2mri_state_dict': G_ct2mri.state_dict(),
            'G_mri2ct_state_dict': G_mri2ct.state_dict(),
            'D_mri_state_dict': D_mri.state_dict(),
            'D_ct_state_dict': D_ct.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_mri_state_dict': optimizer_D_mri.state_dict(),
            'optimizer_D_ct_state_dict': optimizer_D_ct.state_dict(),
        }, checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")