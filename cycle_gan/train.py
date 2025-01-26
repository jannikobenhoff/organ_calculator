import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import torchvision.transforms as T
import itertools
import torchvision.utils as vutils

from dataset import  Nifti2DDataset
from models import Discriminator, GeneratorResNet

print("Starting CycleGAN training...", flush=True)

IMG_SIZE = 256
transform = T.Compose([
    # Convert from [0,1] float => scale to [-1, 1]
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),  # shape [C,H,W], and scales to [0,1] if input was [0,255]
    T.Normalize(mean=[0.5], std=[0.5])  # single-channel => [-1, 1]
])

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        # For BatchNorm or InstanceNorm with affine=True, we can initialize gamma/beta
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

criterion_GAN = nn.MSELoss()  # or BCELoss
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Hyperparameters
batch_size = 2
lr = 2e-4
n_epochs = 101
lambda_cycle = 10.0  # Weight for cycle loss
lambda_identity = 5.0 # Weight for identity loss (sometimes 0.5 * lambda_cycle)

# Dataloaders
root_ct_train = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5008_AMOS_CT_2022/imagesTr/"
root_mri_train = "/midtier/sablab/scratch/data/jannik_data/synth_data/Dataset5009_AMOS_MR_2022/imagesTr/"

train_dataset = Nifti2DDataset(
    ct_dir=root_ct_train,
    mri_dir=root_mri_train,
    transform=transform,
    slice_axis=2,          # typically axial
    min_max_normalize=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,   # small batch for 2D slices
    shuffle=True
)

print("Number of training samples:", len(train_dataset))

# Confirm we get something
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
optimizer_D_mri = optim.Adam(D_mri.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_ct = optim.Adam(D_ct.parameters(), lr=lr, betas=(0.5, 0.999))

# Labels for real and fake
real_label = 1.0
fake_label = 0.0

for epoch in range(n_epochs):
    for i, (real_ct, real_mri) in enumerate(train_loader):
        real_ct = real_ct.to(device)
        real_mri = real_mri.to(device)
        
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        
        # Identity loss (optional): G_ct2mri(mri) should be mri if already MRI
        # Similarly for G_mri2ct(ct) -> ct
        identity_mri = G_ct2mri(real_mri)
        loss_id_mri = criterion_identity(identity_mri, real_mri) * lambda_identity
        
        identity_ct = G_mri2ct(real_ct)
        loss_id_ct = criterion_identity(identity_ct, real_ct) * lambda_identity

        # GAN loss
        fake_mri = G_ct2mri(real_ct)
        pred_fake_mri = D_mri(fake_mri)
        loss_GAN_ct2mri = criterion_GAN(pred_fake_mri, torch.ones_like(pred_fake_mri))

        fake_ct = G_mri2ct(real_mri)
        pred_fake_ct = D_ct(fake_ct)
        loss_GAN_mri2ct = criterion_GAN(pred_fake_ct, torch.ones_like(pred_fake_ct))
        
        # Cycle loss
        rec_ct = G_mri2ct(fake_mri)
        loss_cycle_ct = criterion_cycle(rec_ct, real_ct) * lambda_cycle
        
        rec_mri = G_ct2mri(fake_ct)
        loss_cycle_mri = criterion_cycle(rec_mri, real_mri) * lambda_cycle
        
        # Total generator loss
        loss_G = loss_GAN_ct2mri + loss_GAN_mri2ct + loss_cycle_ct + loss_cycle_mri + loss_id_mri + loss_id_ct
        loss_G.backward()
        optimizer_G.step()
        
        # -----------------------
        #  Train Discriminator MRI
        # -----------------------
        optimizer_D_mri.zero_grad()

        # Real
        pred_real = D_mri(real_mri)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

        # Fake
        fake_mri_detach = fake_mri.detach()
        pred_fake = D_mri(fake_mri_detach)
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D_mri = (loss_D_real + loss_D_fake) * 0.5
        loss_D_mri.backward()
        optimizer_D_mri.step()
        
        # -----------------------
        #  Train Discriminator CT
        # -----------------------
        optimizer_D_ct.zero_grad()

        # Real
        pred_real = D_ct(real_ct)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        
        # Fake
        fake_ct_detach = fake_ct.detach()
        pred_fake = D_ct(fake_ct_detach)
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D_ct = (loss_D_real + loss_D_fake) * 0.5
        loss_D_ct.backward()
        optimizer_D_ct.step()

        if i % 10 == 0:
            print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] "
                  f"[D_mri: {loss_D_mri.item():.4f}, D_ct: {loss_D_ct.item():.4f}] "
                  f"[G: {loss_G.item():.4f}] ")
            
        # if i % 200 == 0 and :  # e.g., save every 200 batches
        #     # Suppose fake_mri is your generated MRI: shape [B, 1, H, W]
        #     # Convert it to a grid and save as a PNG
        #     out_path = f"epoch_{epoch}_batch_{i}_fakeMRI.png"
            
        #     # Denormalize if you used T.Normalize(mean=[0.5], std=[0.5]) etc.
        #     # A quick denormalization can be done in code or you can just accept that
        #     # the images are in [-1, 1].
            
        #     vutils.save_image(fake_mri, out_path, normalize=True, range=(-1, 1))
        #     print(f"Saved synthesized MRI sample to {out_path}")
        
        if i % 100 == 0 and i > 0:
            checkpoint_path = "checkpoints/cyclegan_epoch_{:03d}.pth".format(epoch)
            torch.save({
                'epoch': epoch,
                'G_ct2mri_state_dict': G_ct2mri.state_dict(),
                'G_mri2ct_state_dict': G_mri2ct.state_dict(),
                # (Optionally) Save discriminators if you want:
                'D_mri_state_dict': D_mri.state_dict(),
                'D_ct_state_dict': D_ct.state_dict(),
                # (Optionally) Save optimizers:
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_mri_state_dict': optimizer_D_mri.state_dict(),
                'optimizer_D_ct_state_dict': optimizer_D_ct.state_dict(),
            }, checkpoint_path)

            print(f"Saved checkpoint to {checkpoint_path}")