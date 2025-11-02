"""
U-Net architecture for diffusion model and auxiliary classifier for guidance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic residual block with group normalization."""
    
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, down=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.up = up
        self.down = down
        
        # For up blocks, in_ch is the concatenated channels (upsampled + skip)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU()
        
        if up:
            self.upsample = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        elif down:
            self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        # Spatial transform
        if self.up:
            return self.upsample(h)
        elif self.down:
            return self.downsample(h)
        else:
            return h


class UNet(nn.Module):
    """
    U-Net model for denoising diffusion.
    
    Args:
        img_channels: Number of input image channels (1 for MNIST)
        time_dim: Dimension of time embeddings
    """
    
    def __init__(self, img_channels=1, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Downsampling
        self.conv0 = nn.Conv2d(img_channels, 64, 3, padding=1)
        
        self.down1 = Block(64, 128, time_dim, down=True)
        self.down2 = Block(128, 256, time_dim, down=True)
        
        # Bottleneck (no spatial change)
        self.bot1 = Block(256, 512, time_dim)
        self.bot2 = Block(512, 512, time_dim)
        self.bot3 = Block(512, 256, time_dim)
        
        # Upsampling
        self.upsample1 = nn.ConvTranspose2d(256, 256, 4, 2, 1)  # 7x7 -> 14x14
        self.up1 = Block(256 + 128, 128, time_dim)  # 384 input (256 upsampled + 128 from r2)
        
        self.upsample2 = nn.ConvTranspose2d(128, 128, 4, 2, 1)  # 14x14 -> 28x28
        self.up2 = Block(128 + 64, 64, time_dim)    # 192 input (128 upsampled + 64 from r1)
        
        self.out = nn.Conv2d(64, img_channels, 1)

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv0(x)
        
        # Downsampling
        r1 = x
        x = self.down1(x, t)
        r2 = x
        x = self.down2(x, t)
        
        # Bottleneck
        x = self.bot1(x, t)
        x = self.bot2(x, t)
        x = self.bot3(x, t)
        
        # Upsampling with skip connections
        x = self.upsample1(x)  # Upsample first
        x = torch.cat((x, r2), dim=1)  # Then concatenate
        x = self.up1(x, t)
        
        x = self.upsample2(x)  # Upsample first
        x = torch.cat((x, r1), dim=1)  # Then concatenate
        x = self.up2(x, t)
        
        return self.out(x)


class Classifier(nn.Module):
    """
    Simple CNN classifier for MNIST digit classification.
    Used for classifier-guided diffusion.
    
    Args:
        num_classes: Number of output classes (10 for MNIST)
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # After 3 pooling layers: 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
