import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        half_dim = self.n_channels // 8
        emb = torch.log(torch.Tensor([10000.0])) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, n_channels, n_heads=1, d_k=None, n_groups=8):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_channels=32, ch_mults=(1, 2, 4, 8), is_attn=(False, False, True, True)):
        super().__init__()

        self.image_proj = nn.Conv2d(in_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        time_channels = n_channels * 4
        self.time_emb = TimeEmbedding(time_channels)

        # --- Encoder ---
        self.down1 = nn.Sequential(
            ResidualBlock(n_channels, n_channels, time_channels),
            ResidualBlock(n_channels, n_channels, time_channels)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * ch_mults[1], kernel_size=2, stride=2),
            ResidualBlock(n_channels * ch_mults[1], n_channels * ch_mults[1], time_channels),
            ResidualBlock(n_channels * ch_mults[1], n_channels * ch_mults[1], time_channels)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(n_channels * ch_mults[1], n_channels * ch_mults[2], kernel_size=2, stride=2),
            ResidualBlock(n_channels * ch_mults[2], n_channels * ch_mults[2], time_channels),
            AttentionBlock(n_channels*ch_mults[2]),
            ResidualBlock(n_channels * ch_mults[2], n_channels * ch_mults[2], time_channels),
        )

        # --- Middle ---
        self.middle = nn.Sequential(
            ResidualBlock(n_channels * ch_mults[2], n_channels * ch_mults[2], time_channels),
            AttentionBlock(n_channels * ch_mults[2]),
            ResidualBlock(n_channels * ch_mults[2], n_channels * ch_mults[2], time_channels)
        )

        # --- Decoder ---
        self.up1 = nn.Sequential(
            ResidualBlock(n_channels * ch_mults[2] * 2, n_channels * ch_mults[2], time_channels),
            AttentionBlock(n_channels * ch_mults[2]),
            ResidualBlock(n_channels * ch_mults[2], n_channels * ch_mults[2], time_channels),
            nn.ConvTranspose2d(n_channels * ch_mults[2], n_channels * ch_mults[1], kernel_size=2, stride=2)
        )
        self.up2 = nn.Sequential(
            ResidualBlock(n_channels * ch_mults[1] * 2, n_channels * ch_mults[1], time_channels),
            ResidualBlock(n_channels * ch_mults[1], n_channels * ch_mults[1], time_channels),
            nn.ConvTranspose2d(n_channels * ch_mults[1], n_channels, kernel_size=2, stride=2)
        )
        self.up3 = nn.Sequential(
            ResidualBlock(n_channels * 2, n_channels, time_channels),
            ResidualBlock(n_channels, n_channels, time_channels)
        )

        # --- Final ---
        self.final = nn.Conv2d(n_channels, out_channels, kernel_size=(3, 3), padding=1)

    def _forward_sequential(self, seq, x, t):
        for layer in seq:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        return x

    def forward(self, x, t):
        t = self.time_emb(t)
        x = self.image_proj(x)

        # Encoder
        s1 = self._forward_sequential(self.down1, x, t)
        s2 = self._forward_sequential(self.down2, s1, t)
        s3 = self._forward_sequential(self.down3, s2, t)

        # Middle
        x = self._forward_sequential(self.middle, s3, t)

        # Decoder
        x = self._forward_sequential(self.up1, torch.cat([x, s3], dim=1), t)
        x = self._forward_sequential(self.up2, torch.cat([x, s2], dim=1), t)
        x = self._forward_sequential(self.up3, torch.cat([x, s1], dim=1), t)

        return self.final(x)

class Classifier(nn.Module):
    def __init__(self, in_channels=3, n_channels=64, out_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.out_classes = out_classes

        self.conv1 = nn.Conv2d(in_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_channels, n_channels * 2, kernel_size=(3, 3), padding=(1, 1))
        self.fc1 = nn.Linear(n_channels * 2 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.n_channels * 2 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
