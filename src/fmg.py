import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        h = self.norm(x)
        B, C, H, W = h.shape
        h = h.reshape(B, C, H*W).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        h = self.proj_out(h)
        return x + h

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=False):
        super().__init__()
        self.shortcuts = shortcut
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()
        if shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        if self.shortcuts:
            x = self.shortcut(x)
        return h + x

class FlowMatchingGenerator(nn.Module):
    def __init__(self, base_channels=256, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.conv_in = nn.Conv2d(2*base_channels, base_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        out_ch = base_channels
        for _ in range(num_scales):
            blocks = nn.ModuleList([
                ResidualBlock(in_ch, out_ch, shortcut=True),
                ResidualBlock(out_ch, out_ch),
                Downsample(out_ch)
            ])
            self.down_blocks.append(blocks)
            in_ch = out_ch
            out_ch *= 2

        # Middle
        mid_ch = in_ch
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(mid_ch, mid_ch),
            AttentionBlock(mid_ch),
            ResidualBlock(mid_ch, mid_ch)
        ])

        # Upsampling
        self.up_blocks = nn.ModuleList()
        out_ch = in_ch
        in_ch = out_ch + out_ch//2
        for _ in range(num_scales):
            blocks = nn.ModuleList([
                ResidualBlock(in_ch, out_ch, shortcut=True),
                ResidualBlock(out_ch, out_ch),
                ResidualBlock(out_ch, out_ch),
                Upsample(out_ch)
            ])
            self.up_blocks.append(blocks)
            out_ch = out_ch // 2
            in_ch = out_ch * 2

        self.conv_out = nn.Conv2d(base_channels, base_channels, 3, padding=1)

    def forward(self, x):
        # x: [noisy_latent, prompt_latent] with C=2*base_channels
        h = self.conv_in(x)

        # Downsample
        hs = []
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
            hs.append(h)

        # Middle
        for layer in self.middle_blocks:
            h = layer(h)

        # Upsample
        for i, block in enumerate(self.up_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                h = layer(h)

        return self.conv_out(h)