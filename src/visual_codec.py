import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
    def forward(self, x):
        return x + self.net(x)

class VisualCodec(nn.Module):
    def __init__(self, codebook_size=16384, latent_dim=256, img_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2),  # 128
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),  # 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),  # 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 3, padding=1),
            nn.BatchNorm2d(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # 256
            nn.Sigmoid()
        )
        # Codebook (Gumbel-Softmax)
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.codebook = nn.Embedding(codebook_size, latent_dim)

    def encode(self, x):
        z = self.encoder(x)  # (B, D, 32, 32)
        z = z.permute(0, 2, 3, 1).contiguous()  # (B, 32, 32, D)
        # Straight-through Gumbel-Softmax
        logits = torch.einsum('bhwc,kc->bhwk', z, self.codebook.weight)
        soft_one_hot = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)
        z_q = torch.einsum('bhwk,kc->bhwc', soft_one_hot, self.codebook.weight)
        return z_q.permute(0, 3, 1, 2)  # (B, D, 32, 32)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z