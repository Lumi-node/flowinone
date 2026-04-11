import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorVelocityUNet(nn.Module):
    """
    U-Net architecture for predicting vector-valued velocity fields in flow matching.
    Conditioned on time and high-resolution visual features.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding layer
        self.time_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 512)

        # Decoder
        self.dec4 = self._upconv_block(512 + 512, 256)  # skip connection from enc4
        self.dec3 = self._upconv_block(256 + 256, 128)     # skip connection from enc3
        self.dec2 = self._upconv_block(128 + 128, 64)      # skip connection from enc2
        self.dec1 = self._upconv_block(64 + 64, 64)        # skip connection from enc1

        # Final conv to get output channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        """
        Convolutional block with two 3x3 convs and ReLU activations.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_channels, out_channels):
        """
        Transposed convolution block for upsampling.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _get_features(self, x: torch.Tensor, t: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through encoder to extract feature maps for skip connections.
        Time is embedded and added to each feature map.
        """
        # Embed time
        t_emb = self.time_embed(t)
        t_emb = t_emb.view(t_emb.size(0), -1, 1, 1)  # (B, 64, 1, 1)

        # Ensure x has batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        features = []

        # Encoders with time injection
        x1 = self.enc1(x)
        x1 = x1 + F.interpolate(t_emb, size=x1.shape[-2:], mode='bilinear')
        features.append(x1)

        x2 = self.enc2(F.max_pool2d(x1, 2))
        x2 = x2 + F.interpolate(t_emb, size=x2.shape[-2:], mode='bilinear')
        features.append(x2)

        x3 = self.enc3(F.max_pool2d(x2, 2))
        x3 = x3 + F.interpolate(t_emb, size=x3.shape[-2:], mode='bilinear')
        features.append(x3)

        x4 = self.enc4(F.max_pool2d(x3, 2))
        x4 = x4 + F.interpolate(t_emb, size=x4.shape[-2:], mode='bilinear')
        features.append(x4)

        return features

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet.

        Args:
            x: Input image (B, 3, H, W)
            t: Time scalar (B,)
            cond: Conditioning from VisualPromptEncoder (B, H, W, 256)

        Returns:
            Predicted velocity field (B, 3, H, W)
        """
        # Get encoder features
        enc_features = self._get_features(x, t)

        # Bottleneck
        x = enc_features[-1]  # Already at bottleneck level, no extra pooling
        x = self.bottleneck(x)  # (1, 512, 64, 64)

        # Decoder with skip connections
        x = self.dec4(torch.cat([x, enc_features[-1]], dim=1))  # (1, 1024, 64, 64) -> (1, 256, 128, 128)
        x = self.dec3(torch.cat([x, enc_features[-2]], dim=1))  # (1, 512, 128, 128) -> (1, 128, 256, 256)
        x = self.dec2(torch.cat([x, enc_features[-3]], dim=1))  # (1, 256, 256, 256) -> (1, 64, 512, 512)
        x = self.dec1(torch.cat([x, enc_features[-4]], dim=1))  # (1, 128, 512, 512) -> (1, 64, 512, 512)

        # Final output
        x = self.final_conv(x)
        return x
