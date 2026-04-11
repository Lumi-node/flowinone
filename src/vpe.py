import torch
import torch.nn as nn
from torchvision.models import resnet34


class VisualPromptEncoder(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()
        self.img_size = img_size
        # Local encoder
        self.backbone = resnet34(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final layer

        # ViT for global context
        self.vit = nn.Transformer(
            d_model=768,
            nhead=12,
            num_encoder_layers=12
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, img_size//16, 768))
        self.projection = nn.Linear(512, 768)  # ResNet to ViT dim
        self.color_head = nn.Linear(768, 3 * 16 * 16)  # To patch
        self.unpatchify = lambda x: torch.nn.functional.fold(
            x.transpose(1, 2), output_size=(img_size, img_size), kernel_size=16, stride=16
        )

    def forward(self, x, text_layout=None, rasterizer=None):
        B, C, H, W = x.shape
        h = self.backbone.conv1(x)
        h = self.backbone.bn1(h)
        h = self.backbone.relu(h)
        h = self.backbone.maxpool(h)
        h = self.backbone.layer1(h)
        h = self.backbone.layer2(h)
        h = self.backbone.layer3(h)  # (B, 256, H/8, W/8)
        h = self.projection(h.permute(0,2,3,1)).flatten(1,2)  # (B, S, 768)
        h = h + self.pos_embedding
        h = self.vit(h)
        
        # Expand to full image
        h = self.color_head(h)  # (B, S, 3*256)
        h = h.view(B, -1, 3, 16, 16)
        h = h.flatten(1, 2)  # (B, S*256, 3)
        h = h.permute(0, 2, 1).contiguous()
        h = self.unpatchify(h)
        h = h + x  # Residual connection with input sketch+text
        return torch.sigmoid(h)