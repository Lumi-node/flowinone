import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPProcessor


class CrossModalVisualMapper(nn.Module):
    def __init__(self, num_tokens=4096, token_dim=512, num_layers=6, num_heads=8):
        super().__init__()
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.mapper = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.to_tokens = nn.Linear(512, token_dim)  # CLIP to internal dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, token_dim))
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.grid_size = 64  # 64x64 -> 4096

    def forward(self, input_ids, attention_mask=None):
        # Get CLIP text embeddings
        clip_out = self.clip_text_model(input_ids=input_ids, attention_mask=attention_mask)
        x = clip_out.last_hidden_state  # (B, T, 512)
        x = self.to_tokens(x)  # (B, T, D)

        # Expand to full token grid
        B, T, D = x.shape
        x = x.repeat(1, self.num_tokens // T + 1, 1)[:, :self.num_tokens]
        x = x + self.pos_embedding
        x = self.mapper(x)  # (B, 4096, 512)
        return x.view(B, self.grid_size, self.grid_size, D).permute(0, 3, 1, 2)  # (B, D, 64, 64)