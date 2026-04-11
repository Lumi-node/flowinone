import torch
import torch.nn as nn


def signed_distance_transform(coords, char_pos, scale=1.0):
    """Compute signed distance from character center."""
    return -torch.cdist(coords, char_pos) * scale


class DifferentiableRasterizer(nn.Module):
    def __init__(self, img_size=256, stroke_width=1.5, blur_radius=1.0):
        super().__init__()
        self.img_size = img_size
        self.stroke_width = nn.Parameter(torch.tensor(stroke_width))
        self.blur_radius = nn.Parameter(torch.tensor(blur_radius))
        # Create grid coordinates
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, img_size),
            torch.linspace(-1, 1, img_size),
            indexing='ij'
        )
        self.register_buffer('grid_coords', torch.stack([x, y], dim=-1).view(-1, 2))

    def forward(self, char_positions, char_embeddings=None):
        """
        Render soft masks for characters.
        Args:
            char_positions: (N, 2) tensor of normalized character coordinates
            char_embeddings: ignored (position-only in this version)
        Returns:
            (H, W) soft mask
        """
        distances = signed_distance_transform(self.grid_coords, char_positions)
        # Signed distance function with learnable parameters
        mask = torch.sigmoid(
            (self.stroke_width - distances.abs()) / self.blur_radius
        )
        mask = mask.max(dim=1, keepdim=True)[0]  # Max over characters
        return mask.view(1, self.img_size, self.img_size)