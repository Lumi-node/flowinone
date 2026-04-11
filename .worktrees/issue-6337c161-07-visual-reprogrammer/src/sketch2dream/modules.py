from typing import Dict, List
import torch
import torch.nn as nn


class VisualReprogrammer(nn.Module):
    """
    Maps atomic symbolic inputs (e.g., {'label': 'tree', 'bbox': [x0,y0,x1,y1]})
    into a full visual prompt image using SketchRasterizer.
    """
    def __init__(self, image_size: int = 512):
        super().__init__()
        self.image_size = image_size
        # We'll use SketchRasterizer from sketch-rasterizer package
        # Assuming it's importable as `from sketch_rasterizer import SketchRasterizer`
        try:
            from sketch_rasterizer import SketchRasterizer
            self.rasterizer = SketchRasterizer(image_size=image_size)
        except ImportError:
            raise ImportError(
                "sketch_rasterizer not found. Install with: pip install sketch-rasterizer"
            )

    def forward(self, symbol: Dict) -> torch.Tensor:
        """
        Convert a single symbol dict into a visual prompt.

        Args:
            symbol (Dict): Symbol with 'label' and 'bbox' keys, or 'type' and 'bbox'.

        Returns:
            torch.Tensor: (1, 3, 512, 512) visual prompt
        """
        # Wrap single symbol in batch and page structure expected by SketchRasterizer
        symbols_batch = [[symbol]]  # Shape: (B=1, N_symbols)
        image = self.rasterizer(symbols_batch)  # (1, 3, 512, 512)
        return image
