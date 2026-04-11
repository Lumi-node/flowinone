import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------
# 1. SketchRasterizer
# ---------------------

class SketchRasterizer(nn.Module):
    """
    Differentiable renderer using Bzier curves and affine transforms
    to render symbolic layout primitives to sketch-like images.
    """
    def __init__(self, image_size=512, num_channels=3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(0, image_size-1, image_size),
            torch.linspace(0, image_size-1, image_size),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=-1)  # (H, W, 2)
        self.register_buffer('grid', grid)

    def _bezier_curve(self, p0, p1, p2, p3, steps=100):
        """Cubic Bzier curve: (1-t)^3 p0 + 3(1-t)^2 t p1 + 3(1-t)t^2 p2 + t^3 p3"""
        t = torch.linspace(0, 1, steps, device=p0.device).view(-1, 1)  # (steps, 1)
        return (
            (1-t)**3 * p0 +
            3*(1-t)**2 * t * p1 +
            3*(1-t) * t**2 * p2 +
            t**3 * p3
        )  # (steps, 2)

    def _draw_line(self, start, end, width, color):
        """Draw antialiased line using distance transform"""
        grid = self.grid  # (H, W, 2)
        # Vector from start to end
        line_vec = end - start
        line_len = torch.norm(line_vec)
        if line_len < 1e-6:
            return torch.zeros(self.image_size, self.image_size, device=start.device)
        # Unit direction
        u = line_vec / line_len
        # Perpendicular unit vector
        n = torch.tensor([-u[1], u[0]], device=u.device)
        # Distance from each pixel to line
        to_pixel = grid - start.view(1, 1, 2)
        proj = torch.sum(to_pixel * u.view(1, 1, 2), dim=-1)  # (H, W)
        proj = torch.clamp(proj, 0, line_len)
        closest = start.view(1, 1, 2) + proj.unsqueeze(-1) * u.view(1, 1, 2)
        dist = torch.norm(grid - closest, dim=-1)  # (H, W)
        # Antialiased line
        mask = torch.sigmoid((width / 2 - dist) * 10)
        return mask

    def forward(self, symbols):
        """
        Render list of symbolic elements to image.
        Args:
            symbols: list of list of dict with keys:
                - label: str
                - bbox: [x, y, w, h]
                - text: optional str
                - type: 'rectangle', 'arrow', 'checkbox'
        Returns:
            (B, C, H, W) rendered image
        """
        if isinstance(symbols, dict):
            symbols = [symbols]
        if not isinstance(symbols, list):
            raise TypeError("symbols must be list of dict or list of list of dict")
        
        if len(symbols) == 0:
            return torch.ones(1, self.num_channels, self.image_size, self.image_size, device=self.grid.device)
        
        if isinstance(symbols[0], dict):
            symbols = [symbols]  # Wrap in batch
            
        device = self.grid.device
        batch_size = len(symbols)
        canvas = torch.ones(batch_size, self.num_channels, self.image_size, self.image_size, device=device)

        for b, sym_list in enumerate(symbols):
            if isinstance(sym_list, dict):
                sym_list = [sym_list]  # Handle single dict as list
            elif not isinstance(sym_list, list):
                raise TypeError("each batch item must be dict or list of dict")
            
            for obj in sym_list:
                if "bbox" not in obj:
                    continue
                
                # Ensure bbox is list
                bbox = obj["bbox"]
                if len(bbox) != 4:
                    continue
                x, y, w, h = map(float, bbox)
                center = torch.tensor([(x + w/2), (y + h/2)], device=device)
                
                if obj.get("type") == "rectangle" or "label" in obj:
                    # Draw rectangle
                    corners = [
                        (x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)
                    ]
                    for i in range(4):
                        start = torch.tensor(corners[i], device=device)
                        end = torch.tensor(corners[i+1], device=device)
                        line_mask = self._draw_line(start, end, width=3.0, color=0)
                        canvas[b, :, :, :] *= (1 - line_mask).unsqueeze(0)

                if obj.get("text"):
                    # Simple rasterization of text as rectangle
                    tx, ty, tw, th = x, y+h+5, len(obj["text"]) * 10, 20
                    text_mask = torch.zeros(self.image_size, self.image_size, device=device)
                    if tx >= 0 and ty >= 0 and tx+tw < self.image_size and ty+th < self.image_size:
                        text_mask[int(ty):int(ty+th), int(tx):int(tx+tw)] = 1
                    canvas[b, :, :, :] *= (1 - text_mask).unsqueeze(0)

                if obj.get("type") == "arrow":
                    # Arrow from center upward
                    head_len = min(w, h) * 0.4
                    shaft = torch.tensor([center[0], center[1]+h/2-10], device=device)
                    head = torch.tensor([center[0], center[1]+h/2-10-head_len], device=device)
                    
                    # Shaft
                    shaft_mask = self._draw_line(shaft, head, width=2.5, color=0)
                    canvas[b, :, :, :] *= (1 - shaft_mask).unsqueeze(0)
                    
                    # Head (triangle)
                    left = torch.tensor([center[0]-5, center[1]+h/2-10-head_len+8], device=device)
                    right = torch.tensor([center[0]+5, center[1]+h/2-10-head_len+8], device=device)
                    for p in [left, head, right, left]:
                        line_mask = self._draw_line(p, head, width=2.5, color=0)
                        canvas[b, :, :, :] *= (1 - line_mask).unsqueeze(0)

                if obj.get("type") == "checkbox":
                    # Box
                    cx, cy, cw, ch = x, y, w, h
                    corners = [(cx,cy), (cx+cw,cy), (cx+cw,cy+ch), (cx,cy+ch), (cx,cy)]
                    for i in range(4):
                        start = torch.tensor(corners[i], device=device)
                        end = torch.tensor(corners[i+1], device=device)
                        line_mask = self._draw_line(start, end, width=2.0, color=0)
                        canvas[b, :, :, :] *= (1 - line_mask).unsqueeze(0)
                    # Check mark
                    if w > 10 and h > 10:
                        check_start = torch.tensor([cx+5, cy+ch/2], device=device)
                        check_turn = torch.tensor([cx+cw/2, cy+ch-5], device=device)
                        check_end = torch.tensor([cx+cw-5, cy+5], device=device)
                        for a, b in [(check_start, check_turn), (check_turn, check_end)]:
                            line_mask = self._draw_line(a, b, width=3.0, color=0)
                            canvas[b, :, :, :] *= (1 - line_mask).unsqueeze(0)

        return canvas

# -------------------
# 2. TextRenderer
# -------------------

class TextRenderer(nn.Module):
    """
    Neural font synthesizer using StyleGAN2-ADA conditioned on handwriting style.
    This is a simplified surrogate; in practice would use pre-trained GAN.
    """
    def __init__(self, style_dim=256, out_channels=3, max_text_len=10):
        super().__init__()
        self.style_dim = style_dim
        self.out_channels = out_channels
        self.max_text_len = max_text_len
        self.char_height = 64
        self.char_width = 51  # will be padded to 512
        # Simple CNN decoder from style + text embedding
        self.char_embedding = nn.Embedding(256, 64)  # ASCII
        self.projection = nn.Linear(style_dim + 64, 128*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, text: str, style_emb: torch.Tensor) -> torch.Tensor:
        device = style_emb.device
        if text is None:
            text = ""
        # Tile style_emb to match text length
        char_tensors = []
        # For each character, run through decoder
        for i, c in enumerate(text):
            if c == "\u2192":
                c = "-"
            if c == "\u2713":
                c = "v"
            idx = max(32, min(ord(c), 255))  # clamp to [32,255]
            char_emb = self.char_embedding(torch.tensor(idx, device=device))
            # Combine with style
            x = torch.cat([style_emb[0], char_emb], dim=0)  # (D+64,)
            x = self.projection(x)  # (128*4*4)
            x = x.view(1, 128, 4, 4)  # (1, 128, 4, 4)
            img = self.decoder(x)  # (1, C, 32, 32)
            img = F.interpolate(img, size=(self.char_height, self.char_width), mode='bilinear', align_corners=False)
            char_tensors.append(img)
        
        if not char_tensors:
            # Return empty image
            return torch.zeros(1, self.out_channels, self.char_height, 512, device=device)
        
        # Concatenate horizontally
        row = torch.cat(char_tensors, dim=3)  # (1, C, H, W_total)
        # Pad to 512
        total_width = row.shape[3]
        if total_width < 512:
            pad = torch.zeros(1, self.out_channels, self.char_height, 512 - total_width, device=device)
            row = torch.cat([row, pad], dim=3)
        elif total_width > 512:
            row = row[:, :, :, :512]
        
        # Ensure final shape
        assert row.shape[-1] == 512, f"TextRenderer output width must be 512, got {row.shape[-1]}"
        return row  # (1, C, 64, 512)

# ------------------------
# 3. VisualReprogrammer
# ------------------------

class VisualReprogrammer(nn.Module):
    """
    Maps symbolic inputs to synthetic visual prompts with learned style transfer.
    """
    def __init__(self, image_size=512):
        super().__init__()
        self.rasterizer = SketchRasterizer(image_size=image_size)

    def forward(self, symbol_dict: dict) -> torch.Tensor:
        # Wrap single dict in list of lists: [[dict]]
        if isinstance(symbol_dict, dict):
            symbol_dict = [[symbol_dict]]
        # If list of dict, wrap in batch: [[dict1, dict2, ...]]
        elif isinstance(symbol_dict, list) and isinstance(symbol_dict[0], dict):
            symbol_dict = [symbol_dict]
        return self.rasterizer(symbol_dict)

# ---------------------------
# 4. VisualPromptEncoder
# ---------------------------

class VisualPromptEncoder(nn.Module):
    """
    CNN-Transformer hybrid with ResNet-50 stem and coordinate-aware attention.
    """
    def __init__(self, out_features=256, image_size=512):
        super().__init__()
        self.out_features = out_features
        self.image_size = image_size
        # Use a lighter ResNet stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Simplified: skip full ResNet, use basic blocks
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        
        # Coordinate-aware transformer blocks
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, 64, out_features))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=8, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Project stem features to latent dim
        self.proj = nn.Conv2d(256, out_features, 1)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.stem(x)  # (B, 64, H/4, W/4)
        x = self.layer1(x)  # (B, 64, ...)
        x = self.layer2(x)  # (B, 128, H/8, W/8)
        x = self.layer3(x)  # (B, 256, H/16, W/16)
        
        x = self.proj(x)  # (B, D, H/16, W/16)
        
        # Reshape to (B, L, D)
        D = x.shape[1]
        L = x.shape[2] * x.shape[3]
        x = x.permute(0, 2, 3, 1).reshape(B, L, D)  # (B, L, D)
        
        # Add coordinate embeddings
        pos = F.interpolate(
            self.pos_embedding.permute(0,3,1,2), 
            size=x.shape[1],
            mode='nearest'
        )
        pos = pos.permute(0,2,3,1).reshape(1, -1, D)
        pos = pos[:, :x.shape[1], :]
        x = x + pos
        
        # Transformer
        x = self.transformer(x)  # (B, L, D)
        
        # Reshape back
        H_, W_ = H//16, W//16
        x = x.reshape(B, H_, W_, D)  # (B, H/16, W/16, D)
        
        # Upsample to full size
        x = x.permute(0,3,1,2)  # (B, D, H/16, W/16)
        x = F.interpolate(x, size=(H, W), mode='bilinear')  # (B, D, H, W)
        x = x.permute(0,2,3,1)  # (B, H, W, D)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# -------------------------------
# 5. FlowConditioningInjector
# -------------------------------

# Already defined

class FlowConditioningInjector(nn.Module):
    """Spatially-gated cross-modulation layer for U-Net conditioning."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.gamma_proj = nn.Conv2d(256, num_channels, 1)
        self.beta_proj = nn.Conv2d(256, num_channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, H, W, 256) -> reshape to (B, 256, H, W)
        cond = cond.permute(0, 3, 1, 2)
        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)
        return x * (1 + gamma) + beta

# ------------------------
# 6. VectorVelocityUNet
# ------------------------

class VectorVelocityUNet(nn.Module):
    """U-Net with time-embedded Fourier features and velocity heads."""
    def __init__(self, in_channels=3, out_channels=3, base_channels=128, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.base_channels = base_channels

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels),  # Match to in_channels
        )

        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        for i in range(num_scales):
            out_ch = base_channels * (2**i)
            self.encoder.append(UNetBlock(in_ch, out_ch, time_emb_dim=in_channels))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = UNetBlock(in_ch, in_ch, time_emb_dim=in_channels)

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_scales):
            out_ch = base_channels * (2**(num_scales-i-1))
            self.decoder.append(UNetBlock(in_ch + out_ch, out_ch, time_emb_dim=in_channels, upsample=True))
            in_ch = out_ch

        # Final conv
        self.final = nn.Conv2d(in_ch, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        t_emb = self.time_emb(t)  # (B, in_channels)
        t_emb = t_emb.view(t_emb.shape[0], -1, 1, 1)  # (B, C, 1, 1)

        # Downsample
        skips = []
        for block in self.encoder:
            x = block(x, t_emb)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x, t_emb)

        # Upsample
        for i, block in enumerate(self.decoder):
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = block(x, t_emb)

        # Final velocity prediction
        return self.final(x)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, upsample=False):
        super().__init__()
        self.upsample = upsample
        mid_channels = out_channels if upsample else in_channels
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ReLU()

        # Skip connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.activation(self.norm1(self.conv1(x)))
        
        if t_emb is not None:
            h = h + t_emb
        
        h = self.activation(self.norm2(self.conv2(h)))
        
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            
        residual = self.shortcut(x)
        if self.upsample:
            residual = F.interpolate(residual, scale_factor=2, mode='nearest')
            
        return h + residual

# ------------------------
# 7. ProbabilityPathTracer
# ------------------------

# Already defined in api.py but moved here

class ProbabilityPathTracer(nn.Module):
    """Constructs linear interpolation paths between clean and noisy images."""
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t = (1 - t) * x0 + t * x1
        return (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
