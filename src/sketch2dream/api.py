import torch
import numpy as np
from typing import Dict, Any
import pytorch_lightning as pl

# Import modules
from .modules import (
    VisualPromptEncoder,
    VectorVelocityUNet,
    FlowConditioningInjector,
)
from .losses import FlowMatchingLoss
from .datasets import RealSketchLoader, DualPathAugmentor
class Sketch2ImageGenerator(pl.LightningModule):
    """
    PyTorch Lightning module for training and inference.
    """
    def __init__(
        self,
        lr: float = 1e-4,
        image_size: int = 512,
        batch_size: int = 16
    ):
        super().__init__()
        self.lr = lr
        self.image_size = image_size
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.encoder = VisualPromptEncoder(out_features=256, image_size=image_size)
        self.unet = VectorVelocityUNet(in_channels=3, out_channels=3, base_channels=128)
        self.injector = FlowConditioningInjector(num_channels=128)
        self.loss_fn = FlowMatchingLoss()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        return self.unet(x, t, cond)

    def training_step(self, batch, batch_idx):
        x0 = batch  # clean sketches or real images
        cond = self.encoder(x0)  # (B, H, W, 256)
        loss = self.loss_fn(self.unet, x0, cond)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def on_validation_start(self):
        if self.current_epoch % 10 == 0:
            self._run_inference_sample()
            
    def _run_inference_sample(self):
        # Synthetic sample for demo
        from .modules import SketchRasterizer
        rasterizer = SketchRasterizer()
        symbols = [[
            {"type": "rectangle", "bbox": [100, 50, 200, 150], "text": "tree"},
            {"type": "arrow", "bbox": [220, 100, 20, 60], "text": "\u2192 clouds"},
            {"type": "checkbox", "bbox": [20, 20, 40, 40], "text": "\u2713 remove grass"}
        ]]
        cond_image = rasterizer(symbols).detach().cpu()
        # Encode
        cond = self.encoder(cond_image.to(self.device))
        # Sample
        x_noisy = torch.randn(1, 3, 512, 512, device=self.device)
        t_steps = 50
        x_t = x_noisy
        for i in range(t_steps, 0, -1):
            t = torch.ones(1, device=self.device) * (i / t_steps)
            v_t = self.unet(x_t, t, cond)
            x_t = x_t - (1 / t_steps) * v_t
        
        # Log image
        if self.logger:
            self.logger.experiment.add_images("val/gen", (x_t.clamp(0,1).cpu()), self.current_epoch)

def train(config: Dict[str, Any]):
    """
    Train the sketch2image model.
    
    Args:
        config: Dictionary with keys:
            - batch_size: int
            - lr: float
            - max_epochs: int
            - data_dir: str
            - accelerator: str ("gpu" or "cpu")
    """
    model = Sketch2ImageGenerator(
        lr=config.get("lr", 1e-4),
        batch_size=config["batch_size"]
    )
    
    dataset = RealSketchLoader(data_dir=config["data_dir"], image_size=512)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(42)
    )
    
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 100),
        accelerator=config.get("accelerator", "gpu"),
        devices=1,
        deterministic=True,
        log_every_n_steps=10
    )
    
    trainer.fit(model, dataloader)

def predict(image: np.ndarray) -> np.ndarray:
    """
    Run inference on a single image.
    
    Args:
        image: Input image as (H, W, 3) numpy array, uint8 [0,255] or float32 [0,1]
    
    Returns:
        Generated image as (H, W, 3) numpy array, float32 [0,1]
    """
    # Assume CPU
    device = "cpu"
    
    # Preprocess
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Ensure (H, W, C) and in [0,1]
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] not in [1, 3]:
        image = image.transpose(1, 2, 0)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    # Clip
    image = np.clip(image, 0, 1)
    
    # Add batch dim
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1, 3, H, W)
    
    # Create model
    model = Sketch2ImageGenerator()
    model = model.to(device)
    model.eval()
    
    # Encode
    with torch.no_grad():
        cond = model.encoder(x)

        # Sampling
        x_noisy = torch.randn(1, 3, 512, 512, device=device)
        t_steps = 50
        x_t = x_noisy
        for i in range(t_steps, 0, -1):
            t = torch.ones(1, device=device) * (i / t_steps)
            v_t = model.unet(x_t, t, cond)
            x_t = x_t - (1 / t_steps) * v_t

        # Convert back
        out = x_t[0].permute(1, 2, 0).cpu().numpy()
        return np.clip(out, 0, 1)
