import torch
import torch.nn as nn

# ------------------------
# 9. FlowMatchingLoss
# ------------------------

class FlowMatchingLoss(nn.Module):
    """Per-pixel L2 loss over the probability path integral."""
    def __init__(self, num_steps=1000):
        super().__init__()
        self.num_steps = num_steps
        self.tracer = ProbabilityPathTracer()

    def forward(self, model, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss.
        Args:
            model: callable v_t = model(x_t, t, cond)
            x0: clean data (B, C, H, W)
            cond: encoded condition (B, H, W, D)
        Returns:
            scalar loss
        """
        B = x0.shape[0]
        # Random noise
        x1 = torch.randn_like(x0)
        
        # Random time steps
        t = torch.rand(B, device=x0.device)  # (B,)
        
        # Get interpolated points
        x_t = self.tracer(x0, x1, t)  # (B, C, H, W)
        
        # Optimal velocity is constant: v* = (x1 - x0)
        s_t = x1 - x0  # (B, C, H, W)
        
        # Predicted velocity
        v_t = model(x_t, t, cond)  # (B, C, H, W)
        
        # L2 loss
        loss = ((v_t - s_t) ** 2).mean()
        
        return loss

# Ensure ProbabilityPathTracer is available
from .modules import ProbabilityPathTracer
