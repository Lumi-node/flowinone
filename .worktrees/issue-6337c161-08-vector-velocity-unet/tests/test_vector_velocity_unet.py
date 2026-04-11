import pytest
import torch
from src.sketch2dream.modules import VectorVelocityUNet

def test_vector_velocity_unet_forward():
    """Test forward pass with expected shapes."""
    model = VectorVelocityUNet()
    x = torch.randn(1, 3, 512, 512)
    t = torch.rand(1)
    c = torch.randn(1, 512, 512, 256)
    
    v = model(x, t, c)
    
    assert v.shape == (1, 3, 512, 512), f"Expected (1,3,512,512), got {v.shape}"

def test_vector_velocity_unet_get_features():
    """Test _get_features returns correct number of feature maps."""
    model = VectorVelocityUNet()
    x = torch.randn(1, 3, 512, 512)
    t = torch.rand(1)
    
    features = model._get_features(x, t)
    
    assert len(features) == 4, f"Expected 4 feature maps, got {len(features)}"
    expected_channels = [64, 128, 256, 512]
    h, w = 512, 512
    for i, (feat, exp_ch) in enumerate(zip(features, expected_channels)):
        h, w = h // (2**i), w // (2**i)
        assert feat.shape == (1, exp_ch, h, w), f"Feature {i} has shape {feat.shape}, expected (1, {exp_ch}, {h}, {w})"

def test_time_embedding():
    """Test time embedding produces correct shape."""
    model = VectorVelocityUNet()
    t = torch.rand(1)
    t_emb = model.time_embed(t)
    assert t_emb.shape == (1, 64), f"Time embedding shape mismatch: {t_emb.shape}"