import torch
import pytest
from unittest.mock import Mock

# Import the loss class
import sys; sys.path.insert(0, '../../src'); from sketch2dream.losses import FlowMatchingLoss

def test_flow_matching_loss_output_shape_and_nonnegativity():
    # Mock a UNet that returns zero velocity
    def mock_unet_call(x, t, condition=None):
        # Should receive xt, t, condition
        assert x.shape == (2, 3, 64, 64)
        assert t.shape == (2,)
        assert condition is None
        return torch.zeros_like(x)

    # Create model mock with __call__ overridden
    mock_unet = Mock()
    mock_unet.side_effect = lambda x, t, c: mock_unet_call(x, t, c)

    # Instantiate loss
    loss_fn = FlowMatchingLoss()

    # Create dummy inputs matching the acceptance criterion
    x0 = torch.randn(2, 3, 64, 64)
    x1 = torch.randn(2, 64, 64, 256)  # Note: unusual channel dim, but we trust the spec

    # Compute loss
    loss = loss_fn(mock_unet, x0, x1)

    # Check output shape: single scalar
    assert loss.numel() == 1, f"Expected scalar loss, got shape {loss.shape}"
    
    # Check non-negativity
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    # Verify mock was called once
    assert mock_unet.call_count == 1


def test_flow_matching_loss_shape_compatibilities():
    # Test internal handling of x1 shape: (B, H, W, C) -> (B, C, H, W)
    loss_fn = FlowMatchingLoss()
    def mock_call(x, t, c): return torch.zeros_like(x)
    mock_unet = Mock(side_effect=mock_call)

    x0 = torch.randn(1, 3, 32, 32)
    x1 = torch.randn(1, 48, 48, 256)  # (B, H, W, C)

    loss = loss_fn(mock_unet, x0, x1)

    # Should handle transpose and spatial resize
    assert loss.numel() == 1
    assert loss.item() >= 0


def test_flow_matching_loss_channel_reduction():
    loss_fn = FlowMatchingLoss()
    def mock_call(x, t, c): return torch.zeros_like(x)
    mock_unet = Mock(side_effect=mock_call)

    x0 = torch.randn(1, 3, 64, 64)
    x1 = torch.randn(1, 64, 64, 256)  # (B, H, W, C=256)

    loss = loss_fn(mock_unet, x0, x1)

    # Should permute and slice channels
    assert loss.numel() == 1
    assert loss.item() >= 0


def test_flow_matching_loss_edge_cases():
    loss_fn = FlowMatchingLoss()

    # Test empty tensor
    with pytest.raises(ValueError):
        x0_empty = torch.randn(0, 3, 64, 64)
        x1_empty = torch.randn(0, 64, 64, 256)
        loss_fn(None, x0_empty, x1_empty)

    # Test None inputs
    with pytest.raises(AttributeError):
        loss_fn(None, None, torch.randn(2, 64, 64, 256))

    with pytest.raises(AttributeError):
        loss_fn(None, torch.randn(2, 3, 64, 64), None)

    # Test device mismatch
    if torch.cuda.is_available():
        x0 = torch.randn(2, 3, 64, 64).cuda()
        x1 = torch.randn(2, 64, 64, 256).cpu()
        x1 = x1.permute(0, 3, 1, 2)[:, :3]
        x1 = torch.nn.functional.interpolate(x1, size=(64, 64))

        with pytest.raises(RuntimeError):
            mock_unet = Mock(return_value=torch.zeros(2, 3, 64, 64).cuda())
            loss_fn(mock_unet, x0, x1)  # Should fail due to device mismatch


def test_acceptance_criterion_direct():
    """
    Reproduce the exact acceptance criterion command.
    """
    try:
        # This should run without error
        cmd = "python -c \"from sketch2dream.losses import FlowMatchingLoss; import torch; l=FlowMatchingLoss(); u=type('MockUNet', (), {'__call__': lambda s,x,t,c: torch.zeros_like(x)})(); loss=l(u, torch.randn(2,3,64,64), torch.randn(2,64,64,256)); assert loss.numel() == 1 and loss.item() >= 0\""
        result = Bash(command=f"cd /media/lumi-node/Storage2/research-radar/lab_builds/hfpaper-2604.06757/.worktrees/issue-6337c161-09-flow-matching-loss && PYTHONPATH=src {cmd}")
        assert result.exit_code == 0, f"Acceptance criterion command failed: {result.output}"        
    except Exception as e:
        pytest.fail(f"Acceptance criterion test failed: {str(e)}")