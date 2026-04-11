import torch
from sketch2dream.modules import VisualReprogrammer


def test_visual_reprogrammer_label_bbox():
    """Test that VisualReprogrammer handles label+bbox input correctly."""
    v = VisualReprogrammer()
    output = v({"label": "tree", "bbox": [100, 50, 200, 150]})
    assert output.shape == (1, 3, 512, 512)
    assert output.dtype == torch.float32
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_visual_reprogrammer_arrow():
    """Test that VisualReprogrammer handles arrow symbol."""
    v = VisualReprogrammer()
    output = v({"type": "arrow", "bbox": [100, 100, 400, 400]})
    assert output.shape == (1, 3, 512, 512)
    assert output.dtype == torch.float32


def test_visual_reprogrammer_checkbox():
    """Test that VisualReprogrammer handles checkbox symbol."""
    v = VisualReprogrammer()
    output = v({"type": "checkbox", "bbox": [50, 50, 150, 150], "checked": True})
    assert output.shape == (1, 3, 512, 512)
    assert output.dtype == torch.float32
