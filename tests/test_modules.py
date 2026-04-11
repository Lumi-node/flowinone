from torch import nn
import torch

def test_skeleton_modules_smoke():
    # Just import and instantiate placeholders to ensure files exist
    from sketch2dream.modules import SketchRasterizer, TextRenderer, VisualReprogrammer, VisualPromptEncoder, FlowConditioningInjector, VectorVelocityUNet
    assert isinstance(SketchRasterizer(), nn.Module)
    assert isinstance(TextRenderer(), nn.Module)
    assert isinstance(VisualReprogrammer(), nn.Module)
    assert isinstance(VisualPromptEncoder(), nn.Module)
    assert isinstance(FlowConditioningInjector(128), nn.Module)
    assert isinstance(VectorVelocityUNet(), nn.Module)
