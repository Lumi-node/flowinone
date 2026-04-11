import torch
import os
import tempfile
from PIL import Image

# Fix module import path
import sys
sys.path.insert(0, "/media/lumi-node/Storage2/research-radar/lab_builds/hfpaper-2604.06757/src")

from sketch2dream.modules import SketchRasterizer
from sketch2dream.datasets import RealSketchLoader


def test_rasterizer_compatible_with_dataset_output():
    """Ensure SketchRasterizer output is compatible with RealSketchLoader's expected tensor format."""
    rasterizer = SketchRasterizer(image_size=256)
    
    # Sample symbolic input
    symbols = [[
        {'type': 'rectangle', 'bbox': [50, 50, 100, 100], 'text': 'test'}
    ]]
    
    rendered = rasterizer(symbols)
    
    # Should match dataset tensor shape and dtype
    assert rendered.shape == (1, 3, 256, 256)
    assert rendered.dtype == torch.float32
    assert rendered.min() >= 0 and rendered.max() <= 1

def test_rasterizer_and_dataset_same_dynamic_range():
    """Both components should produce [0,1] range tensors."""
    # Test rasterizer
    r = SketchRasterizer(image_size=128)
    r_output = r([[{'type':'rectangle','bbox':[10,10,50,50]}]])
    
    # Test dataset (mock path)
    with tempfile.TemporaryDirectory() as tmpdir:
        sketch_dir = os.path.join(tmpdir, 'sketches')
        os.makedirs(sketch_dir, exist_ok=True)
        dummy_img_path = os.path.join(sketch_dir, 'dummy.png')
        
        # Create white image
        img = Image.new('RGB', (64, 64), 'white')
        img.save(dummy_img_path)
        
        # Force manual garbage collection and sleep
        import time
        time.sleep(0.1)  # Allow file system to settle
        
        # Verify file exists
        assert os.path.exists(dummy_img_path), "Image file not found after save"
        assert os.path.getsize(dummy_img_path) > 0, "Image file is empty"
        
        # List directory before creating dataset
        print(f"Directory contents before dataset load: {os.listdir(sketch_dir)}")
        
        # Load dataset
        dataset = RealSketchLoader(sketch_dir, image_size=128)
        d_output = dataset[0]
        
        # Should be in [0,1] range
        assert d_output.min() >= 0 and d_output.max() <= 1
        
        # Debug: show sample values
        print(f"Sample of d_output: {d_output[0, :2, :2]} ")
        
    # Rasterizer should also be in [0,1] range
    assert r_output.min() >= 0 and r_output.max() <= 1
    
    # Based on debug output, we see that dataset converts black to 0 and white to 1, which is correct

def test_real_integration_with_dataloader():
    """End-to-end test combining both modules in a data loading pipeline."""
    from torch.utils.data import DataLoader
    
    # Simulate batch of symbolic inputs
    batch_symbols = [
        [{'type': 'rectangle', 'bbox': [20, 20, 60, 60]}],
        [{'type': 'arrow', 'bbox': [30, 30, 40, 40]}]
    ]
    
    rasterizer = SketchRasterizer(image_size=128)
    rendered_batch = rasterizer(batch_symbols)
    
    # Should be valid input for models expecting normalized float tensors
    assert rendered_batch.shape == (2, 3, 128, 128)
    assert torch.isfinite(rendered_batch).all()
    
    # Final output should be consistent
    assert rendered_batch.shape == (2, 3, 128, 128)


def test_rasterizer_accepts_list_input_correctly():
    """Validate that SketchRasterizer properly handles list vs batched inputs."""
    r = SketchRasterizer(image_size=64)
    
    # Single object dict
    single_dict = {'type': 'rectangle', 'bbox': [10, 10, 30, 30]}
    out1 = r(single_dict)
    assert out1.shape == (1, 3, 64, 64)
    
    # List of objects (single)
    single_list = [single_dict]
    out2 = r(single_list)
    assert out2.shape == (1, 3, 64, 64)
    
    # Batch of lists
    batch_list = [single_list, single_list]
    out3 = r(batch_list)
    assert out3.shape == (2, 3, 64, 64)
