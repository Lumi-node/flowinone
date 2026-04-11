import sys
sys.path.insert(0, 'src')
from sketch2dream.datasets import RealSketchLoader
import os
import tempfile
import torch
from PIL import Image

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        sketch_dir = os.path.join(tmpdir, 'sketches')
        os.makedirs(sketch_dir)
        img_path = os.path.join(sketch_dir, 'test.png')
        
        # Create and save a white image
        Image.new('RGB', (64, 64), 'white').save(img_path)
        
        # Verify it was saved
        print(f"Files in {sketch_dir}: {os.listdir(sketch_dir)}")
        
        # Load dataset
        dataset = RealSketchLoader(sketch_dir, image_size=64)
        x = dataset[0]
        print(f"Dataset output shape: {x.shape}, min: {x.min().item()}, max: {x.max().item()}")

if __name__ == "__main__":
    main()