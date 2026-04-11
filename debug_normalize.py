import sys
sys.path.insert(0, 'src')
from sketch2dream.datasets import RealSketchLoader
import torch
import tempfile
import os
from PIL import Image

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        sketch_dir = os.path.join(tmpdir, 'sketches')
        os.makedirs(sketch_dir)
        img_path = os.path.join(sketch_dir, 'white.png')
        Image.new('RGB', (64, 64), 'white').save(img_path)
        
        ds = RealSketchLoader(sketch_dir, image_size=64)
        x = ds[0]
        print('x.min() =', x.min().item(), 'x.max() =', x.max().item())

if __name__ == "__main__":
    main()