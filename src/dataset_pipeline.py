import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import os
import random


class SketchTextDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=256):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))]
        self.handwriting_fonts = [
            'Arial', 'Courier New'  # Simulate handwriting
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Simulate input: sketch + text overlay
        sketch_path = os.path.join(self.data_dir, self.files[idx])
        sketch = Image.open(sketch_path).convert('RGB').resize((self.img_size, self.img_size))
        draw = ImageDraw.Draw(sketch)
        
        # Simulate user text instruction
        prompt = random.choice([
            "make it cyberpunk",
            "add sunglasses",
            "steampunk style",
            "neon lights"
        ])
        
        font = ImageFont.load_default()
        try:
            font = ImageFont.truetype(random.choice(self.handwriting_fonts), 16)
        except:
            pass

        # Bottom-right corner
        bbox = [0.7, 0.7, 1.0, 1.0]
        x = int(bbox[0] * self.img_size)
        y = int(bbox[1] * self.img_size)
        draw.text((x, y), prompt, fill="white", font=font)
        
        # Apply augmentations
        if self.split == 'train':
            if random.random() < 0.5:
                # Color jitter
                from torchvision import transforms
                jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                sketch = jitter(sketch)
        
        return {
            "input_image": torch.from_numpy(np.array(sketch)).permute(2, 0, 1).float() / 255.0,
            "text_prompt": prompt
        }

# Example usage:
# dataset = SketchTextDataset("/path/to/sketchy_coco")
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)