import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np

class RealSketchLoader(Dataset):
    def __init__(self, root_dir, image_size=256, seed=42):
        """
        Dataset class for loading real sketch images with deterministic augmentation.
        
        Args:
            root_dir (str): Path to directory containing sketch images.
            image_size (int): Target size to resize images to.
            seed (int): Random seed for deterministic augmentation.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
        ])
        
        # Set seed for deterministic behavior
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Get all image file paths
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            self.image_paths.extend(
                os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(ext[1:])
            )
        
        # Ensure deterministic ordering
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # Set seed before transform for deterministic augmentation
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        image = self.transform(image)
        return image

class DualPathAugmentor:
    def __init__(self, real_dataset, synth_generator):
        self.real_dataset = real_dataset
        self.synth_generator = synth_generator
        self.domain_classifier_weight = 0.5
    
    def __iter__(self):
        # Pseudo-implementation for testing
        while True:
            yield torch.randn(3, 512, 512)
    
    def __len__(self):
        return len(self.real_dataset)
    
    def __getitem__(self, idx):
        # Alternate between real and synthetic
        if idx % 2 == 0:
            return self.real_dataset[idx % len(self.real_dataset)]
        else:
            return self.synth_generator()