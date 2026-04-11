import os
import torch
import tempfile
import shutil
from PIL import Image

# Use fixed seed for deterministic tests
TEST_SEED = 42
torch.manual_seed(TEST_SEED)

from sketch2dream.datasets import RealSketchLoader

class TestRealSketchLoader:
    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.image_size = 512
        
        # Create mock sketch images
        for i in range(3):
            img = Image.new('RGB', (100 + i*50, 100 + i*50), color=(73, 109, 137))
            img_path = os.path.join(self.test_dir, f'sketch_{i}.png')
            img.save(img_path)

    def teardown_method(self):
        """Tear down test fixtures after each test method."""
        shutil.rmtree(self.test_dir)

    def test_loader_initialization(self):
        """Test that RealSketchLoader initializes correctly."""
        dataset = RealSketchLoader(self.test_dir, image_size=self.image_size)
        assert len(dataset) == 3
        assert isinstance(dataset, torch.utils.data.Dataset)

    def test_image_loading_and_shape(self):
        """Test that images are loaded and transformed with correct shape and dtype."""
        dataset = RealSketchLoader(self.test_dir, image_size=self.image_size, seed=TEST_SEED)
        
        for i in range(len(dataset)):
            img = dataset[i]
            # Check shape
            assert img.shape == (3, self.image_size, self.image_size)
            # Check dtype
            assert img.dtype == torch.float32
            # Check value range
            assert img.min() >= 0.0
            assert img.max() <= 1.0

    def test_deterministic_augmentation(self):
        """Test that augmentation is deterministic with fixed seed."""
        dataset1 = RealSketchLoader(self.test_dir, image_size=self.image_size, seed=TEST_SEED)
        dataset2 = RealSketchLoader(self.test_dir, image_size=self.image_size, seed=TEST_SEED)
        
        # Get same index from both datasets
        img1 = dataset1[0]
        img2 = dataset2[0]
        
        # Should be identical due to seed
        assert torch.equal(img1, img2)

    def test_different_augmentation_without_seed(self):
        """Test that without seed, augmentations differ."""
        dataset1 = RealSketchLoader(self.test_dir, image_size=self.image_size, seed=123)
        dataset2 = RealSketchLoader(self.test_dir, image_size=self.image_size, seed=456)
        
        img1 = dataset1[0]
        img2 = dataset2[0]
        
        # Should be different due to different seeds
        assert not torch.equal(img1, img2)
