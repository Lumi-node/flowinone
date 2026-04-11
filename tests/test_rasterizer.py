import torch
import unittest
from src.differentiable_rasterizer import DifferentiableRasterizer

class TestDifferentiableRasterizer(unittest.TestCase):
    def test_output_shape(self):
        rasterizer = DifferentiableRasterizer(img_size=256)
        char_pos = torch.tensor([[[0.5, 0.5]]])  # Center
        mask = rasterizer(char_pos)
        self.assertEqual(mask.shape, (1, 256, 256))

    def test_gradient_flow(self):
        rasterizer = DifferentiableRasterizer(img_size=64)
        char_pos = torch.tensor([[[0.5, 0.5]]], requires_grad=True)
        mask = rasterizer(char_pos)
        loss = mask.sum()
        loss.backward()
        self.assertIsNotNone(char_pos.grad)

if __name__ == '__main__':
    unittest.main()