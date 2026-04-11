import argparse
import torch
import numpy as np
from PIL import Image
from sketch2dream.modules import SketchRasterizer, VisualPromptEncoder, VectorVelocityUNet, ProbabilityPathTracer
from sketch2dream.datasets import RealSketchLoader


def generate(input_sketch: str, output_image: str, ckpt_path: str = "last.ckpt", image_size: int = 512):
    """
    CLI tool for end-to-end inference.
    """
    # Load input sketch
    img = Image.open(input_sketch).convert("RGB")
    img = img.resize((image_size, image_size))
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # Initialize modules (in practice, load from checkpoint)
    encoder = VisualPromptEncoder()
    unet = VectorVelocityUNet()
    tracer = ProbabilityPathTracer()

    # Encode visual prompt
    with torch.no_grad():
        cond = encoder(x)  # (1, H, W, 256)

    # Sampling: use probability path tracing and flow matching
    x_noisy = torch.randn_like(x)
    t_steps = 100
    x_t = x_noisy
    for i in range(t_steps, 0, -1):
        t = torch.ones(1, device=x_t.device) * (i / t_steps)
        with torch.no_grad():
            v_t = unet(x_t, t, cond)
            x_t = x_t - (1 / t_steps) * v_t

    # Save result
    x_out = x_t[0].permute(1, 2, 0).cpu().numpy()
    x_out = (np.clip(x_out, 0, 1) * 255).astype(np.uint8)
    out_img = Image.fromarray(x_out)
    out_img.save(output_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_sketch", type=str, required=True)
    parser.add_argument("--output_image", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="last.ckpt")
    args = parser.parse_args()
    generate(args.input_sketch, args.output_image, args.ckpt)
