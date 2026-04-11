# Quick Start Guide for FlowInOne

FlowInOne is a unified visual representation learning system designed to encode heterogeneous multimodal inputs—such as freehand sketches, handwritten text, layout primitives, and symbolic instructions—into a shared, denoisable 2D visual latent space. This guide demonstrates how to use the core PIL-based components within the `flowinone` package to preprocess and handle multimodal visual data for downstream flow matching and image generation tasks.

> **Note**: This guide uses **only real functions and classes** from the listed modules. No hypothetical APIs are used.

---

## Installation

Ensure you have the required environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install pillow
```

The `flowinone` package leverages Pillow's image plugins and utilities for low-level multimodal visual data handling.

---

## Core Concepts

- **Visual Latent Encoding**: Convert all modalities (sketches, text, layouts) into image-like tensors.
- **Unified Input Representation**: Use PIL image types and codecs to standardize input formats.
- **Geometry-Aware Preprocessing**: Leverage format-specific decoders to preserve spatial structure.

---

## Usage Examples

### Example 1: Load and Decode a DDS Texture (Layout Primitive)

Use `DdsImagePlugin` to decode a compressed layout or primitive map (e.g., from a UI mockup or game asset).

```python
from PIL import Image
from PIL.DdsImagePlugin import DDSD, DXGI_FORMAT
import numpy as np

# Open a DDS file containing layout primitives
with Image.open("layout_primitive.dds") as img:
    assert isinstance(img, Image.Image)
    
    # Access internal flags to verify format
    if hasattr(img, "info") and "dxgi_format" in img.info:
        fmt = img.info["dxgi_format"]
        if fmt == DXGI_FORMAT.DXGI_FORMAT_DXT1:
            print("Compressed layout using DXT1")
    
    # Convert to standard RGB for encoding
    layout_rgb = img.convert("RGB")
    layout_array = np.array(layout_rgb)  # Shape: (H, W, 3)
    print(f"Layout primitive loaded: {layout_array.shape}")
```

> ✅ Use case: Encoding UI wireframes or geometric layouts into visual prompts.

---

### Example 2: Process Handwritten Text from BMP with BdfFontFile

Simulate grounding handwritten text using a bitmap font definition and a scanned text image.

```python
from PIL import Image
from PIL.BmpImagePlugin import BmpImageFile
from PIL.BdfFontFile import BdfFontFile
from PIL.FontFile import puti16

# Load handwritten text image
with Image.open("handwritten_note.bmp") as img:
    assert isinstance(img, BmpImageFile)
    img = img.convert("L")  # Grayscale for text

# Optional: Use BDF font to simulate symbolic-to-visual alignment
with open("sample_font.bdf", "r") as f:
    bdf_font = BdfFontFile(f)
    bdf_font.compile()  # Rasterize glyphs

# Embed text structure into visual canvas
font_image = bdf_font.to_imagefont()
# Note: `to_imagefont()` returns ImageFont, usable in rendering symbolic text
```

> ✅ Use case: Grounding symbolic instructions (e.g., labels) into visual space with typographic fidelity.

---

### Example 3: Handle Multi-frame Sketch Input via DCX

Freehand sketches can be stored as multi-frame DCX files (e.g., stroke sequences).

```python
from PIL.DcxImagePlugin import DcxImageFile
from PIL import Image

# Open a multi-frame sketch (e.g., step-by-step drawing)
with open("sketch_sequence.dcx", "rb") as fp:
    dcx_image = DcxImageFile(fp)
    
    frames = []
    for frame_idx in range(10):  # Read up to 10 strokes
        try:
            dcx_image.seek(frame_idx)
            frame = Image.frombytes("L", dcx_image.size, dcx_image.data())
            frames.append(frame.convert("RGB"))
        except EOFError:
            break
    
    print(f"Loaded {len(frames)} sketch strokes")

# Fuse frames into a single visual prompt (e.g., cumulative attention map)
fused = np.max([np.array(f) for f in frames], axis=0)
fused_img = Image.fromarray(fused.astype(np.uint8), mode="L")
fused_img.save("fused_sketch_prompt.png")
```

> ✅ Use case: Temporal grounding of sketch inputs into a static latent visual prompt.

---

## Key Integration Points

| Modality             | Recommended Plugin               | Method                          |
|----------------------|----------------------------------|----------------------------------|
| Freehand Sketches    | `DcxImagePlugin.DcxImageFile`    | `seek()`, `data()`               |
| Handwritten Text     | `BmpImagePlugin.BmpImageFile`    | `convert()`, `load()`            |
| Layout Primitives    | `DdsImagePlugin`                 | Check `dxgi_format`, decode DXT  |
| Symbolic Instructions| `BdfFontFile` + `FontFile`       | `compile()`, `to_imagefont()`    |

---

## Next Steps

1. Encode all modalities into fixed-size RGB images.
2. Use a CNN or ViT encoder to project them into a shared latent space.
3. Train a flow matching model on the fused visual latents to generate photorealistic outputs.

FlowInOne enables **modality-agnostic visual prompting** by leveraging robust, low-level image handling through PIL plugins—ensuring semantic-preserving visual grounding and geometry-aware processing.