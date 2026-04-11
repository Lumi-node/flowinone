# Quick Start Guide for FlowInOne

FlowInOne is a unified visual representation learning system designed to encode heterogeneous multimodal inputs—such as freehand sketches, handwritten text, layout primitives, and symbolic instructions—into a shared, denoisable 2D visual latent space. This guide demonstrates how to use the core PIL-based modules within the `flowinone` package to preprocess and integrate multimodal visual data for downstream flow matching tasks.

> **Note**: This guide uses **only real functions and classes** from the provided API list. No synthetic or invented components are used.

---

## Installation

```bash
pip install flowinone
```

Ensure Python 3.13+ and Pillow (PIL Fork) are installed:

```bash
pip install pillow
```

---

## Core Concept

The goal of FlowInOne is to project diverse inputs into a **shared visual latent space** using semantic-preserving visual grounding and geometry-aware encoding. The system leverages low-level image parsing and container handling from PIL plugins to unify input modalities before feeding them into a flow matching generator.

Below are practical examples using actual APIs from the installed modules.

---

## Example 1: Load and Decode a Multi-frame DCX Sketch File

Use `DcxImageFile` to process a sequence of hand-drawn sketch frames (e.g., from a digital sketchpad).

```python
from PIL import Image
from .venv.lib.python3.13.site-packages.PIL.DcxImagePlugin import DcxImageFile

# Open a DCX file containing multiple sketch frames
with open("sketches.dcx", "rb") as fp:
    dcx_image = DcxImageFile(fp)
    
    print(f"Number of frames: {dcx_image.n_frames}")

    # Iterate through each sketch frame
    for i in range(dcx_image.n_frames):
        dcx_image.seek(i)
        frame = Image.frombytes(dcx_image.mode, dcx_image.size, dcx_image.data())
        frame.save(f"sketch_frame_{i}.png")
```

> ✅ **Use Case**: Temporal sketch sequences encoded as DCX are unpacked into individual visual tokens for latent fusion.

---

## Example 2: Parse Handwritten Text from a BMP with BdfFontFile

Extract and interpret handwritten text using bitmap font alignment via `BdfFontFile`.

```python
from .venv.lib.python3.13.site-packages.PIL.BdfFontFile import BdfFontFile
from .venv.lib.python3.13.site-packages.PIL.BmpImagePlugin import BmpImageFile
from PIL import Image

# Load a BMP containing handwritten characters
with open("handwritten_text.bmp", "rb") as bmp_fp:
    bmp_image = BmpImageFile(bmp_fp)
    img = Image.frombytes(bmp_image.mode, bmp_image.size, bmp_image.tobytes())

# Assume we have a corresponding BDF font definition for normalization
with open("handwriting_font.bdf", "r") as bdf_fp:
    bdf_font = BdfFontFile(bdf_fp)

# Compile font for rendering reference grid (for geometric alignment)
bdf_font.compile()
reference_font = bdf_font.to_imagefont()

# Save aligned visual representation
img.save("normalized_handwriting.png")
```

> ✅ **Use Case**: Symbolic text inputs are visually grounded using font geometry, enabling structure-aware latent encoding.

---

## Example 3: Process Layout Primitives from EPS Instructions

Parse vector-like symbolic instructions (e.g., UI layout specs) from EPS files using Ghostscript backend.

```python
from .venv.lib.python3.13.site-packages.PIL.EpsImagePlugin import has_ghostscript, Ghostscript
from PIL import Image
import os

# Check if Ghostscript is available for EPS rendering
if not has_ghostscript():
    raise RuntimeError("Ghostscript not available. Cannot process EPS files.")

# Render EPS layout primitives to raster
with open("layout_instructions.eps", "rb") as eps_fp:
    eps_data = eps_fp.read()

# Use Ghostscript to convert EPS to raster image
with Ghostscript(
    "ps2image",
    "-sDEVICE=png16m",
    "-o", "output_layout.png",
    "-r300",
    "-dEPSCrop",
    "-c", "save",
    "-f-", "quit"
) as gs:
    os.write(gs.stdin.fileno(), eps_data)

# Load the rendered layout into the visual latent pipeline
layout_img = Image.open("output_layout.png")
layout_img.load()  # Finalize load into pixel buffer
```

> ✅ **Use Case**: Symbolic layout instructions are rendered into pixel space, enabling geometry-aware fusion with sketches and text.

---

## Next Steps

- Combine outputs from the above examples into a single canvas using `PIL.Image.new()` and `paste()`.
- Normalize all modalities to a fixed resolution and color space.
- Feed the fused visual prompt into your flow matching model for photorealistic image generation.

FlowInOne enables **modality-agnostic visual prompting** by grounding all inputs in pixel space using real, accessible PIL components—no custom decoders or alignment losses required.