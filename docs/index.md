---
title: FlowInOne
description: Unified image-to-image generation via multimodal flow matching.
---

# FlowInOne
::: { .text-center }
**Unified image-to-image generation via multimodal flow matching**  
Seamlessly fuse sketches, text, layouts, and symbols into photorealistic images with a single flow model.
:::

<br>

:::: {.grid .grid-3}
::: {.card}
### 🖌️ Multimodal Visual Encoding
Encode freehand sketches, handwritten text, layout primitives, and symbolic instructions into a shared 2D visual latent space—preserving semantics and spatial structure without modality-specific decoders.
:::

::: {.card}
### 🔄 Geometry-Aware Flow Matching
Leverage geometry-preserving flow dynamics to generate high-fidelity images with accurate spatial alignment and structural coherence from fused visual prompts.
:::

::: {.card}
### 🧩 Unified Latent Space
Train a single denoisable latent space that supports diverse input modalities, eliminating the need for alignment losses or separate conditioning pathways.
:::

::: {.card}
### 🚀 End-to-End Generation
Generate photorealistic target images directly from multimodal visual prompts—no cascaded models, no post-processing, no compromise on quality.
:::
::::

<br>

::: {.grid .grid-2}
```bash
pip install flowinone
```
:::
::: { .text-center }
[Get Started →](getting-started.md){ .md-button .md-button--primary }
:::