# FlowInOne: Unifying Multimodal Generation as Image-in, Image-out Flow Matching

## Research Background

### 1. Research Problem

Designing intuitive and flexible human-AI interfaces for image generation remains a central challenge in multimodal AI. Current systems often require users to provide inputs in rigid, modality-specific formats—such as text prompts, reference images, or structured layouts—limiting accessibility and expressive fidelity. While users naturally combine sketches, handwritten annotations, spatial layouts, and symbolic cues when conceptualizing visual ideas, existing generative models treat these modalities in isolation or rely on complex fusion mechanisms that require modality-specific decoders, alignment losses, or multi-stage pipelines.

This work addresses the problem of **heterogeneous multimodal grounding in image generation**: how to unify diverse, user-generated visual and symbolic inputs—such as freehand sketches, handwritten text, bounding box layouts, and icon-based instructions—into a coherent generative process that preserves both semantic intent and geometric structure. The core challenge lies in creating a **unified visual representation** that encodes disparate modalities into a shared, denoisable latent space, enabling a single generative model to produce high-fidelity, photorealistic images without modality-specific conditioning or post-hoc alignment.

Existing approaches struggle with either semantic fidelity (e.g., missing textual instructions) or geometric precision (e.g., misplacing objects from layouts), often due to disjoint encoding pathways or weak cross-modal grounding. FlowInOne proposes to solve this by treating multimodal generation as a **visual-to-visual flow transformation**: all inputs are first rendered into a 2D visual prompt, encoded into a shared latent, and then denoised via a single flow matching objective to produce the target image.

---

### 2. Related Work and Existing Approaches

Recent advances in diffusion-based image generation have enabled high-quality synthesis from text (Rombach et al., 2022), sketches (Sanghi et al., 2023), and layouts (Zhao et al., 2023). However, most systems are modality-specific or rely on late fusion strategies. For example:

- **Text-to-image models** like Stable Diffusion (Rombach et al., 2022) and Imagen (Saharia et al., 2022) use CLIP (Radford et al., 2021) text encoders but fail to interpret spatial or geometric cues.
- **Layout-to-image methods** (Zhao et al., 2023; Hong et al., 2023) condition on bounding boxes or masks but often ignore textual semantics or freeform annotations.
- **Sketch-based generation** (Sanghi et al., 2023; Liu et al., 2024) focuses on edge maps but lacks support for symbolic or linguistic inputs.

Multimodal fusion has been explored via **cross-attention mechanisms** (e.g., Flamingo, Alayrac et al., 2022) or **joint embedding spaces** (e.g., CLIP, ALIGN), but these typically require paired data and alignment losses. More recent efforts like MIM (Xie et al., 2023) and PaLI (Chen et al., 2023) scale multimodal understanding but remain focused on classification or captioning, not geometry-aware generation.

Flow matching (Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022) has emerged as a principled alternative to diffusion, modeling generation as a continuous transformation from noise to data. Recent work such as Rectified Flow (Liu et al., 2022) shows improved sample quality and efficiency. However, these models are typically unimodal and lack mechanisms for integrating heterogeneous visual inputs.

Notably, **Visual Prompt Tuning** (Bahng et al., 2022) and **Visual Representations for Control** (Zhang et al., 2023) suggest that rendering non-visual inputs (e.g., layouts) as 2D visual prompts can improve grounding. FlowInOne builds on this insight but extends it to a full multimodal, end-to-end flow-based generation framework.

---

### 3. Advancement of the Field

FlowInOne advances the state of the art in multimodal image generation through three key contributions:

1. **Unified Visual Latent Encoding**:  
   We introduce a visual encoder that renders all inputs—sketches, handwritten text, layout primitives (boxes, lines), and symbolic icons—into a single 2D visual prompt, which is then encoded into a shared latent space. This eliminates the need for modality-specific conditioning paths or alignment losses, enabling true unification at the representation level.

2. **Image-in, Image-out Flow Matching**:  
   We reformulate multimodal generation as a flow matching problem where both the input (fused visual prompt) and output (target image) exist in the same visual space. The model learns a continuous vector field that transforms a noisy version of the visual prompt into a photorealistic image, preserving spatial semantics and geometric structure throughout the flow.

3. **Semantic- and Geometry-Aware Grounding**:  
   By operating in a shared visual latent space, FlowInOne inherently aligns semantics and geometry without explicit constraints. Handwritten labels are spatially grounded to their associated regions; layout primitives guide object placement; and symbolic icons are interpreted in context—all through a single, denoisable pathway.

This approach enables a **decoder-free, alignment-free, and modality-agnostic** generation pipeline. Unlike prior fusion methods, FlowInOne does not require cross-modal attention, paired supervision per modality, or multi-stage refinement. It demonstrates that a wide range of user intents can be captured through visual rendering and transformed via flow matching, opening a path toward more intuitive, human-centered generative interfaces.

While currently a research prototype, FlowInOne establishes a proof-of-concept for **visual representation as the universal interface** for multimodal generation. Future work may explore real-time interaction, scalability to video, and integration with embodied agents.

---

### 4. References

- Alayrac, J.-B., et al. (2022). Flamingo: a visual language model for few-shot learning. *Advances in Neural Information Processing Systems*, 35, 23716–23736.  
- Albergo, M. S., & Vanden-Eijnden, E. (2022). Building normalizing flows with stochastic interpolants. *International Conference on Learning Representations (ICLR)*.  
- Bahng, H., et al. (2022). Exploring visual prompt tuning for vision transformers. *CVPR Workshop on Vision and Language*.  
- Chen, X., et al. (2023). PaLI: A jointly scaled multilingual language-image model. *arXiv preprint arXiv:2209.06794*.  
- Hong, Y., et al. (2023). LayoutDiffusion: Controllable diffusion models for layout-to-image generation. *CVPR*.  
- Lipman, Y., et al. (2022). Flow matching for generative modeling. *International Conference on Learning Representations (ICLR)*.  
- Liu, L., et al. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR*.  
- Liu, Z., et al. (2024). SketchGen: Few-shot sketch-based image generation. *SIGGRAPH Asia*.  
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.  
- Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. *CVPR*.  
- Saharia, C., et al. (2022). Photorealistic text-to-image diffusion models with deep language understanding. *NeurIPS*.  
- Sanghi, A., et al. (2023). Sketch-guided text-to-image diffusion models. *ICCV*.  
- Xie, Z., et al. (2023). MIM: Masked image modeling for self-supervised vision learning. *CVPR*.  
- Zhang, Y., et al. (2023). Visual control via neural rendering. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.  
- Zhao, L., et al. (2023). LayoutDiffuser: Diffusion-based generation from scene graphs with layout control. *ICCV*.