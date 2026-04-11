import torch
import torch.nn as nn

class UnifiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vpe = None  # VisualPromptEncoder
        self.cmvm = None  # CrossModalVisualMapper
        self.visual_codec = None  # VisualCodec
        self.fmg = None  # FlowMatchingGenerator

    def forward(self, x):
        # x: input image with sketch + text
        visual_prompt = self.vpe(x)
        latents = self.visual_codec.encode(visual_prompt)
        # FMG takes [noisy_latent, prompt_latent] → predicts velocity
        # Final output generated via integration in latent space
        return self.visual_codec.decode(latents)

    @classmethod
    def from_pretrained(cls, path: str):
        model = cls()
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        return model