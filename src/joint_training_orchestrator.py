import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import clip


class JointTrainingOrchestrator:
    def __init__(self, model, lr=1e-4, weight_decay=0.01):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device="cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.eval()
        self.lpips = None  # Will use lpips package
        try:
            import lpips
            self.lpips = lpips.LPIPS(net='vgg')
        except ImportError:
            pass

        # Loss weights
        self.lambda_v = 1.0
        self.lambda_clip = 0.5
        self.lambda_lpips = 0.8

    def compute_clip_sim(self, img1, img2):
        return torch.cosine_similarity(
            self.clip_model.encode_image(img1),
            self.clip_model.encode_image(img2),
            dim=1
        ).mean()

    def training_step(self, batch):
        input_image = batch["input_image"].to("cuda")
        target_image = batch["target_image"].to("cuda")  # Assumed in batch

        # Forward through full system
        visual_prompt = self.model.vpe(input_image)  # Includes text rendering
        prompt_latent = self.model.visual_codec.encode(visual_prompt)
        
        # Simulate noisy latent
        noisy_latent = prompt_latent + torch.randn_like(prompt_latent)
        
        # Concatenate in channel dimension
        fmg_input = torch.cat([noisy_latent, prompt_latent], dim=1)
        predicted_velocity = self.model.fmg(fmg_input)
        
        # Target velocity
        target_velocity = target_latent - noisy_latent
        
        loss_velocity = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)

        # Reconstruct output image
        final_latent = noisy_latent + predicted_velocity
        output_image = self.model.visual_codec.decode(final_latent)

        # Perceptual losses
        if self.lpips:
            loss_lpips = self.lpips(output_image, target_image).mean()
        else:
            loss_lpips = torch.tensor(0.0)

        sim_clip = self.compute_clip_sim(output_image, target_image)
        loss_clip = 1 - sim_clip

        total_loss = (
            self.lambda_v * loss_velocity +
            self.lambda_clip * loss_clip +
            self.lambda_lpips * loss_lpips
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "velocity_loss": loss_velocity.item(),
            "clip_loss": loss_clip.item(),
            "lpips_loss": loss_lpips.item()
        }