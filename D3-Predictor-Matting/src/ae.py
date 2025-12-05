import diffusers
import torch
from torch import nn

class AutoencoderKL(nn.Module):
    def __init__(self, scale: float = 0.18215, shift: float = 0.0, repo="stabilityai/stable-diffusion-2-1"):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.ae = diffusers.AutoencoderKL.from_pretrained(repo, subfolder="vae")
        self.ae.eval()
        self.ae.requires_grad_(False)

    def forward(self, img):
        return self.encode(img)

    @torch.no_grad()
    def encode(self, img):
        # # Ensure input dtype matches model weight dtype (handle BF16 mixed precision)
        # if hasattr(self.ae.encoder.conv_in.weight, 'dtype'):
        #     img = img.to(dtype=self.ae.encoder.conv_in.weight.dtype)
        latent = self.ae.encode(img, return_dict=False)[0].sample()
        return (latent - self.shift) * self.scale

    # @torch.no_grad()
    def decode(self, latent):
        # # Ensure input dtype matches model weight dtype (handle BF16 mixed precision)
        # if hasattr(self.ae.decoder.conv_in.weight, 'dtype'):
        #     latent = latent.to(dtype=self.ae.decoder.conv_in.weight.dtype)
        rec = self.ae.decode(latent / self.scale + self.shift, return_dict=False)[0]
        return rec