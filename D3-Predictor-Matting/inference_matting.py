from omegaconf import OmegaConf
import hydra
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import os
from tqdm import tqdm
import torch.nn.functional as F
import h5py
import matplotlib
from vae_image_processor import VaeImageProcessor
from src.ae import AutoencoderKL
import gc
import numpy as np

diffusion_image_size = 768
ae = AutoencoderKL(repo="stabilityai/stable-diffusion-2-1")
ae = ae.cuda().bfloat16()

model_config_path = "configs/sd21_d3_predictor_matting.yaml"
cfg_model = OmegaConf.load(model_config_path)
OmegaConf.resolve(cfg_model)
d3_predictor = hydra.utils.instantiate(cfg_model).model
state_dict = torch.load("../checkpoints/matting/matting_checkpoint.pth")

d3_predictor.load_state_dict(state_dict, strict=False)
del d3_predictor.visual_experts, d3_predictor.adapters
torch.cuda.empty_cache()
gc.collect()
d3_predictor = d3_predictor.cuda().bfloat16()
d3_predictor.requires_grad_(False)
d3_predictor.eval()

for root, dirs, files in os.walk('path/to/rgb/images'):
    for name in tqdm(files, desc="Processing files", unit="file"):
        file_path = os.path.join(root, name)
        img = Image.open(file_path).convert('RGB')
        img = to_tensor(img)[None].to(device='cuda', dtype=torch.bfloat16) * 2 - 1
        B, C, H, W = img.shape
        if H != diffusion_image_size or W != diffusion_image_size:
            img = F.interpolate(
                img, size=(diffusion_image_size, diffusion_image_size), mode="bilinear", align_corners=False
            )
        with torch.no_grad():
            features = d3_predictor.get_features(
                img,
                ["A human portrait matting map."] * B
            )

            img_tensor = ae.decode(features)
            if H != diffusion_image_size or W != diffusion_image_size:
                img_tensor = F.interpolate(img_tensor, (H, W), mode="bilinear", align_corners=False)

            processor = VaeImageProcessor()

            imgs = processor.postprocess(img_tensor, output_type="pil")

            img = imgs[0]

            os.makedirs(f'../inference_res/matting', exist_ok=True)
            img.save(os.path.join(f'../inference_res/matting', name.replace('jpg', 'png')))
