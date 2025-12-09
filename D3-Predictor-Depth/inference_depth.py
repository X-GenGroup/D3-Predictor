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

# Reference: Marigold's depth visualization code (https://github.com/prs-eth/Marigold)
def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    else:
        raise TypeError("img should be np.ndarray or torch.Tensor")
    return hwc

diffusion_image_size = 768
ae = AutoencoderKL(repo="stabilityai/stable-diffusion-2-1")
ae = ae.cuda().bfloat16()


model_config_path = "configs/sd21_d3_predictor_depth.yaml"
cfg_model = OmegaConf.load(model_config_path)
OmegaConf.resolve(cfg_model)
d3_predictor = hydra.utils.instantiate(cfg_model).model
state_dict = torch.load("../checkpoints/depth/depth_checkpoint.pth")

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
                ["A grayscale depth estimation image, where darker areas represent closer depths and lighter areas indicate farther depths."] * B
            )

            img_tensor = ae.decode(features)
            if H != diffusion_image_size or W != diffusion_image_size:
                img_tensor = F.interpolate(img_tensor, (H, W), mode="bilinear", align_corners=False)

            processor = VaeImageProcessor()

            np_imgs = processor.postprocess(img_tensor, output_type="np")

            data = np_imgs[0]

            data = np.mean(data, axis=-1)

            if data.max() <= data.min():
                print(f"Skipping: All valid pixels have same value ({data.min():.3f})")

            image = data

            valid_mask = (image > 1e-5) & (image < 80.0)

            if np.any(valid_mask):
                temp_image = image.copy()
                temp_image[~valid_mask] = 0
                
                _min, _max = np.quantile(
                    temp_image[valid_mask],
                    q=[0.02, 0.98]
                )
                
                if _max <= _min:
                    print(f"Skipping: All valid pixels have same value ({_min:.3f})")
                
                image = (image - _min) / (_max - _min)
                image = np.clip(image, 0, 1)
                
                image[~valid_mask] = 1.0
            else:
                print("No valid pixels")

            depth_colored = colorize_depth_maps(
                image, 0, 1, cmap="Spectral"
            )  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)

            # Remove batch dimension if present
            if depth_colored.ndim == 4:
                depth_colored = depth_colored.squeeze(0)  # Remove batch dimension

            # # Set invalid depth regions to white
            # depth_colored[:, ~valid_mask] = 255  # Set to white [255, 255, 255]

            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
            
            os.makedirs('../inference_res/depth', exist_ok=True)

            depth_colored_img.save(os.path.join('../inference_res/depth', name.replace('jpg', 'png')))
