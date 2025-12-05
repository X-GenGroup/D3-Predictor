import torch
import numpy as np
from PIL import Image
from typing import Union, List, Optional

# Adapted from diffusers (https://github.com/huggingface/diffusers/blob/main/src/diffusers/image_processor.py#L88)
class VaeImageProcessor:
    def __init__(self, do_normalize: bool = True):
        """
        Args:
            do_normalize (bool): Whether to normalize decoded latent images from [-1,1] to [0,1].
        """
        self.do_normalize = do_normalize

    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize image tensors from [-1,1] range to [0,1].
        """
        return (images * 0.5 + 0.5).clamp(0, 1)

    def _denormalize_conditionally(self, images: torch.Tensor, do_denormalize: Optional[List[bool]] = None) -> torch.Tensor:
        """
        Conditionally normalize each image to [0,1] based on the do_denormalize list.
        """
        if do_denormalize is None:
            return self.denormalize(images) if self.do_normalize else images

        return torch.stack([
            self.denormalize(images[i]) if do_denormalize[i] else images[i]
            for i in range(images.shape[0])
        ])

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert torch.Tensor [B,C,H,W] to numpy format [B,H,W,C].
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
        """
        Convert numpy image array (B,H,W,C) or (H,W,C) to PIL.Image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            return [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            return [Image.fromarray(image) for image in images]

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None
    ) -> Union[List[Image.Image], np.ndarray, torch.Tensor]:
        """
        Convert decoded image tensor to PIL, numpy, tensor, or latent.

        Args:
            image: torch.Tensor with shape [B,C,H,W].
            output_type: Output type, supports "latent", "pt", "np", "pil".
            do_denormalize: Whether to normalize each image from [-1,1] to [0,1].

        Returns:
            Processed image.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Input should be a torch.Tensor, got {type(image)}.")

        if output_type not in ["latent", "pt", "np", "pil"]:
            raise ValueError(f"Unsupported output_type: {output_type}. Expected one of ['latent', 'pt', 'np', 'pil'].")

        if output_type == "latent":
            return image

        image = self._denormalize_conditionally(image, do_denormalize)

        if output_type == "pt":
            return image

        image = self.pt_to_numpy(image)

        if output_type == "np":
            return image

        if output_type == "pil":
            return self.numpy_to_pil(image)