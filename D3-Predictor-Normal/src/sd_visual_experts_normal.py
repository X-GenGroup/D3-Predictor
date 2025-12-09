import torch
from torch import nn
import torch.nn.functional as F
import einops
from diffusers import DiffusionPipeline
from jaxtyping import Float, Int
from pydoc import locate
from typing import Literal
import gc
from .layers import FeedForwardBlock, FourierFeatures, Linear, MappingNetwork
from .min_sd21 import SD21UNetModel

class SD21UNetFeatureExtractor(SD21UNetModel):
    def __init__(self):
        super().__init__()

    def forward(self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs):
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8, s9] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s10, s11] = self.down_blocks[3](
            sample,
            temb=emb,
        )

        # 4. mid
        sample_mid = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        _, [us1, us2, us3] = self.up_blocks[0](
            hidden_states=sample_mid,
            temb=emb,
            res_hidden_states_tuple=[s9, s10, s11],
        )

        _, [us4, us5, us6] = self.up_blocks[1](
            hidden_states=us3,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us7, us8, us9] = self.up_blocks[2](
            hidden_states=us6,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us10, us11, us12] = self.up_blocks[3](
            hidden_states=us9,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
            encoder_hidden_states=encoder_hidden_states,
        )

        # 6. post-process
        sample = self.conv_norm_out(us12)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return {
            "mid": sample_mid,
            "us1": us1,
            "us2": us2,
            "us3": us3,
            "us4": us4,
            "us5": us5,
            "us6": us6,
            "us7": us7,
            "us8": us8,
            "us9": us9,
            "us10": us10,
            "sample": sample,
        }

class FeedForwardBlockCustom(FeedForwardBlock):
    def __init__(self, d_model: int, d_ff: int, d_cond_norm: int = None, norm_type: Literal['AdaRMS', 'FiLM'] = 'AdaRMS', use_gating: bool = True):
        super().__init__(d_model=d_model, d_ff=d_ff, d_cond_norm=d_cond_norm)
        if not use_gating:
            self.up_proj = LinearSwish(d_model, d_ff, bias=False)
        if norm_type == 'FiLM':
            self.norm = FiLMNorm(d_model, d_cond_norm)

class FFNStack(nn.Module):
    def __init__(self, dim: int, depth: int, ffn_expansion: float, dim_cond: int, 
                 norm_type: Literal['AdaRMS', 'FiLM'] = 'AdaRMS', use_gating: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FeedForwardBlockCustom(d_model=dim, d_ff=int(dim * ffn_expansion), d_cond_norm=dim_cond, norm_type=norm_type, use_gating=use_gating) 
             for _ in range(depth)])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond_norm=cond)
        return x

class FiLMNorm(nn.Module):
    def __init__(self, features, cond_features):
        super().__init__()
        self.linear = Linear(cond_features, features * 2, bias=False)
        self.feature_dim = features

    def forward(self, x, cond):
        B, _, D = x.shape
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        # broadcast scale and shift across all features
        scale = scale.view(B, 1, D)
        shift = shift.view(B, 1, D) 
        return scale * x + shift

class LinearSwish(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.silu(super().forward(x))
    

class ArgSequential(nn.Module):  # Utility class to enable instantiating nn.Sequential instances with Hydra
    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

class StableFeatureAligner(nn.Module):
    def __init__(
        self,
        ae: nn.Module,
        mapping,
        adapter_layer_class: str,
        feature_dims: dict[str, int],
        visual_expert_cls: str,
        adapter_layer_params: dict = {},
        use_text_condition: bool = False,
        t_min: int = 1,
        t_max: int = 999,
        t_max_model: int = 999,
        num_t_stratification_bins: int = 3,
        alignment_loss: Literal["cossim"] = "cossim",
        train_unet: bool = True,
        train_adapter: bool = True,
        t_init: int = 261,
        learn_timestep: bool = False,
        val_dataset: torch.utils.data.Dataset | None = None,
        val_t: int = 261,
        val_feature_key: str = "us6",
        val_chunk_size: int = 10,
        use_adapters: bool = True
    ):
        super().__init__()
        self.ae = ae
        self.val_t = val_t
        self.val_feature_key = val_feature_key
        self.val_dataset = val_dataset
        self.val_chunk_size = val_chunk_size
        self.use_adapters = use_adapters

        self.repo = "stabilityai/stable-diffusion-2-1"

        self.device = None

        self.mapping = None
        if use_adapters:
            self.time_emb = FourierFeatures(1, mapping.width)
            self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
            self.mapping = MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout)

        if use_adapters:
            self.adapters = nn.ModuleDict()
            for k, dim in feature_dims.items():
                self.adapters[k] = locate(adapter_layer_class)(dim=dim, **adapter_layer_params)
                self.adapters[k].requires_grad_(train_adapter)

        self.visual_experts = locate(visual_expert_cls)()
        self.pipe = DiffusionPipeline.from_pretrained(
            self.repo,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.visual_experts.load_state_dict(self.pipe.unet.state_dict())
        self.visual_experts.eval()
        self.visual_experts.requires_grad_(False)

        self.d3_predictor = locate(visual_expert_cls)()
        self.d3_predictor.load_state_dict(
            {k: v.detach().clone() for k, v in self.visual_experts.state_dict().items()}
        )

        if train_unet or learn_timestep:
            self.d3_predictor.train()
        else:
            self.d3_predictor.eval()
        self.d3_predictor.requires_grad_(train_unet)

        self.use_text_condition = use_text_condition
        if self.use_text_condition:
            pass
        else:
            with torch.no_grad():
                prompt_embeds_dict = self.get_prompt_embeds([""])
                self._empty_prompt_embeds = prompt_embeds_dict["prompt_embeds"]
                del self.pipe.text_encoder

        del self.pipe.unet, self.pipe.vae
        torch.cuda.empty_cache()
        gc.collect()

        self.t_min = t_min
        self.t_max = t_max
        self.t_max_model = t_max_model
        self.num_t_stratification_bins = num_t_stratification_bins
        self.alignment_loss = alignment_loss
        self.timestep = nn.Parameter(
            torch.tensor(float(t_init), requires_grad=learn_timestep), requires_grad=learn_timestep
        )

    def get_prompt_embeds(self, prompt: list[str], device) -> dict[str, torch.Tensor | None]:
        self.pipe = self.pipe.to(device)
        self.prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {"prompt_embeds": self.prompt_embeds}

    def _get_unet_conds(self, prompts: list[str], device, dtype, N_T) -> dict[str, torch.Tensor]:
        B = len(prompts)
        if self.use_text_condition:
            prompt_embeds_dict = self.get_prompt_embeds(prompts, device)
        else:
            prompt_embeds_dict = {"prompt_embeds": einops.repeat(self._empty_prompt_embeds, "b ... -> (B b) ...", B=B)}

        unet_conds = {
            "encoder_hidden_states": einops.repeat(
                prompt_embeds_dict["prompt_embeds"], "B ... -> (B N_T) ...", N_T=N_T
            ).to(dtype=dtype, device=device),
            "added_cond_kwargs": {},
        }

        return unet_conds
    
    def _ensure_device_sync(self, device):
        """Ensure all components are on the same device"""
        if self.device != device:
            self.device = device
            # Move all components to correct device
            self.pipe = self.pipe.to(device)
            self.visual_experts = self.visual_experts.to(device)
            self.d3_predictor = self.d3_predictor.to(device)
            if hasattr(self, 'mapping') and self.mapping is not None:
                self.mapping = self.mapping.to(device)
            if hasattr(self, 'adapters'):
                for adapter in self.adapters.values():
                    adapter = adapter.to(device)

    def forward(
        self, x_condition: Float[torch.Tensor, "b c h w"], x_image: Float[torch.Tensor, "b c h w"], x_normal_ori: Float[torch.Tensor, "b c h w"], caption: list[str], **kwargs
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO: Temporarily written this way to ensure sample dtype matches model weight dtype. Need to refactor this more elegantly later.
        x_condition = x_condition.to(torch.bfloat16)
        x_image = x_image.to(torch.bfloat16)
        x_normal_ori = x_normal_ori.to(torch.bfloat16)
        B, *_ = x_condition.shape
        device = x_condition.device
        self._ensure_device_sync(device)
        t_range = self.t_max - self.t_min
        t_range_per_bin = t_range / self.num_t_stratification_bins
        t: Int[torch.Tensor, "B N_T"] = (
            self.t_min
            + torch.rand((B, self.num_t_stratification_bins), device=device) * t_range_per_bin
            + torch.arange(0, self.num_t_stratification_bins, device=device)[None, :] * t_range_per_bin
        ).long()
        # Fix timestep to maximum 999, N_T=1
        # t: Int[torch.Tensor, "B N_T"] = torch.full((B, 1), 999, device=device, dtype=torch.long)
        B, N_T = t.shape

        with torch.no_grad():
            unet_conds = self._get_unet_conds(caption, device, x_condition.dtype, N_T)
            # print(f"x_condition.shape={x_condition.shape}, x_image.shape={x_image.shape}")
            x_condition_0: Float[torch.Tensor, "(B N_T) ..."] = self.ae.encode(x_condition)
            x_condition_0 = einops.repeat(x_condition_0, "B ... -> (B N_T) ...", N_T=N_T)
            x_image_0: Float[torch.Tensor, "(B N_T) ..."] = self.ae.encode(x_image)
            x_image_0 = einops.repeat(x_image_0, "B ... -> (B N_T) ...", N_T=N_T)
            # print(f"x_condition_0.shape={x_condition_0.shape}, x_image_0.shape={x_image_0.shape}")
            _, *latent_shape = x_image_0.shape
            noise_sample = torch.randn((B * N_T, *latent_shape), device=device, dtype=x_image.dtype)

            x_condition_t: Float[torch.Tensor, "(B N_T) ..."] = self.pipe.scheduler.add_noise(
                x_condition_0,
                noise_sample,
                einops.rearrange(t, "B N_T -> (B N_T)"),
            )

            feats_visual_experts_raw = self.visual_experts(
                x_condition_t,
                einops.rearrange(t, "B N_T -> (B N_T)"),
                **unet_conds,
            )
            sample_visual_experts = feats_visual_experts_raw["sample"]  # shape: (B*N_T, D, H, W)
            feats_visual_experts = {
                k: einops.rearrange(v, "(B N_T) D H W -> B N_T (H W) D", B=B, N_T=N_T)
                for k, v in feats_visual_experts_raw.items() if k != "sample"
            }

        feats_d3_predictor_raw = self.d3_predictor(
            x_condition_0,
            einops.rearrange(torch.ones_like(t) * self.timestep, "B N_T -> (B N_T)"),
            **unet_conds,
        )
        sample_d3_predictor = feats_d3_predictor_raw["sample"]  # shape: (B*N_T, D, H, W)
        feats_d3_predictor = {
            k: einops.rearrange(v, "(B N_T) D H W -> B N_T (H W) D", N_T=N_T)
            for k, v in feats_d3_predictor_raw.items() if k != "sample"
        }

        if self.use_adapters:
            # time conditioning for adapters
            if not self.mapping is None:
                map_cond: Float[torch.Tensor, "(B N_T) ..."] = self.mapping(
                    self.time_in_proj(
                        self.time_emb(
                            einops.rearrange(t, "B N_T -> (B N_T) 1").to(dtype=x_condition.dtype, device=device) / self.t_max_model
                        )
                    )
                )
   
            feats_d3_predictor: dict[str, Float[torch.Tensor, "B N_T ..."]] = {
                k: einops.rearrange(
                    self.adapters[k](einops.rearrange(v, "B N_T ... -> (B N_T) ..."), cond=map_cond),
                    "(B N_T) ... -> B N_T ...",
                    B=B,
                    N_T=N_T,
                )
                for k, v in feats_d3_predictor.items()
            }

        if self.alignment_loss == "cossim":
            losses = {
                f"neg_cossim_{k}": -F.cosine_similarity(feats_d3_predictor[k], v.detach(), dim=-1).mean()
                for k, v in feats_visual_experts.items()
            }
            decoded_sample = self.ae.decode(sample_d3_predictor)  # shape: (B*N_T, C, H, W), range [-1,1]
            x_image_expanded = einops.repeat(x_image, "B C H W -> (B N_T) C H W", N_T=N_T).contiguous()  # shape: (B*N_T, C, H, W), range [-1,1]
            _, _, orig_h, orig_w = x_normal_ori.shape
            decoded_1ch_orig_size = F.interpolate(decoded_sample, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            x_image_1ch_orig_size = F.interpolate(x_image_expanded, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            # Create valid mask based on dataset type and depth value range
            # Note: x_depth_unnorm shape is (B, 1, 368, 1232), need to expand to (B*N_T, 1, 368, 1232)
            x_normal_ori_expanded = einops.repeat(x_normal_ori, "B C H W -> (B N_T) C H W", N_T=N_T)

            eps = 1e-4

            # Create mask: check if all three channels of x_normal_ori_expanded are not all zero
            # For each pixel position, check if L2 norm of three channels is greater than 0
            normal_mask = torch.norm(x_normal_ori_expanded, dim=1, keepdim=True) > eps  # shape: (B*N_T, 1, H, W)
            normal_mask = normal_mask.squeeze(1)  # shape: (B*N_T, H, W)

            # Calculate mse_sample loss (only on valid pixels)
            if normal_mask.sum() > 0:  # Ensure there are valid pixels
                # Expand mask to all channel dimensions
                mask_expanded = normal_mask.unsqueeze(1).expand_as(decoded_1ch_orig_size)  # shape: (B*N_T, C, H, W)
                mse_loss = F.mse_loss(decoded_1ch_orig_size, x_image_1ch_orig_size.detach(), reduction='none')
                losses["mse_sample"] = mse_loss[mask_expanded].mean()
            else:  # If no valid pixels, return 0 loss
                print("No valid pixels, returning 0 loss")
                losses["mse_sample"] = torch.tensor(0.0, device=decoded_1ch_orig_size.device, dtype=decoded_1ch_orig_size.dtype)            

            # Calculate angular loss
            loss_ang = torch.cosine_similarity(decoded_1ch_orig_size.to(torch.float64), x_normal_ori_expanded.to(torch.float64), dim=1)
            loss_ang = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()
            loss_ang = loss_ang.to(decoded_1ch_orig_size.dtype)

            # Calculate loss only on valid pixels
            if normal_mask.sum() > 0:  # Ensure there are valid pixels
                losses["angular_sample"] = loss_ang[normal_mask].mean()
            else:  # If no valid pixels, return 0 loss
                print("No valid pixels, returning 0 loss")
                losses["angular_sample"] = torch.tensor(0.0, device=loss_ang.device, dtype=loss_ang.dtype)
            
            return losses
        else:
            raise ValueError(f"Invalid alignment loss type: {self.alignment_loss}")

    @torch.no_grad()
    def get_features(
        self,
        x: Float[torch.Tensor, "b c h w"],
        caption: list[str] | None,
    ) -> Float[torch.Tensor, "b d h' w'"]:
        (B, *_), device = x.shape, x.device

        if caption is None:
            caption = [""] * B

        unet_conds = self._get_unet_conds(caption, device, x.dtype, 1)
        x_0 = self.ae.encode(x)

        feats_d3_predictor = self.d3_predictor(
                x_0,
                torch.ones((B,), device=device, dtype=self.timestep.dtype) * self.timestep,
                **unet_conds,
            )
        
        feats_d3_predictor = feats_d3_predictor["sample"]  # shape: (B, D, H, W)

        return feats_d3_predictor


