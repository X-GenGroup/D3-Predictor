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

        self.repo = "/data/xcl/cleandift/model/sd21/stabilityai/stable-diffusion-2-1"

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

        self.unet_feature_extractor_base = locate(visual_expert_cls)()
        self.pipe = DiffusionPipeline.from_pretrained(
            self.repo,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.unet_feature_extractor_base.load_state_dict(self.pipe.unet.state_dict())
        self.unet_feature_extractor_base.eval()
        self.unet_feature_extractor_base.requires_grad_(False)

        self.unet_feature_extractor_cleandift = locate(visual_expert_cls)()
        self.unet_feature_extractor_cleandift.load_state_dict(
            {k: v.detach().clone() for k, v in self.unet_feature_extractor_base.state_dict().items()}
        )

        if train_unet or learn_timestep:
            self.unet_feature_extractor_cleandift.train()
        else:
            self.unet_feature_extractor_cleandift.eval()
        self.unet_feature_extractor_cleandift.requires_grad_(train_unet)

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
            # Move all components to the correct device
            self.pipe = self.pipe.to(device)
            self.unet_feature_extractor_base = self.unet_feature_extractor_base.to(device)
            self.unet_feature_extractor_cleandift = self.unet_feature_extractor_cleandift.to(device)
            if hasattr(self, 'mapping') and self.mapping is not None:
                self.mapping = self.mapping.to(device)
            if hasattr(self, 'adapters'):
                for adapter in self.adapters.values():
                    adapter = adapter.to(device)

    def forward(
        self, x_condition: Float[torch.Tensor, "b c h w"], x_image: Float[torch.Tensor, "b c h w"], x_depth_unnorm: Float[torch.Tensor, "b c h w"], caption: list[str], dataset_type: list[str], **kwargs
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO: Temporarily written this way to ensure sample data type matches model weight data type. Need to refactor this more elegantly later.
        x_condition = x_condition.to(torch.bfloat16)
        x_image = x_image.to(torch.bfloat16)
        x_depth_unnorm = x_depth_unnorm.to(torch.bfloat16)
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
        B, N_T = t.shape

        with torch.no_grad():
            unet_conds = self._get_unet_conds(caption, device, x_condition.dtype, N_T)
            x_condition_0: Float[torch.Tensor, "(B N_T) ..."] = self.ae.encode(x_condition)
            x_condition_0 = einops.repeat(x_condition_0, "B ... -> (B N_T) ...", N_T=N_T)
            x_image_0: Float[torch.Tensor, "(B N_T) ..."] = self.ae.encode(x_image)
            x_image_0 = einops.repeat(x_image_0, "B ... -> (B N_T) ...", N_T=N_T)
            _, *latent_shape = x_image_0.shape
            noise_sample = torch.randn((B * N_T, *latent_shape), device=device, dtype=x_image.dtype)

            x_condition_t: Float[torch.Tensor, "(B N_T) ..."] = self.pipe.scheduler.add_noise(
                x_condition_0,
                noise_sample,
                einops.rearrange(t, "B N_T -> (B N_T)"),
            )

            feats_base_raw = self.unet_feature_extractor_base(
                x_condition_t,
                einops.rearrange(t, "B N_T -> (B N_T)"),
                **unet_conds,
            )
            sample_base = feats_base_raw["sample"]  # shape: (B*N_T, D, H, W)
            feats_base = {
                k: einops.rearrange(v, "(B N_T) D H W -> B N_T (H W) D", B=B, N_T=N_T)
                for k, v in feats_base_raw.items() if k != "sample"
            }

        feats_cleandift_raw = self.unet_feature_extractor_cleandift(
            x_condition_0,
            einops.rearrange(torch.ones_like(t) * self.timestep, "B N_T -> (B N_T)"),
            **unet_conds,
        )
        sample_cleandift = feats_cleandift_raw["sample"]  # shape: (B*N_T, D, H, W)
        feats_cleandift = {
            k: einops.rearrange(v, "(B N_T) D H W -> B N_T (H W) D", N_T=N_T)
            for k, v in feats_cleandift_raw.items() if k != "sample"
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
   
            feats_cleandift: dict[str, Float[torch.Tensor, "B N_T ..."]] = {
                k: einops.rearrange(
                    self.adapters[k](einops.rearrange(v, "B N_T ... -> (B N_T) ..."), cond=map_cond),
                    "(B N_T) ... -> B N_T ...",
                    B=B,
                    N_T=N_T,
                )
                for k, v in feats_cleandift.items()
            }

        if self.alignment_loss == "cossim":
            losses = {
                f"neg_cossim_{k}": -F.cosine_similarity(feats_cleandift[k], v.detach(), dim=-1).mean()
                for k, v in feats_base.items()
            }
            decoded_sample = self.ae.decode(sample_cleandift)  # shape: (B*N_T, C, H, W), range [-1,1]
            x_image_expanded = einops.repeat(x_image, "B C H W -> (B N_T) C H W", N_T=N_T).contiguous()  # shape: (B*N_T, C, H, W), range [-1,1]
            decoded_1ch = decoded_sample.mean(dim=1, keepdim=True)  # shape: (B*N_T, 1, H, W)
            x_image_1ch = x_image_expanded.mean(dim=1, keepdim=True).detach()  # shape: (B*N_T, 1, H, W)
            # Get original depth image dimensions
            _, _, orig_h, orig_w = x_depth_unnorm.shape
            # Resize decoded and original images to original size
            decoded_1ch_orig_size = F.interpolate(decoded_1ch, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            x_image_1ch_orig_size = F.interpolate(x_image_1ch, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            # Create valid mask based on dataset type and depth value range
            # Note: x_depth_unnorm shape is (B, 1, 368, 1232), need to expand to (B*N_T, 1, 368, 1232)
            x_depth_unnorm_expanded = einops.repeat(x_depth_unnorm, "B C H W -> (B N_T) C H W", N_T=N_T)
            # Since all samples in the same batch come from the same dataset type, use the first sample's type directly
            dataset_type_single = dataset_type[0]
            # Create valid mask based on dataset type
            if dataset_type_single == "vkitti":
                # VKITTI dataset: depth values in (1e-5, 80.0) range are valid
                valid_mask = (x_depth_unnorm_expanded > 1e-5) & (x_depth_unnorm_expanded < 80.0)
            elif dataset_type_single == "hypersim":
                # Hypersim dataset: depth values in (1e-5, 65.0) range are valid
                valid_mask = (x_depth_unnorm_expanded > 1e-5) & (x_depth_unnorm_expanded < 65.0)
            elif dataset_type_single == "coco":
                # COCO dataset: depth values in (1e-5, 80.0) range are valid
                valid_mask = (x_depth_unnorm_expanded > 1e-5) & (x_depth_unnorm_expanded < 80.0)
            else:
                # Unknown dataset type, use default mask (all valid)
                valid_mask = torch.ones_like(x_depth_unnorm_expanded, dtype=torch.bool)
            
            # Calculate MSE loss using valid mask
            if valid_mask.any():
                # Only calculate loss in valid regions
                decoded_valid = decoded_1ch_orig_size[valid_mask]
                x_image_valid = x_image_1ch_orig_size[valid_mask]
                losses["mse_sample"] = F.mse_loss(decoded_valid, x_image_valid)
            else:
                # If no valid regions, fall back to version without valid mask
                # losses["mse_sample"] = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
                losses["mse_sample"] = F.mse_loss(decoded_1ch_orig_size, x_image_1ch_orig_size)
            
            # Calculate SSI loss: L1 loss after least squares alignment
            if valid_mask.any():
                # Reprocess for SSI loss: VAE decode → VaeImageProcessor postprocess → resize → channel average
                from vae_image_processor import VaeImageProcessor
                processor = VaeImageProcessor()
                
                # Directly postprocess VAE decoded results, convert range from [-1, 1] to [0, 1]
                decoded_sample_processed = processor.postprocess(decoded_sample, output_type="pt")  # (B*N_T, 3, H, W), range [0, 1]
                
                # Resize to original dimensions
                decoded_sample_orig_size = F.interpolate(decoded_sample_processed, size=(orig_h, orig_w), mode='bilinear', align_corners=False)  # (B*N_T, 3, H, W), range [0, 1]
                
                # Average across channel dimension to single channel
                decoded_1ch_orig_size_processed = decoded_sample_orig_size.mean(dim=1, keepdim=True)  # (B*N_T, 1, H, W), range [0, 1]
                
                # Calculate SSI loss for each sample
                ssi_losses = []
                for i in range(B):
                    # Get current sample's valid mask
                    sample_mask = valid_mask[i*N_T:(i+1)*N_T].view(N_T, orig_h, orig_w)  # (N_T, H, W)
                    # Get current sample's predicted and ground truth depth (using postprocessed predicted values)
                    sample_pred = decoded_1ch_orig_size_processed[i*N_T:(i+1)*N_T].view(N_T, orig_h, orig_w)  # (N_T, H, W)
                    sample_gt = x_depth_unnorm_expanded[i*N_T:(i+1)*N_T].view(N_T, orig_h, orig_w)  # (N_T, H, W)
                    
                    # Calculate SSI loss for each timestep
                    for t in range(N_T):
                        pred_t = sample_pred[t]  # (H, W)
                        gt_t = sample_gt[t]      # (H, W)
                        mask_t = sample_mask[t]  # (H, W)
                        
                        if not mask_t.any():
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.warning(f"SSI loss calculation failed - sample {i}, timestep {t}: no valid region")
                            ssi_losses.append(torch.tensor(0.0, device=device, dtype=x_image.dtype))
                            continue

                        try:
                            # Convert to float32 for accumulation to avoid numerical instability
                            pred_f = pred_t.float()
                            gt_f   = gt_t.float()
                            mask_f = mask_t.float()

                            # Closed-form solution components
                            a_00 = torch.sum(mask_f * pred_f * pred_f)
                            a_01 = torch.sum(mask_f * pred_f)
                            a_11 = torch.sum(mask_f)
                            b_0  = torch.sum(mask_f * pred_f * gt_f)
                            b_1  = torch.sum(mask_f * gt_f)

                            det = a_00 * a_11 - a_01 * a_01
                            eps = 1e-6

                            if det > eps:
                                s = (a_11 * b_0 - a_01 * b_1) / (det + eps)
                                b_val = (-a_01 * b_0 + a_00 * b_1) / (det + eps)

                                # Restore dtype
                                s = s.to(pred_t.dtype)
                                b_val = b_val.to(pred_t.dtype)

                                # Align predicted values and calculate L1 loss
                                pred_aligned = s * pred_t + b_val
                                ssi_loss = F.l1_loss(pred_aligned[mask_t], gt_t[mask_t])
                                ssi_losses.append(ssi_loss)
                            else:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning(f"SSI loss calculation failed - sample {i}, timestep {t}: matrix degenerate (det≈0)")
                                ssi_losses.append(torch.tensor(0.0, device=device, dtype=x_image.dtype))

                        except Exception as e:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.warning(f"SSI loss calculation failed - sample {i}, timestep {t}: {str(e)}")
                            ssi_losses.append(torch.tensor(0.0, device=device, dtype=x_image.dtype))
                
                # Calculate average SSI loss across all samples
                if ssi_losses:
                    losses["ssi_sample"] = torch.stack(ssi_losses).mean()
                else:
                    # If no valid SSI loss, log warning and set to 0
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("SSI loss calculation failed: no valid SSI loss values")
                    losses["ssi_sample"] = torch.tensor(0.0, device=device, dtype=x_image.dtype)
            else:
                # If no valid regions, log warning and set SSI loss to 0
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("SSI loss calculation failed: entire batch has no valid regions")
                losses["ssi_sample"] = torch.tensor(0.0, device=device, dtype=x_image.dtype)
            
            # Calculate gradient loss (Gradient Loss)
            if valid_mask.sum() > 0:
                # Use existing decoded_1ch_orig_size_processed as pred_depth for gradient loss
                pred_depth = decoded_1ch_orig_size_processed  # shape: (B*N_T, 1, orig_h, orig_w), range [0,1]
                
                # Use x_image_expanded as GT, convert from [-1,1] to [0,1]
                # Complete in one step: average channels + resize + convert range + log transform
                if x_image_expanded.shape[-2:] != (orig_h, orig_w):
                    x_image_resized = F.interpolate(
                        x_image_expanded, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                    )
                else:
                    x_image_resized = x_image_expanded
                
                # Convert to log space
                pred_log = torch.log1p(pred_depth)  # pred_depth in [0,1]
                # GT processing: average channels + range conversion [-1,1]→[0,1] + clamp + log transform
                gt_log = torch.log1p(torch.clamp(x_image_resized.mean(dim=1, keepdim=True).detach() * 0.5 + 0.5, 0, 1)) 
                
                # Calculate log difference
                log_d_diff = pred_log - gt_log  # shape: (B*N_T, 1, orig_h, orig_w)
                
                # Calculate gradient (using step size 2 to reduce noise impact)
                step = 2
                
                # Check if there are enough spatial dimensions for step size 2 calculation
                _, _, h, w = log_d_diff.shape
                if h > step and w > step:
                    # Memory optimization: directly calculate masked gradient, avoid storing intermediate tensors
                    h_mask = valid_mask[:, :, :, :-step] & valid_mask[:, :, :, step:]  # shape: (B*N_T, 1, orig_h, orig_w-2)
                    v_mask = valid_mask[:, :, :-step, :] & valid_mask[:, :, step:, :]  # shape: (B*N_T, 1, orig_h-2, orig_w)
                    
                    # Calculate masked gradient sum (logically equivalent but reduces intermediate tensors)
                    h_gradient_sum = (torch.abs(log_d_diff[:, :, :, :-step] - log_d_diff[:, :, :, step:]) * h_mask.float().detach()).sum()
                    v_gradient_sum = (torch.abs(log_d_diff[:, :, :-step, :] - log_d_diff[:, :, step:, :]) * v_mask.float().detach()).sum()
                    
                    # Calculate number of valid pixels
                    h_valid_count = h_mask.sum().detach()
                    v_valid_count = v_mask.sum().detach()
                    total_valid_count = h_valid_count + v_valid_count
                    
                    if total_valid_count > 0:
                        losses["gradient"] = (h_gradient_sum + v_gradient_sum) / total_valid_count
                    else:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning("No valid gradient pixels for step=2! Cannot perform proper gradient loss training.")
                        losses["gradient"] = torch.tensor(0.0, device=device, dtype=x_image.dtype)
                else:
                    # If image is too small for step size 2, fall back to step size 1
                    # Memory optimization: directly calculate masked gradient, avoid storing intermediate tensors
                    h_mask = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
                    v_mask = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]
                    
                    # Calculate masked gradient sum
                    h_gradient_sum = (torch.abs(log_d_diff[:, :, :, 1:] - log_d_diff[:, :, :, :-1]) * h_mask.float().detach()).sum()
                    v_gradient_sum = (torch.abs(log_d_diff[:, :, 1:, :] - log_d_diff[:, :, :-1, :]) * v_mask.float().detach()).sum()
                    
                    total_valid_count = h_mask.sum().detach() + v_mask.sum().detach()
                    if total_valid_count > 0:
                        losses["gradient"] = (h_gradient_sum + v_gradient_sum) / total_valid_count
                    else:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning("No valid gradient pixels for step=1! Cannot perform proper gradient loss training.")
                        losses["gradient"] = torch.tensor(0.0, device=device, dtype=x_image.dtype)
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("No valid pixels for gradient loss calculation! valid_mask.sum() = {}. Cannot perform proper training.".format(valid_mask.sum().item()))
                losses["gradient"] = torch.tensor(0.0, device=device, dtype=x_image.dtype)
            
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

        feats = self.unet_feature_extractor_cleandift(
                x_0,
                torch.ones((B,), device=device, dtype=self.timestep.dtype) * self.timestep,
                **unet_conds,
            )
        
        feats = feats["sample"]  # shape: (B, D, H, W)

        return feats


