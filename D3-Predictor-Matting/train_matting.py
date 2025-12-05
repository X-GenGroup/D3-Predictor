import hydra
import logging
import os
import torch
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from src.utils import set_seed, dict_to
from tqdm.auto import tqdm
import wandb
import time
import deepspeed

def setup_distributed():
    """Initialize distributed environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("No distributed environment variables detected, using single GPU training")
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank

def parse_deepspeed_args():
    """Parse DeepSpeed arguments"""
    import sys
    
    local_rank = 0
    new_argv = []
    
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--local_rank='):
            local_rank = int(arg.split('=')[1])
        elif arg == '--local_rank' and i + 1 < len(sys.argv):
            local_rank = int(sys.argv[i + 1])
            i += 1
        else:
            new_argv.append(arg)
        i += 1
    
    sys.argv = new_argv
    
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    return local_rank

@hydra.main(config_path="configs", config_name="sd21_d3_predictor_matting", version_base="1.1")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    
    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    
    set_seed(cfg.seed)
    logger = logging.getLogger(f"{__name__}")
    
    # Instantiate configuration
    cfg = hydra.utils.instantiate(cfg)
    model = cfg.model

    # DeepSpeed configuration - complete built-in config, no external JSON file needed
    ds_config = {
        "train_batch_size": cfg.get('train_batch_size', cfg.data.batch_size * world_size * cfg.grad_accum_steps),
        "train_micro_batch_size_per_gpu": cfg.data.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "steps_per_print": cfg.get('steps_per_print', 200),
        "wall_clock_breakdown": cfg.get('wall_clock_breakdown', False),
        "memory_breakdown": cfg.get('memory_breakdown', False),
        
        # ZeRO optimization configuration
        "zero_optimization": {
            "stage": cfg.get('zero_stage', 2),
            "allgather_partitions": True,
            "allgather_bucket_size": cfg.get('allgather_bucket_size', 2e8),
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": cfg.get('reduce_bucket_size', 2e8),
            "contiguous_gradients": True,
            "cpu_offload": cfg.get('cpu_offload', False)
        },
        
        # Optimizer configuration
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": cfg.lr,
                "betas": cfg.get('optimizer_betas', [0.9, 0.999]),
                "eps": cfg.get('optimizer_eps', 1e-8),
                "weight_decay": cfg.get('weight_decay', 0.01)
            }
        },
        
        # Learning rate scheduler configuration
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": cfg.get('warmup_min_lr', 0),
                "warmup_max_lr": cfg.lr,
                "warmup_num_steps": cfg.lr_scheduler.num_warmup_steps,
                "total_num_steps": cfg.lr_scheduler.num_training_steps
            }
        },
        
        # Gradient clipping
        "gradient_clipping": cfg.get('gradient_clipping', 1.0),
    }
    
    mixed_precision_type = cfg.get('mixed_precision', 'auto')  # auto, bf16, fp16, none
    
    if mixed_precision_type == 'auto':
        # Auto-detect optimal precision type
        if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            mixed_precision_type = 'bf16'
            if rank == 0:
                print("🎯 Auto-selected: Using BF16 mixed precision training")
        else:
            mixed_precision_type = 'fp16'
            if rank == 0:
                print("🎯 Auto-selected: Using FP16 mixed precision training")
    
    if mixed_precision_type == 'bf16':
        ds_config["bf16"] = {
            "enabled": True
        }
        if rank == 0:
            print("✅ Enabled BF16 mixed precision training (recommended)")
    elif mixed_precision_type == 'fp16':
        ds_config["fp16"] = {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": cfg.get('fp16_initial_scale_power', 16),
            "loss_scale_window": cfg.get('fp16_loss_scale_window', 1000),
            "hysteresis": cfg.get('fp16_hysteresis', 2),
            "min_loss_scale": cfg.get('fp16_min_loss_scale', 1)
        }
        if rank == 0:
            print("✅ Enabled FP16 mixed precision training")
    else:
        if rank == 0:
            print("✅ Using FP32 full precision training")
    
    # Activation checkpointing (optional, for memory saving)
    ds_config["activation_checkpointing"] = {
        "partition_activations": cfg.get('partition_activations', False),
        "cpu_checkpointing": cfg.get('cpu_checkpointing', False),
        "contiguous_memory_optimization": cfg.get('contiguous_memory_optimization', False),
        "synchronize_checkpoint_boundary": cfg.get('synchronize_checkpoint_boundary', False)
    }
    
    # Optional: Performance profiler (for debugging)
    if cfg.get('enable_flops_profiler', False):
        ds_config["flops_profiler"] = {
            "enabled": True,
            "profile_step": cfg.get('profile_step', 1),
            "module_depth": cfg.get('module_depth', -1),
            "top_modules": cfg.get('top_modules', 3),
            "detailed": cfg.get('detailed_profile', True)
        }

    # Optional: If external DeepSpeed config file is provided, use it to override built-in config
    if hasattr(cfg, 'deepspeed_config_path') and cfg.deepspeed_config_path and os.path.exists(cfg.deepspeed_config_path):
        import json
        with open(cfg.deepspeed_config_path, 'r') as f:
            external_config = json.load(f)
        # Merge configs, external config takes priority
        ds_config.update(external_config)
        if rank == 0:
            print(f"✅ Loaded external DeepSpeed config: {cfg.deepspeed_config_path}")
    else:
        if rank == 0:
            print("✅ Using built-in DeepSpeed config")

    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    local_rank = model_engine.local_rank
    global_rank = model_engine.global_rank
    world_size = model_engine.world_size

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        
    if global_rank == 0:
        wandb.init(
            project="d3_predictor",
            name=f"d3_predictor_matting_{time.strftime('%Y%m%d-%H%M%S')}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Create data module, pass distributed parameters
    data = cfg.data
    # Set DataModule's distributed parameters
    data.set_distributed_params(world_size, local_rank)
    
    dataloader_train = data.train_dataloader()

    # Training state initialization
    i_epoch = -1
    accum_count = 0
    stop = False
    max_steps: Optional[int] = cfg.max_steps

    checkpoint_freq: Optional[int] = cfg.checkpoint_freq
    checkpoint_dir: str = cfg.checkpoint_dir
    resume_from_checkpoint: Optional[str] = cfg.get('resume_from_checkpoint', None)
    
    if global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume functionality: restore training from checkpoint
    start_step = 0
    if resume_from_checkpoint:
        if global_rank == 0:
            print(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
        
        # Load checkpoint
        _, client_state = model_engine.load_checkpoint(
            load_dir=checkpoint_dir,
            tag=resume_from_checkpoint,
            load_optimizer_states=True,
            load_lr_scheduler_states=False
        )
        
        if client_state:
            start_step = client_state.get("step", 0)
            i_epoch = client_state.get("epoch", -1)
            lr_scheduler.load_state_dict(client_state["lr_scheduler"])
            if global_rank == 0:
                print(f"✅ Successfully restored to step {start_step}, epoch {i_epoch + 1}, Current LR: {lr_scheduler.get_last_lr()[0]}")
                if "gradient_loss" in client_state:
                    print(f"📊 Restored loss values: gradient_loss={client_state['gradient_loss']:.6f}")
        else:
            if global_rank == 0:
                print("⚠️ client_state not found, starting from step 0")
    else:
        if global_rank == 0:
            print("🚀 Starting new training")

    grad_accum_steps = cfg.grad_accum_steps    
    if global_rank == 0:
        print(f"grad_accum_steps={grad_accum_steps}")
        print(f"world_size={world_size}")
        print(f"rank={global_rank}")

    step = start_step  # Start from restored step
    loss_sum, align_loss_sum, mse_loss_sum, l1_loss_sum, gradient_loss_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    align_losses_sum = {}

    while not stop:  # Epochs
        i_epoch += 1
        
        # Set epoch for distributed sampler
        if hasattr(dataloader_train.batch_sampler, 'set_epoch'):
            dataloader_train.batch_sampler.set_epoch(i_epoch)
            
        for batch in (
            pbar := tqdm(dataloader_train, desc=f"Optimizing (Epoch {i_epoch + 1})", disable=(global_rank != 0))
        ):  
            losses = model_engine(**dict_to(batch, device=model_engine.device))

            align_losses = {k: v for k, v in losses.items() if "sample" not in k and "gradient" not in k}
            mse_sample = losses.get("mse_sample", torch.tensor(0.0, device=model_engine.device))
            l1_sample = losses.get("l1_sample", torch.tensor(0.0, device=model_engine.device))
            gradient = losses.get("gradient", torch.tensor(0.0, device=model_engine.device))

            # Normalize neg_cossim (1 + neg_cossim)
            for k in align_losses:
                if "neg_cossim" in k:
                    align_losses[k] = 1 + align_losses[k]

            align_loss_total = sum(v.mean() for v in align_losses.values())
            mse_loss_total = mse_sample.mean() if hasattr(mse_sample, 'mean') else mse_sample
            l1_loss_total = l1_sample.mean() if hasattr(l1_sample, 'mean') else l1_sample
            gradient_loss_total = gradient.mean() if hasattr(gradient, 'mean') else gradient

            lambda_align = 1.0
            lambda_mse = 5.0
            lambda_l1 = 10.0
            lambda_gradient = 50.0
            total_loss = lambda_align * align_loss_total + lambda_mse * mse_loss_total + lambda_l1 * l1_loss_total + lambda_gradient * gradient_loss_total

            model_engine.backward(total_loss)
            
            loss_sum += float(total_loss.detach().item())
            align_loss_sum += float(align_loss_total.detach().item())
            mse_loss_sum += float(mse_loss_total.detach().item())
            l1_loss_sum += float(l1_loss_total.detach().item())
            gradient_loss_sum += float(gradient_loss_total.detach().item())

            for k, v in align_losses.items():
                align_losses_sum[k] = align_losses_sum.get(k, 0.0) + float(v.mean().detach().item())



            accum_count += 1

            if accum_count % grad_accum_steps == 0:
                model_engine.step()

                avg_loss = loss_sum / grad_accum_steps
                avg_align_loss = align_loss_sum / grad_accum_steps
                avg_mse_loss = mse_loss_sum / grad_accum_steps
                avg_l1_loss = l1_loss_sum / grad_accum_steps
                avg_gradient_loss = gradient_loss_sum / grad_accum_steps
                avg_align_losses = {k: v / grad_accum_steps for k, v in align_losses_sum.items()}


                if global_rank == 0:
                    pbar.set_postfix({ 'loss': avg_loss })
                    wandb.log({
                        "Train/Align Loss Total": avg_align_loss,
                        "Train/MSE Loss Total": avg_mse_loss,
                        "Train/L1 Loss Total": avg_l1_loss,
                        "Train/Gradient Loss Total": avg_gradient_loss,
                        "Train/Total Loss": avg_loss,
                        "Train/Learning Rate": model_engine.get_lr()[0],
                        "Train/Step": step
                    }, step=step)
                    for k, v in avg_align_losses.items():
                        wandb.log({f"Align/{k}": v}, step=step)

                if not checkpoint_freq is None and (step + 1) % checkpoint_freq == 0:
                    tag = f"step_{step + 1}"
                    client_state = {
                        "step": step + 1,
                        "epoch": i_epoch,
                        "loss": avg_loss,
                        "align_loss": avg_align_loss,
                        "mse_loss": avg_mse_loss,
                        "l1_loss": avg_l1_loss,
                        "gradient_loss": avg_gradient_loss,
                        "lr_scheduler": lr_scheduler.state_dict()
                    }
                    
                    model_engine.save_checkpoint(
                        save_dir=checkpoint_dir,
                        tag=tag,
                        client_state=client_state
                    )
                    
                    if global_rank == 0:
                        logger.info(f"✅ Saved checkpoint: {tag} (step: {step + 1}, epoch: {i_epoch + 1})")
                
                step += 1
                loss_sum, align_loss_sum, mse_loss_sum, l1_loss_sum, gradient_loss_sum = 0.0, 0.0, 0.0, 0.0, 0.0
                align_losses_sum = {}

                if not max_steps is None and step == max_steps:
                    stop = True
                    break


if __name__ == "__main__":
    local_rank = parse_deepspeed_args()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    main()
    if int(os.environ.get("RANK", 0)) == 0:
        wandb.finish()