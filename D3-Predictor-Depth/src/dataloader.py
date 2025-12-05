import copy
import json
import os
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from typing import Tuple, List
import torch.nn.functional as F
import h5py
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler, ConcatDataset, DataLoader
import random
import numpy as np

def load_image(path: str, img_size: int):
    image = Image.open(path).convert("RGB")
    resize_transform = transforms.Resize((img_size, img_size))
    image = resize_transform(image)
    image = to_tensor(image) * 2 - 1
    return image

def load_image_from_h5(path: str, img_size: int):
    with h5py.File(path, 'r') as f:
        image = f['depth'][:]  # shape (368, 1232, 3), float32, normalized [0, 1]

    image_tensor = to_tensor(image)  # shape (3, H, W), float32, [0,1]
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = F.interpolate(image_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
    image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor * 2 - 1
    return image_tensor

def load_image_from_h5_2(path: str):
    with h5py.File(path, 'r') as f:
        image = f['depth'][:]  # shape (368, 1232), not normalized
    image_tensor = to_tensor(image) # shape (1, 368, 1232), not normalized
    return image_tensor

class MixedBatchSampler(BatchSampler):
    """
    Mixed batch sampler, implemented with reference to Marigold (https://github.com/prs-eth/Marigold)
    Each batch is sampled from datasets with specified probabilities
    Supports distributed training
    """
    def __init__(
        self, 
        src_dataset_ls: List[data.Dataset], 
        batch_size: int, 
        drop_last: bool, 
        shuffle: bool, 
        prob: List[float] = None, 
        generator=None,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0
    ):
        self.base_sampler = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        
        # Distributed training parameters
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)

        if prob is not None:
            prob = torch.as_tensor(prob, dtype=torch.float)
            if len(prob) != len(src_dataset_ls):
                raise ValueError(f"Number of probabilities ({len(prob)}) does not match number of datasets ({len(src_dataset_ls)})")
            if torch.any(prob < 0):
                raise ValueError("Probabilities cannot be negative")
            if torch.sum(prob) == 0:
                raise ValueError("Sum of probabilities cannot be 0")
            # Normalize probabilities
            self.prob = prob / torch.sum(prob)
        else:
            self.prob = None

        # Dataset lengths
        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]  # Cumulative dataset lengths

        # Create batch samplers for each source dataset
        self._create_batch_samplers()
        
        # Set probabilities (after knowing batch counts)
        if self.prob is None:
            # If no probabilities given, determine based on dataset lengths
            self.prob = torch.tensor(self.n_batches, dtype=torch.float) / self.n_total_batch
    
    def _create_batch_samplers(self):
        """Create batch samplers, supports distributed training"""
        self.src_batch_samplers = []
        
        for ds in self.src_dataset_ls:
            if self.num_replicas > 1:
                # Distributed training: create distributed sampler for each dataset
                base_sampler = DistributedSampler(
                    ds,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    shuffle=self.shuffle,
                    seed=self.seed,
                    drop_last=self.drop_last
                )
            else:
                # Single GPU training
                if self.shuffle:
                    base_sampler = RandomSampler(ds, replacement=False, generator=self.generator)
                else:
                    base_sampler = SequentialSampler(ds)
            
            batch_sampler = BatchSampler(
                sampler=base_sampler,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            self.src_batch_samplers.append(batch_sampler)
        
        for bs in self.src_batch_samplers:
            if hasattr(bs.sampler, "set_epoch"):
                bs.sampler.set_epoch(self.epoch)
        
        self.raw_batches = [
            list(bs) for bs in self.src_batch_samplers
        ]  # Indices in original datasets
        self.n_batches = [len(b) for b in self.raw_batches]
        self.n_total_batch = sum(self.n_batches)
        
        self.active_idx = [i for i, n in enumerate(self.n_batches) if n > 0]
        
        if hasattr(self, "prob") and len(self.n_batches) == len(self.prob):
            prob = self.prob.clone().float()
            # Set to 0 for datasets with no batches
            mask0 = torch.tensor(self.n_batches) == 0
            if mask0.any():
                print(f"Warning: Found {mask0.sum().item()} datasets with no batches, will be excluded from sampling")
                prob[mask0] = 0
            
            # Get probabilities on active set and normalize
            self.prob_active = prob[self.active_idx]
            if self.prob_active.sum() == 0:
                # Fallback: uniform distribution
                self.prob_active = torch.ones(len(self.active_idx), dtype=torch.float) / len(self.active_idx)
            else:
                self.prob_active = self.prob_active / self.prob_active.sum()
        else:
            # Case when prob not provided: proportional to batch counts → normalize only on active set
            tmp = torch.tensor([self.n_batches[i] for i in self.active_idx], dtype=torch.float)
            self.prob_active = tmp / tmp.sum() if tmp.sum() > 0 else torch.ones_like(tmp) / len(tmp)
        
        # Important: total batch count = sum of batch counts in active set
        self.n_total_batch = int(sum(self.n_batches[i] for i in self.active_idx))
        
        if len(self.active_idx) < len(self.n_batches):
            print(f"Active dataset indices: {self.active_idx}")
            print(f"Active sampling probabilities: {self.prob_active.tolist()}")
            print(f"Total valid batch count: {self.n_total_batch}")
    
    def set_epoch(self, epoch: int):
        """Set epoch for randomization in distributed training"""
        self.epoch = epoch
        self._create_batch_samplers()

    def __iter__(self):
        """
        Generator that yields batch indices
        Returns:
            list(int): Indices for one batch, corresponding to ConcatDataset
        """
        if self.generator is not None:
            consistent_seed = self.seed + self.epoch
            self.generator.manual_seed(consistent_seed)
        
        yielded = 0
        active = self.active_idx.copy()
        prob_active = self.prob_active.clone()
        
        while yielded < self.n_total_batch:
            # Select dataset from active set
            k = torch.multinomial(prob_active, 1, replacement=True, generator=self.generator).item()
            idx_ds = active[k]
            
            # If batch list is empty, try to regenerate
            if len(self.raw_batches[idx_ds]) == 0:
                # This dataset is exhausted: remove it from current active set
                active.pop(k)
                if len(active) == 0:
                    break  # Should not happen in theory, since n_total_batch is sum of active set
                
                # Renormalize probabilities of remaining active datasets
                remaining_prob = self.prob_active[[self.active_idx.index(i) for i in active]]
                prob_active = remaining_prob / remaining_prob.sum()
                continue
            
            # Get one batch from selected dataset
            batch_raw = self.raw_batches[idx_ds].pop()
            
            # Offset using cumulative lengths in original order
            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]
            
            yield batch
            yielded += 1

    def __len__(self):
        return self.n_total_batch

class VKITTIDataset(data.Dataset):
    """VKITTI dataset"""
    def __init__(self, dataset_dir: str, img_size: int = 768):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.data = []

        # Treat dense prediction as conditional image generation, hence called "conditions_dir" (same below)
        conditions_dir = os.path.join(dataset_dir, "rgb")
        if not os.path.exists(conditions_dir):
            print(f"Warning: conditions directory does not exist: {conditions_dir}")
            return

        # Only load VKITTI data (files starting with Scene)
        condition_files = [f for f in os.listdir(conditions_dir) 
                          if f.endswith('.jpg') and f.startswith('Scene')]
        
        for condition_file in condition_files:
            image_path = os.path.join(
                os.path.join(dataset_dir, "depth"), 
                os.path.splitext(condition_file)[0] + ".h5"
            )
            depth_unnorm_path = os.path.join(
                os.path.join(dataset_dir, "depth_ori"), 
                os.path.splitext(condition_file)[0] + ".h5"
            )
            json_path = os.path.join(
                dataset_dir, 
                os.path.splitext(condition_file)[0] + ".json"
            )
            
            if os.path.exists(json_path) and os.path.exists(image_path) and os.path.exists(depth_unnorm_path):
                with open(json_path, 'r') as json_file:
                    json_dict = json.load(json_file)
                if "caption" in json_dict.keys():
                    self.data.append({
                        "condition_path": os.path.join(conditions_dir, condition_file), 
                        "image_path": image_path,
                        "depth_unnorm_path": depth_unnorm_path,
                        "caption": json_dict["caption"],
                        "dataset_type": "vkitti"
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {}
        sample["x_condition"] = load_image(item["condition_path"], img_size=self.img_size)
        sample["x_image"] = load_image_from_h5(item["image_path"], img_size=self.img_size)
        sample["x_depth_unnorm"] = load_image_from_h5_2(item["depth_unnorm_path"])
        sample["caption"] = item["caption"]
        sample["dataset_type"] = item["dataset_type"]
        return sample

class HypersimDataset(data.Dataset):
    """Hypersim dataset"""
    def __init__(self, dataset_dir: str, img_size: int = 768):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.data = []

        conditions_dir = os.path.join(dataset_dir, "rgb")
        if not os.path.exists(conditions_dir):
            print(f"Warning: conditions directory does not exist: {conditions_dir}")
            return

        # Only load Hypersim data (files starting with ai)
        condition_files = [f for f in os.listdir(conditions_dir) 
                          if f.endswith('.png') and f.startswith('ai')]
        
        for condition_file in condition_files:
            image_path = os.path.join(
                os.path.join(dataset_dir, "depth"), 
                os.path.splitext(condition_file)[0] + ".h5"
            )
            depth_unnorm_path = os.path.join(
                os.path.join(dataset_dir, "depth_ori"), 
                os.path.splitext(condition_file)[0] + ".h5"
            )
            json_path = os.path.join(
                dataset_dir, 
                os.path.splitext(condition_file)[0] + ".json"
            )
            
            if os.path.exists(json_path) and os.path.exists(image_path) and os.path.exists(depth_unnorm_path):
                with open(json_path, 'r') as json_file:
                    json_dict = json.load(json_file)
                if "caption" in json_dict.keys():
                    self.data.append({
                        "condition_path": os.path.join(conditions_dir, condition_file), 
                        "image_path": image_path,
                        "depth_unnorm_path": depth_unnorm_path,
                        "caption": json_dict["caption"],
                        "dataset_type": "hypersim"
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {}
        sample["x_condition"] = load_image(item["condition_path"], img_size=self.img_size)
        sample["x_image"] = load_image_from_h5(item["image_path"], img_size=self.img_size)
        sample["x_depth_unnorm"] = load_image_from_h5_2(item["depth_unnorm_path"])
        sample["caption"] = item["caption"]
        sample["dataset_type"] = item["dataset_type"]
        return sample

class COCODataset(data.Dataset):
    """COCO dataset"""
    def __init__(self, dataset_dir: str, img_size: int = 768):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.data = []

        conditions_dir = os.path.join(dataset_dir, "rgb")
        if not os.path.exists(conditions_dir):
            print(f"Warning: conditions directory does not exist: {conditions_dir}")
            return

        # Only load COCO data (files starting with coco)
        condition_files = [f for f in os.listdir(conditions_dir) 
                          if f.endswith('.jpg') and f.startswith('coco')]
        
        for condition_file in condition_files:
            image_path = os.path.join(
                os.path.join(dataset_dir, "depth"), 
                os.path.splitext(condition_file)[0] + ".h5"
            )
            depth_unnorm_path = os.path.join(
                os.path.join(dataset_dir, "depth_ori"), 
                os.path.splitext(condition_file)[0] + ".h5"
            )
            json_path = os.path.join(
                dataset_dir, 
                os.path.splitext(condition_file)[0] + ".json"
            )
            
            if os.path.exists(json_path) and os.path.exists(image_path) and os.path.exists(depth_unnorm_path):
                with open(json_path, 'r') as json_file:
                    json_dict = json.load(json_file)
                if "caption" in json_dict.keys():
                    self.data.append({
                        "condition_path": os.path.join(conditions_dir, condition_file), 
                        "image_path": image_path,
                        "depth_unnorm_path": depth_unnorm_path,
                        "caption": json_dict["caption"],
                        "dataset_type": "coco"
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {}
        sample["x_condition"] = load_image(item["condition_path"], img_size=self.img_size)
        sample["x_image"] = load_image_from_h5(item["image_path"], img_size=self.img_size)
        sample["x_depth_unnorm"] = load_image_from_h5_2(item["depth_unnorm_path"])
        sample["caption"] = item["caption"]
        sample["dataset_type"] = item["dataset_type"]
        return sample

class MixedDataModule:
    """
    Mixed data module, implements mixed sampling strategy similar to Marigold (https://github.com/prs-eth/Marigold)
    """
    def __init__(
        self, 
        dataset_dir: str, 
        batch_size: int = 1, 
        img_size: int = 768, 
        mixing_prob: List[float] = [0.76, 0.12, 0.12],  # [Hypersim probability, VKITTI probability, COCO probability]
        seed: int = 0,
        num_workers: int = 1
    ):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.mixing_prob = mixing_prob
        self.seed = seed
        self.num_workers = num_workers
        
        # Default distributed parameters
        self.world_size = 1
        self.rank = 0
        
        # Lazy creation
        self._train_loader = None
        self._hypersim_dataset = None
        self._vkitti_dataset = None
        self._coco_dataset = None
        self._dataset_list = None
        self._prob_list = None

    def set_distributed_params(self, world_size: int, rank: int):
        """Set distributed parameters"""
        self.world_size = world_size
        self.rank = rank
        # Reset loader so it will be recreated on next call
        self._train_loader = None

    def _create_datasets(self):
        """Create datasets (lazy creation)"""
        if self._hypersim_dataset is None:
            self._hypersim_dataset = HypersimDataset(dataset_dir=self.dataset_dir, img_size=self.img_size)
        if self._vkitti_dataset is None:
            self._vkitti_dataset = VKITTIDataset(dataset_dir=self.dataset_dir, img_size=self.img_size)
        if self._coco_dataset is None:
            self._coco_dataset = COCODataset(dataset_dir=self.dataset_dir, img_size=self.img_size)
        
        return self._hypersim_dataset, self._vkitti_dataset, self._coco_dataset

    def _prepare_datasets_and_probs(self):
        """Prepare dataset list and probability list"""
        hypersim_dataset, vkitti_dataset, coco_dataset = self._create_datasets()
        
        if self._dataset_list is None:
            self._dataset_list = []
            self._prob_list = []
            
            if len(hypersim_dataset) > 0:
                self._dataset_list.append(hypersim_dataset)
                self._prob_list.append(self.mixing_prob[0])
            
            if len(vkitti_dataset) > 0:
                self._dataset_list.append(vkitti_dataset)
                self._prob_list.append(self.mixing_prob[1])
            
            if len(coco_dataset) > 0:
                self._dataset_list.append(coco_dataset)
                self._prob_list.append(self.mixing_prob[2])
            
            # Normalize probabilities
            if len(self._prob_list) == 0:
                raise ValueError("No available datasets!")
            
            total_prob = sum(self._prob_list)
            if total_prob <= 0:
                raise ValueError("Sum of probabilities must be greater than 0!")
            
            self._prob_list = [p / total_prob for p in self._prob_list]
        
        return self._dataset_list, self._prob_list

    def train_dataloader(self):
        if self._train_loader is None:
            dataset_list, prob_list = self._prepare_datasets_and_probs()
            
            if self.rank == 0:
                print(f"Hypersim dataset size: {len(dataset_list[0]) if len(dataset_list) > 0 else 0}")
                if len(dataset_list) > 1:
                    print(f"VKITTI dataset size: {len(dataset_list[1])}")
                if len(dataset_list) > 2:
                    print(f"COCO dataset size: {len(dataset_list[2])}")
                print(f"Sampling probabilities: {prob_list}")
            
            # Check if datasets are empty
            if all(len(ds) == 0 for ds in dataset_list):
                raise ValueError("No valid data found!")
            
            # Create mixed dataset
            if len(dataset_list) > 1:
                # Use mixed sampling
                concat_dataset = ConcatDataset(dataset_list)
                
                # Set random seed
                loader_generator = torch.Generator().manual_seed(self.seed) if self.seed else None
                
                # Create mixed batch sampler (supports distributed training)
                mixed_sampler = MixedBatchSampler(
                    src_dataset_ls=dataset_list,
                    batch_size=self.batch_size,
                    drop_last=True,
                    prob=prob_list,
                    shuffle=True,
                    generator=loader_generator,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    seed=self.seed if self.seed else 0
                )
                
                self._train_loader = DataLoader(
                    concat_dataset,
                    batch_sampler=mixed_sampler,
                    num_workers=self.num_workers,
                    pin_memory=True
                )
                
            else:
                # Only one dataset, use regular data loader
                single_dataset = dataset_list[0]
                if self.world_size > 1:
                    train_sampler = DistributedSampler(
                        single_dataset, 
                        num_replicas=self.world_size, 
                        rank=self.rank,
                        shuffle=True,
                        drop_last=True
                    )
                    self._train_loader = DataLoader(
                        single_dataset, 
                        batch_size=self.batch_size, 
                        sampler=train_sampler,
                        num_workers=self.num_workers,
                        pin_memory=True
                    )
                else:
                    self._train_loader = DataLoader(
                        single_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=True,
                        num_workers=self.num_workers,
                        pin_memory=True
                    )
        
        return self._train_loader