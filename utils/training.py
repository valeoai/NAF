import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.transforms as T
from hydra.utils import instantiate
from omegaconf import ListConfig
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import InterpolationMode

from src.backbone.vit_wrapper import PretrainedViTWrapper
from utils.img import PILToTensor


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def round_to_nearest_multiple(value, multiple=14):
    return multiple * round(value / multiple)


def compute_feats(cfg, backbone, image_batch, min_rescale=0.60, max_rescale=0.25):
    _, _, H, W = image_batch.shape  # Get original height and width

    with torch.no_grad():
        hr_feats = backbone(image_batch)

        if cfg.get("lr_img_size", None) is not None:
            size = (cfg.lr_img_size, cfg.lr_img_size)
        else:
            # Downscale
            if cfg.down_factor == "random":
                downscale_factor = np.random.uniform(min_rescale, max_rescale)

            elif cfg.down_factor == "fixed":
                downscale_factor = 0.5

            new_H = round_to_nearest_multiple(H * downscale_factor, backbone.patch_size)
            new_W = round_to_nearest_multiple(W * downscale_factor, backbone.patch_size)
            size = (new_H, new_W)
        low_res_batch = F.interpolate(image_batch, size=size, mode="bilinear")
        lr_feats = backbone(low_res_batch)

        return hr_feats, lr_feats


def logger(args, base_log_dir):
    os.makedirs(base_log_dir, exist_ok=True)
    existing_versions = [
        int(d.split("_")[-1])
        for d in os.listdir(base_log_dir)
        if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("version_")
    ]
    new_version = max(existing_versions, default=-1) + 1
    new_log_dir = os.path.join(base_log_dir, f"version_{new_version}")

    # Create the SummaryWriter with the new log directory
    writer = SummaryWriter(log_dir=new_log_dir)
    return writer, new_version, new_log_dir


def get_dataloaders(cfg, shuffle=True):
    """Get dataloaders for either training or evaluation.

    Args:
        cfg: Configuration object
        backbone: Backbone model for normalization parameters
    """
    # Default ImageNet normalization values
    transforms = {
        "image": T.Compose(
            [
                T.Resize(cfg.img_size, interpolation=InterpolationMode.BILINEAR),
                T.CenterCrop((cfg.img_size, cfg.img_size)),
                T.ToTensor(),
            ]
        )
    }

    transforms["label"] = T.Compose(
        [
            # T.ToTensor(),
            T.Resize(cfg.target_size, interpolation=InterpolationMode.NEAREST_EXACT),
            T.CenterCrop((cfg.target_size, cfg.target_size)),
            PILToTensor(),
        ]
    )
    train_dataset = cfg.dataset
    val_dataset = cfg.dataset.copy()
    if hasattr(val_dataset, "split"):
        val_dataset.split = "val"

    train_dataset = instantiate(
        train_dataset,
        transform=transforms["image"],
        target_transform=transforms["label"],
    )
    val_dataset = instantiate(
        val_dataset,
        transform=transforms["image"],
        target_transform=transforms["label"],
    )

    # Create generator for reproducibility
    if not shuffle:
        g = torch.Generator()
        g.manual_seed(0)
    else:
        g = None

    # Prepare dataloader configs - set worker_init_fn to None when shuffling for randomness
    train_dataloader_cfg = cfg.train_dataloader.copy()
    val_dataloader_cfg = cfg.val_dataloader.copy()

    if shuffle:
        # Set worker_init_fn to None to allow true randomness when shuffling
        if "worker_init_fn" in train_dataloader_cfg:
            train_dataloader_cfg["worker_init_fn"] = None
        if "worker_init_fn" in val_dataloader_cfg:
            val_dataloader_cfg["worker_init_fn"] = None

    return (
        instantiate(train_dataloader_cfg, dataset=train_dataset, generator=g),
        instantiate(val_dataloader_cfg, dataset=val_dataset, generator=g),
    )


def get_batch(batch, device):
    """Process batch and return required tensors."""
    batch["image"] = batch["image"].to(device)
    return batch


def setup_training_optimizations(model, cfg):
    """
    Setup training optimizations based on configuration

    Args:
        model: The model to apply optimizations to
        cfg: Configuration object with use_bf16 and use_checkpointing flags

    Returns:
        tuple: (scaler, use_bf16, use_checkpointing) for use in training loop
    """
    # Get configuration values with defaults
    use_bf16 = getattr(cfg, "use_bf16", False)
    use_checkpointing = getattr(cfg, "use_checkpointing", False)

    # Initialize gradient scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_bf16)

    # Enable gradient checkpointing if requested
    if use_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("   ✓ Using built-in gradient checkpointing")
        else:
            # For custom models, wrap forward methods
            def checkpoint_wrapper(module):
                if hasattr(module, "forward"):
                    original_forward = module.forward

                    def checkpointed_forward(*args, **kwargs):
                        return checkpoint.checkpoint(original_forward, *args, **kwargs)

                    module.forward = checkpointed_forward

            # Apply to key modules (adjust based on your model structure)
            checkpointed_modules = []
            for name, module in model.named_modules():
                if any(key in name for key in ["cross_decode", "encoder", "sft"]):
                    checkpoint_wrapper(module)
                    checkpointed_modules.append(name)

            if checkpointed_modules:
                print(f"   ✓ Applied custom gradient checkpointing to: {checkpointed_modules}")
            else:
                print("   ⚠ No modules found for gradient checkpointing")

    print(f"Training optimizations:")
    print(f"  Mixed precision (bfloat16): {use_bf16}")
    print(f"  Gradient checkpointing: {use_checkpointing}")

    return scaler, use_bf16, use_checkpointing


def load_multiple_backbones(cfg, backbone_configs, device):
    """
    Load multiple backbone models based on configuration.

    Args:
        cfg: Hydra configuration object
        device: PyTorch device to load models on

    Returns:
        tuple: (backbones, backbone_names, primary_backbone)
            - backbones: List of loaded backbone models
            - backbone_names: List of backbone names
    """
    backbones = []
    backbone_names = []
    backbone_img_sizes = []

    if not isinstance(backbone_configs, list) and not isinstance(backbone_configs, ListConfig):
        backbone_configs = [backbone_configs]
    print(f"Loading {len(backbone_configs)} backbone(s)...")

    for i, backbone_config in enumerate(backbone_configs):
        name = backbone_config["name"]
        if name == "rgb":
            backbone = instantiate(cfg.backbone)
        else:
            backbone = PretrainedViTWrapper(name=name)
        print(f"  [{i}] Loaded {backbone_config['name']}")

        # Move to device and set to eval mode
        backbone = backbone.to(device)
        backbone.eval()  # Set to eval mode for feature extraction

        # Store backbone and name
        backbones.append(backbone)
        backbone_names.append(backbone_config["name"])
        backbone_img_sizes.append(backbone.config["input_size"][1:])

    return backbones, backbone_names, backbone_img_sizes
