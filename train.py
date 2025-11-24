import datetime
import os
import random

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, log
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.console import Console
from rich.syntax import Syntax
from tqdm import tqdm

from utils.training import (
    compute_feats,
    get_batch,
    get_dataloaders,
    load_multiple_backbones,
    logger,
    setup_training_optimizations,
)

FREQ = 100


@hydra.main(config_path="config", config_name="base")
def trainer(cfg: DictConfig):
    # ============ Logger ============ #
    log_dir = HydraConfig.get().runtime.output_dir
    writer, _, new_log_dir = logger(cfg, log_dir)

    terminal_console = Console()  # Terminal output
    file_name = os.path.join(log_dir, "train.log")
    file_console = Console(
        file=open(file_name, "w"),
    )

    def log_print(*args, **kwargs):
        """Log to both terminal and file with immediate flushing"""
        terminal_console.print(*args, **kwargs)
        file_console.print(*args, **kwargs)
        file_console.file.flush()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    # ============ Backbones ============ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbones, names, _ = load_multiple_backbones(cfg, cfg.backbone, device)
    backbone = backbones[0]
    name = names[0]
    ups_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    back_norm = T.Normalize(mean=backbone.config["mean"], std=backbone.config["std"])
    log_print(f"[bold cyan]Loaded {len(backbones)} backbones: {names}[/bold cyan]")
    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")
    log_print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")
    img_batch_size = (cfg.img_size, cfg.img_size)

    # ============ Model ============ #
    model = instantiate(cfg.model)
    model.cuda()
    model.train()

    if cfg.model_ckpt is not None:
        model.load_state_dict(torch.load(cfg.model_ckpt, map_location=device))
        log_print(f"[bold green]Loaded model checkpoint from {cfg.model_ckpt}[/bold green]")

    # ============ Optimizers ============ #
    all_params = []
    all_params.extend(list(model.parameters()))
    print("Number of parameters: ", sum(p.numel() for p in all_params))
    optimizer_model = instantiate(cfg.optimizer, params=all_params)

    # ============ Losses ============ #
    criterion = {name: instantiate(cfg) for name, cfg in cfg.loss.items()}

    # ============ Training Optimizations ============ #
    _, use_bf16, _ = setup_training_optimizations(model, cfg)
    log_print(f"[bold yellow]Training optimizations: bf16={use_bf16}[/bold yellow]")

    # ============ Datasets and Dataloaders ============ #
    train_dataloader, _ = get_dataloaders(cfg)
    log_print(f"[bold cyan]Train Dataset size: {len(train_dataloader.dataset)}[/bold cyan]")

    # ============ Training ============ #
    total_batches = cfg.train_steps
    checkpoint_interval = total_batches // 4

    # Calculate total training steps
    total_epochs = cfg.epochs
    total_steps = total_epochs * total_batches

    # Loop
    loss = {}
    for epoch in range(cfg.epochs):
        # Training loop
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            current_step = epoch * len(train_dataloader) + batch_idx
            overall_progress = (current_step / total_steps) * 100

            batch = get_batch(batch, device)
            img_batch = batch["image"]

            optimizer_model.zero_grad()
            loss[name] = 0.0

            # Prepare Images
            img_batch = F.interpolate(img_batch, size=img_batch_size, mode="bilinear", align_corners=False)
            img_ups = ups_norm(img_batch)
            img_back = back_norm(img_batch)

            # Mixed precision context
            with torch.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16 if use_bf16 else torch.float32):

                # ============ Extract Backbone Features ============ #
                hr_feats, lr_feats = compute_feats(cfg, backbone, img_back)

                # Main HR prediction loss
                img_ups_hr = F.interpolate(img_ups, size=[min(224, v * 4) for v in hr_feats.shape[-2:]], mode="bilinear")
                pred_feats = model(img_ups_hr, lr_feats, hr_feats.shape[-2:])

                pred_feats = pred_feats.to(torch.float32)
                target_feats = hr_feats.to(torch.float32)

                loss[name] += criterion["mse"](pred_feats, target_feats, normalize=False)["total"]

            # Total loss calculation
            total_loss = sum([v for v in loss.values() if v != 0.0])
            total_loss.backward()
            optimizer_model.step()

            if batch_idx % FREQ == 0:
                for loss_name, loss_value in loss.items():
                    if loss_value != 0:
                        writer.add_scalar(
                            f"Loss/{loss_name}",
                            loss_value.item() if hasattr(loss_value, "item") else loss_value,
                            current_step,
                        )

                writer.add_scalar("Learning Rate", optimizer_model.param_groups[0]["lr"], current_step)
                loss_str = " | ".join(
                    [f"{k}: {v.item():.4f}" if hasattr(v, "item") else f"{k}: {v:.4f}" for k, v in loss.items() if v != 0]
                )
                log_print(
                    f"Epoch={epoch}/{total_epochs} | "
                    f"Batch={batch_idx}/{len(train_dataloader)} | "
                    f"Progress: {overall_progress:.1f}% | "
                    f"Image Size={img_batch_size} | "
                    f"{loss_str} | "
                )

            # Checkpointing
            if (batch_idx % checkpoint_interval == 0 and batch_idx != 0) or (current_step >= cfg.train_steps):
                checkpoint_path = os.path.join(new_log_dir, f"model_{current_step}steps.pth")
                torch.save(model.state_dict(), checkpoint_path)
                log_print(f"Saved checkpoint: {checkpoint_path}")

                if current_step >= cfg.train_steps:
                    break

            if cfg.sanity:
                break

        writer.flush()

    file_console.file.close()


if __name__ == "__main__":
    trainer()
