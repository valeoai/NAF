import datetime
import os
import random
import warnings

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.console import Console
from rich.syntax import Syntax
from tqdm import tqdm

# Import project components
from utils.training import get_batch, get_dataloaders, logger, setup_training_optimizations

warnings.filterwarnings("ignore")


class NoiseGenerator:
    """Generate various types of noise for image denoising training"""

    def __init__(self, device="cuda", noise_type="gaussian"):
        self.device = device
        self.noise_type = noise_type

    def add_gaussian_noise(self, image, std=0.1):
        """Add Gaussian noise to image"""
        noise = torch.randn_like(image) * std
        return image + noise

    def add_salt_pepper_noise(self, image, prob=0.05):
        """Add salt and pepper noise"""
        mask = torch.rand_like(image) < prob
        salt_pepper = torch.rand_like(image)
        noise = torch.where(salt_pepper > 0.5, torch.ones_like(image), torch.zeros_like(image))

        return torch.where(mask, noise, image)

    def generate_noisy_image(self, image, noise_params=None):
        """Generate noisy image based on specified noise type"""
        if self.noise_type == "gaussian":
            std = noise_params.get("std", 0.1) if noise_params else 0.1
            if std == "range":
                std = random.uniform(0.1, 0.5)
            return self.add_gaussian_noise(image, std)
        elif self.noise_type == "salt_pepper":
            prob = noise_params.get("prob", 0.1) if noise_params else 0.1
            if prob == "range":
                prob = random.uniform(0.1, 0.5)
            return self.add_salt_pepper_noise(image, prob)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")


class MetricsCalculator:
    """Comprehensive metrics for image denoising evaluation"""

    @staticmethod
    def calculate_psnr(pred, target, max_val=1.0):
        """Calculate PSNR between predicted and target images"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(max_val / torch.sqrt(mse))

    @staticmethod
    def calculate_ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
        """Calculate SSIM for a batch"""

        def create_window(window_size, channel=1):
            gaussian = torch.exp(
                -torch.arange(window_size, dtype=torch.float32).sub(window_size // 2).pow(2) / (2 * (window_size / 6) ** 2)
            )
            gaussian = gaussian / gaussian.sum()
            _1D_window = gaussian.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        if pred.dim() == 4:
            channel = pred.size(1)
            window = create_window(window_size, channel).to(pred.device)

            mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
            mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()
        else:
            # Fallback for single images
            return MetricsCalculator._simple_ssim(pred, target)

    @staticmethod
    def _simple_ssim(pred, target):
        """Simple SSIM calculation for single images"""
        mu1 = pred.mean()
        mu2 = target.mean()
        sigma1 = pred.var()
        sigma2 = target.var()
        sigma12 = ((pred - mu1) * (target - mu2)).mean()

        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        return ssim.item()

    @staticmethod
    def calculate_batch_metrics(pred, target):
        """Calculate PSNR and SSIM metrics for evaluation"""
        metrics = {}
        metrics["psnr"] = MetricsCalculator.calculate_psnr(pred, target).item()
        metrics["ssim"] = MetricsCalculator.calculate_ssim(pred, target).item()
        return metrics


class DenoisingLoss(nn.Module):
    """Combined loss for image denoising"""

    def __init__(self, l1_weight=1.0, l2_weight=1.0, ssim_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.ssim_weight = ssim_weight

        # L1 and L2 losses
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def ssim_loss(self, pred, target):
        """Simplified SSIM loss"""
        mu1 = F.avg_pool2d(pred, 3, 1, 1)
        mu2 = F.avg_pool2d(target, 3, 1, 1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

    def forward(self, pred, target):
        """Compute combined denoising loss"""
        losses = {}

        # Basic reconstruction losses
        if self.l1_weight > 0:
            losses["l1"] = self.l1_loss(pred, target) * self.l1_weight

        if self.l2_weight > 0:
            losses["l2"] = self.l2_loss(pred, target) * self.l2_weight

        # SSIM loss for perceptual quality
        if self.ssim_weight > 0:
            losses["ssim"] = self.ssim_loss(pred, target) * self.ssim_weight

        losses["total"] = sum(losses.values())
        return losses


def train_epoch(model, dataloader, noise_gen, criterion, optimizer, cfg, use_bf16, device, epoch, writer, global_step):
    """Train one epoch"""
    model.train()
    epoch_losses = {"total": [], "l1": [], "l2": [], "ssim": []}

    img_size = cfg.img_size
    noise_params = cfg.denoising.noise_params

    # Image normalization for model input
    mean_ups = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_ups = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for _, batch in enumerate(pbar):
        # Get clean images
        batch = get_batch(batch, device)
        if "clean" in batch.keys():
            noisy_images = batch["image"]
            clean_images = batch["clean"].to(device)
        else:
            clean_images = batch["image"]
            noisy_images = noise_gen.generate_noisy_image(clean_images, noise_params)
        if clean_images.shape[-1] != img_size or clean_images.shape[-2] != img_size:
            clean_images = F.interpolate(clean_images, size=(img_size, img_size), mode="bilinear", align_corners=False)
            noisy_images = F.interpolate(noisy_images, size=(img_size, img_size), mode="bilinear", align_corners=False)
        optimizer.zero_grad()

        # Mixed precision training
        with torch.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16 if use_bf16 else torch.float32):

            # Forward pass through model
            noisy_images_normalized = (noisy_images - mean_ups) / std_ups
            denoised_images = model(noisy_images_normalized, noisy_images, (img_size, img_size))

            # Calculate denoising loss
            losses = criterion(denoised_images, clean_images)

            if global_step % cfg.freq_viz == 0:
                # Visualize denoising results every 100 steps
                writer.add_images("Train/Noisy", noisy_images, global_step)
                writer.add_images("Train/Denoised", denoised_images, global_step)
                writer.add_images("Train/Clean", clean_images, global_step)

        # Backward pass
        losses["total"].backward()
        optimizer.step()

        # Track losses
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                epoch_losses[key].append(value.item())

        # Increment global step counter
        global_step += 1

        # Log to tensorboard
        if global_step % 50 == 0:  # Log every 50 steps
            for loss_name, loss_value in losses.items():
                if loss_name != "total" and loss_value != 0:
                    writer.add_scalar(
                        f"Loss/{loss_name}",
                        loss_value.item() if hasattr(loss_value, "item") else loss_value,
                        global_step,
                    )
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], global_step)

        # Update progress bar (removed metric calculations)
        pbar.set_postfix(
            {
                "L1": f"{losses['l1'].item():.4f}" if "l1" in losses else "0.0000",
                "L2": f"{losses['l2'].item():.4f}" if "l2" in losses else "0.0000",
                "SSIM_Loss": f"{losses['ssim'].item():.4f}" if "ssim" in losses else "0.0000",
                "Total": f"{losses['total'].item():.4f}",
            }
        )

        # Stopping criterion
        if global_step >= cfg.train_steps:
            break

    # Calculate epoch averages
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}

    return avg_losses, global_step


def validate_model(model, dataloader, noise_gen, metrics_calc, cfg, device, use_bf16):
    """Validate the model"""
    model.eval()
    val_metrics = {"psnr": [], "ssim": []}

    img_size = cfg.img_size
    noise_params = cfg.denoising.noise_params

    # Image normalization for model input
    mean_ups = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_ups = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc="Validation")

        for step, batch in enumerate(val_pbar):
            if step >= cfg.val_steps:  # Limit validation
                break

            batch = get_batch(batch, device)
            if "clean" in batch.keys():
                noisy_images = batch["image"]
                clean_images = batch["clean"].to(device)
            else:
                clean_images = batch["image"]
                noisy_images = noise_gen.generate_noisy_image(clean_images, noise_params)
            if clean_images.shape[-1] != img_size or clean_images.shape[-2] != img_size:
                clean_images = F.interpolate(clean_images, size=(img_size, img_size), mode="bilinear", align_corners=False)
                noisy_images = F.interpolate(noisy_images, size=(img_size, img_size), mode="bilinear", align_corners=False)

            # Model denoising
            noisy_images_normalized = (noisy_images - mean_ups) / std_ups

            with torch.autocast("cuda", enabled=use_bf16):
                denoised_images = model(noisy_images_normalized, noisy_images, (img_size, img_size))
                denoised_images = torch.clamp(denoised_images, 0, 1)

            # Calculate metrics
            batch_metrics = metrics_calc.calculate_batch_metrics(denoised_images, clean_images)

            for metric_name, value in batch_metrics.items():
                val_metrics[metric_name].append(value)

    # Calculate averages
    avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
    return avg_val_metrics


@hydra.main(config_path="config", config_name="base_denoising", version_base=None)
def main(cfg: DictConfig):
    # Print configuration
    yaml_syntax = Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="monokai", line_numbers=True)
    print(yaml_syntax)

    # ============ Logger ============ #
    log_dir = HydraConfig.get().runtime.output_dir
    writer, _, new_log_dir = logger(cfg, log_dir)

    terminal_console = Console()  # Terminal output
    file_name = os.path.join(log_dir, "train_denoising.log")
    file_console = Console(
        file=open(file_name, "w"),
    )

    def log_print(*args, **kwargs):
        """Log to both terminal and file with immediate flushing"""
        # Print to terminal
        terminal_console.print(*args, **kwargs)
        # Print to file and flush immediately
        file_console.print(*args, **kwargs)
        file_console.file.flush()  # Force immediate write to disk

    # Start logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")

    # Setup data loaders
    train_dataloader, val_dataloader = get_dataloaders(cfg, shuffle=True)
    log_print(f"[bold cyan]Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}[/bold cyan]")

    # Initialize model
    model = instantiate(cfg.model)
    model.to(device)

    # Initialize noise generator
    noise_gen = NoiseGenerator(device=device, noise_type=cfg.denoising.noise_type)

    # Initialize loss and optimizer
    criterion = DenoisingLoss(
        l1_weight=cfg.denoising.loss.l1_weight,
        l2_weight=cfg.denoising.loss.l2_weight,
        ssim_weight=cfg.denoising.loss.ssim_weight,
    )

    if cfg.train_steps > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    # Setup training optimizations
    _, use_bf16, _ = setup_training_optimizations(model, cfg)

    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()

    # Training loop
    log_print(f"[bold green]Starting denoising training for {cfg.epochs} epochs...[/bold green]")
    log_print(f"[bold cyan]Training approach: L1 + L2 + SSIM loss optimization[/bold cyan]")
    log_print(f"[bold cyan]Evaluation metrics: PSNR + SSIM[/bold cyan]")
    log_print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    training_history = {"val_metrics": []}

    global_step = 0
    epoch = 0
    while global_step < cfg.train_steps:
        train_losses, global_step = train_epoch(
            model, train_dataloader, noise_gen, criterion, optimizer, cfg, use_bf16, device, epoch, writer, global_step
        )
        val_metrics = validate_model(model, val_dataloader, noise_gen, metrics_calc, cfg, device, use_bf16)
        training_history["val_metrics"].append(val_metrics)

        # Log validation metrics to tensorboard
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f"Val/{metric_name.upper()}", metric_value, epoch)

        # Log training losses to tensorboard (epoch summary)
        for loss_name, loss_value in train_losses.items():
            writer.add_scalar(f"Train_Epoch/{loss_name}", loss_value, epoch)

        # Print epoch summary
        log_print(f"\n[bold]Epoch {epoch+1}/{cfg.epochs} Summary:[/bold]")
        log_print(
            f"[cyan]Training Losses - L1: {train_losses['l1']:.4f}, L2: {train_losses['l2']:.4f}, SSIM: {train_losses['ssim']:.4f}[/cyan]"
        )
        log_print(f"Val   Evaluation - PSNR: {val_metrics['psnr']:.2f}dB, SSIM: {val_metrics['ssim']:.3f}")

        writer.flush()
        epoch += 1

    # Final model save
    final_model_path = os.path.join(new_log_dir, f"final_denoising_model.pth")
    torch.save(model.state_dict(), final_model_path)

    log_print(f"\n[bold green]Training completed![/bold green]")
    log_print(f"[bold cyan]Training approach: L1 + L2 + SSIM loss optimization[/bold cyan]")
    log_print(f"[bold cyan]Evaluation metrics: PSNR + SSIM[/bold cyan]")
    log_print(f"Final validation PSNR: {val_metrics['psnr']:.2f}dB")
    log_print(f"Models saved in: {new_log_dir}")

    file_console.file.close()


if __name__ == "__main__":
    main()

# ? python train_denoising.py train_steps=4000 val_steps=1000 denoising.noise_params.std=0.5 model=dncnn
# model: dncnn
# 18.20 (1000 steps)
# 20.86 (4000 steps)
# model: ircnn
# 22.44 (4000 steps)
# model: rednet
# 23.60 (4000 steps)
# model: naf
# 23.92 (4000 steps) - ks: 15 - dim: 96 - enc: 2 - heads: 1-1 / 0.1 M params
# 23.92 (4000 steps) - ks: 15 - dim: 96 - enc: 2 - heads: 1-4
# 20.26 (4000 steps) - ks: 3 - dim: 96 - enc: 2 - heads: 1-1
# 22.50 (4000 steps) - ks: 5 - dim: 96 - enc: 2 - heads: 1-1
# 23.05 (4000 steps) - ks: 7 - dim: 96 - enc: 2 - heads: 1-1
# 23.34 (4000 steps) - ks: 9 - dim: 96 - enc: 2 - heads: 1-1
# 23.56 (4000 steps) - ks: 11 - dim: 96 - enc: 2 - heads: 1-1
# 23.73 (4000 steps) - ks: 15 - dim: 128 - enc: 1 - heads: 1-1 : 0.08 M params
# 23.96 (4000 steps) - ks: 15 - dim: 128 - enc: 2 - heads: 1-1 : 0.167 M params
# 24.09 (4000 steps) - ks: 15 - dim: 256 - enc: 2 - heads: 1-1: 0.662 M params
# 24.19 (4000 steps) - ks: 15 - dim: 256 - enc: 2 - heads: 1-1: 0.662 M params - bs: 8
# 23.77 (4000 steps) - ks: 15 - dim: 512 - enc: 2 - heads: 1-1: 2.65 M params
# 23.77 (4000 steps) - ks: 15 - dim: 512 - enc: 2 - heads: 1-1: 2.65 M params
# 23.77 (4000 steps) - ks: 15 - dim: 512 - enc: 2 - heads: 1-1: 2.65 M params
# 23.77 (4000 steps) - ks: 15 - dim: 512 - enc: 2 - heads: 1-1: 2.65 M params
# 23.77 (4000 steps) - ks: 15 - dim: 512 - enc: 2 - heads: 1-1: 2.65 M params
