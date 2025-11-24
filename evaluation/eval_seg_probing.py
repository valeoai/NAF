import datetime
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy, JaccardIndex
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from utils.training import get_batch, get_dataloaders, load_multiple_backbones

LOG_INTERVAL = 100
IGNORE = 255


class UpsamplerEvaluator:

    def __init__(self, model, backbone, device, cfg, writer, console):
        self.model, self.backbone, self.device, self.cfg, self.writer, self.console = (
            model,
            backbone,
            device,
            cfg,
            writer,
            console,
        )

        # Use backbone-specific normalization (like train_jafar.py)
        self.mean_bck = backbone.config["mean"]
        self.std_bck = backbone.config["std"]

        # Upsampler normalization (ImageNet standard)
        self.mean_ups = (0.485, 0.456, 0.406)
        self.std_ups = (0.229, 0.224, 0.225)

        # Initialize task-specific components
        self.accuracy_metric = Accuracy(num_classes=cfg.metrics.seg.num_classes, task="multiclass").to(device)
        self.iou_metric = JaccardIndex(num_classes=cfg.metrics.seg.num_classes, task="multiclass").to(device)
        self.classifier = nn.Conv2d(backbone.embed_dim, cfg.metrics.seg.num_classes, 1).to(device)

    def set_up_classifier(self, checkpoint_path):
        """Load classifier weights from a checkpoint."""
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path)
            self.classifier.load_state_dict(checkpoint)
            self.console.print(f"Loaded classifier from checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    def set_optimizer(self, cfg, loader):
        params = []
        params_classifier = self.classifier.parameters()
        params_model = self.model.parameters()

        params = list(params_classifier)
        optimizer = instantiate(cfg.optimizer, params=params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs * len(loader))
        self.optimizer = optimizer
        self.scheduler = scheduler

        num_params = sum(p.numel() for p in params if p.requires_grad)
        self.log_print(f"[bold cyan]Number of optimized parameters: {num_params:,}[/bold cyan]")

    def log_print(self, *args, **kwargs):
        Console(force_terminal=True).print(*args, **kwargs)
        self.console.print(*args, **kwargs)
        if hasattr(self.console, "file") and self.console.file:
            self.console.file.flush()

    def log_tensorboard(self, step, loss=None, metrics=None):
        if loss is not None:
            self.writer.add_scalar("Loss/Step", loss, step)
        if metrics is not None:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, step)

    def process_batch(self, image_batch, target, batch_idx=0):
        H, W = target.shape[-2:]

        image_batch_bck = T.functional.normalize(image_batch, mean=self.mean_bck, std=self.std_bck)
        image_batch_ups = T.functional.normalize(image_batch, mean=self.mean_ups, std=self.std_ups)

        with torch.no_grad():
            pred = self.backbone(image_batch_bck)

        input_shape = pred.shape[-2:]
        with torch.no_grad():
            pred = self.model(image_batch_ups, pred, (H, W)).detach()

        if batch_idx == 0:
            self.log_print(
                f"[red]Model input size {input_shape}, output size {pred.shape[-2:]} for expected size {(H, W)}[/red]"
            )
        pred = self.classifier(pred)

        if pred.shape[-2:] != (H, W):
            pred = F.interpolate(pred, size=(H, W), mode="bilinear")

        if target.shape[-2:] != pred.shape[-2:]:
            target = (
                F.interpolate(
                    target.unsqueeze(1),
                    size=pred.shape[-2:],
                    mode="nearest-exact",
                )
                .squeeze(1)
                .to(target.dtype)
            )

        valid_mask = target != IGNORE

        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b h w -> (b h w)")
        valid_mask = rearrange(valid_mask, "b h w -> (b h w)")

        pred = pred[valid_mask]
        target = target[valid_mask]
        return pred, target

    def train(
        self,
        train_dataloader,
        progress,
        epoch,
        start_time,
    ):
        self.log_print(f"[yellow]Training model epoch {epoch+1}...[/yellow]")
        self.backbone.eval()
        self.model.eval()
        self.classifier.train()

        epoch_task = progress.add_task(
            f"Epoch {epoch+1}/{self.cfg.num_epochs}",
            total=len(train_dataloader),
            loss=0.0,
            step=0,
        )
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            batch = get_batch(batch, self.device)
            image_batch = batch["image"]
            target = batch["label"].to(self.device)

            if random.random() < 0.5:
                assert len(image_batch.shape) == 4 and len(target.shape) == 3
                image_batch = torch.flip(image_batch, dims=[3])
                target = torch.flip(target, dims=[2])

            self.optimizer.zero_grad()

            pred, target = self.process_batch(image_batch, target, batch_idx=batch_idx)

            loss = F.cross_entropy(pred, target)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            avg_loss = total_loss / (batch_idx + 1)

            if (batch_idx + 1) % LOG_INTERVAL == 0 or batch_idx == len(train_dataloader) - 1:
                elapsed_time = datetime.datetime.now() - start_time
                elapsed_str = str(elapsed_time).split(".")[0]
                current_lr = self.optimizer.param_groups[0]["lr"]

                progress.update(epoch_task, advance=LOG_INTERVAL, loss=avg_loss, step=batch_idx + 1)

                self.log_print(
                    f"[cyan]Iteration {batch_idx + 1}[/cyan] - "
                    f"Loss: {avg_loss:.6f} - "
                    f"LR: {current_lr:.5e} - "
                    f"Elapsed Time: {elapsed_str}"
                )
                if self.console and hasattr(self.console, "file"):
                    self.console.file.flush()

                progress.refresh()

                self.log_tensorboard(epoch * len(train_dataloader) + batch_idx, loss=avg_loss)

            if self.cfg.sanity and batch_idx == 0:
                break

            self.scheduler.step()

            if self.cfg.sanity:
                break

        current_lr = self.optimizer.param_groups[0]["lr"]
        self.log_print(
            f"[bold cyan]Epoch {epoch+1} Summary:[/bold cyan] " f"Loss = {avg_loss:.6f} - " f"LR = {current_lr:.2e}"
        )

        return

    def save_checkpoint(self, checkpoint_path):
        console = self.console
        torch.save(self.classifier.state_dict(), checkpoint_path)
        self.log_print(f"[bold green]Training completed. Model saved at: {checkpoint_path}[/bold green]")
        console.file.flush()
        return

    @torch.inference_mode()
    def evaluate(self, dataloader, epoch):
        self.log_print("[yellow]Evaluating model...[/yellow]")
        torch.cuda.empty_cache()

        self.backbone.eval()
        self.model.eval()
        self.classifier.eval()

        self.accuracy_metric.reset()
        self.iou_metric.reset()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = get_batch(batch, self.device)
            image_batch = batch["image"]
            target = batch["label"].to(self.device)

            pred, target = self.process_batch(image_batch, target, batch_idx=batch_idx)

            self.accuracy_metric(pred, target)
            self.iou_metric(pred, target)

            if self.cfg.sanity and batch_idx == 0:
                break

        metrics = {}
        metrics.update(
            {
                "accuracy": self.accuracy_metric.compute().item(),
                "iou": self.iou_metric.compute().item(),
            }
        )

        self.log_tensorboard(step=epoch, metrics=metrics)

        self.log_print(f"[bold green]Results: {metrics}[/bold green]")
        return

    @torch.inference_mode()
    def simple_inference(self, image_batch):
        self.backbone.eval()
        self.model.eval()
        self.classifier.eval()

        H, W = image_batch.shape[-2:]
        image_batch_bck = T.functional.normalize(image_batch, mean=self.mean_bck, std=self.std_bck)
        image_batch_ups = T.functional.normalize(image_batch, mean=self.mean_ups, std=self.std_ups)

        with torch.no_grad():
            lr_feats = self.backbone(image_batch_bck)
            hr_feats = self.model(image_batch_ups, lr_feats, (H, W))

        labels = self.classifier(hr_feats).argmax(dim=1, keepdim=True)
        return hr_feats, labels, lr_feats


@hydra.main(config_path="../config", config_name="eval_probing")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    current_run_dir = hydra_cfg.runtime.output_dir

    task = cfg.eval.task
    model_name = cfg.model.name if hasattr(cfg.model, "name") else "base"

    # Setup Classifier checkpoint in current Hydra directory
    checkpoint_filename = "linear_probe.pth"
    checkpoint_path = os.path.join(current_run_dir, checkpoint_filename)

    # Create persistent consoles instead of creating new ones each time
    terminal_console = Console()
    tag = ""
    if cfg.eval.model_ckpt:
        try:
            tag = cfg.eval.model_ckpt.split("/")[-3]
        except:
            tag = ""

    # Create log file names
    file_name = f"train_{model_name}_{tag}_{task}.log"

    # All outputs in Hydra's current directory
    log_file_path = os.path.join(current_run_dir, file_name)
    tb_dir = os.path.join(current_run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    file_console = Console(file=open(log_file_path, "w"))

    # Initialize TensorBoard writer in current Hydra directory
    writer = SummaryWriter(log_dir=tb_dir)

    def log_print(*args, **kwargs):
        """Log to both terminal and file with immediate flushing"""
        terminal_console.print(*args, **kwargs)
        file_console.print(*args, **kwargs)
        file_console.file.flush()

    # Start logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")

    log_print(f"\n[bold cyan]Processing {task} task:[/bold cyan]")
    log_print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")

    # Setup Backbone
    backbones, *_ = load_multiple_backbones(cfg, cfg.backbone, device)
    backbone = backbones[0]
    backbone.requires_grad_(False)
    backbone.eval()

    # Setup Model
    model = instantiate(cfg.model).to(device)
    if cfg.eval.model_ckpt:
        checkpoint = torch.load(cfg.eval.model_ckpt, map_location=device, weights_only=False)
        if cfg.model.name == "featup":
            new_ckpts = {
                k.replace("model.1.", "norm."): v
                for k, v in checkpoint["state_dict"].items()
                if "upsampler" in k or "model.1.norm" in k
            }
            model.load_state_dict(new_ckpts, strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        log_print(f"[green]Loaded model from checkpoint: {cfg.eval.model_ckpt}[/green]")
    else:
        log_print(f"[yellow]No model checkpoint provided, using untrained model[/yellow]")

    model.requires_grad_(True)
    model.train()

    # Setup Dataloaders - Modified to match train_jafar.py approach
    train_loader, val_loader = get_dataloaders(cfg, shuffle=False)
    log_print(f"[bold cyan]Train Dataset size: {len(train_loader.dataset)}[/bold cyan]")
    log_print(f"[bold cyan]Val Dataset size: {len(val_loader.dataset)}[/bold cyan]")

    # Setup Evaluator
    evaluator = UpsamplerEvaluator(model, backbone, device, cfg, writer, file_console)

    log_print("[yellow]Training classifier...[/yellow]\n")
    evaluator.set_optimizer(cfg, loader=train_loader)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[yellow]Loss: {task.fields[loss]:.6f}"),
        TextColumn("[green]Step: {task.fields[step]:.6e}"),
        console=file_console,
    )

    start_time = datetime.datetime.now()

    with progress:
        log_print(f"[yellow]Training for {cfg.num_epochs} epochs[/yellow]\n")

        for epoch in range(cfg.num_epochs):
            evaluator.train(train_loader, progress, epoch, start_time)
            evaluator.evaluate(val_loader, epoch)

        evaluator.save_checkpoint(checkpoint_path)

    file_console.file.close()
    writer.close()


if __name__ == "__main__":
    main()
