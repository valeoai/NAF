# Visualization code from https://github.com/Tsingularity/dift/blob/main/src/utils/visualization.py

import io
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

FONT_SIZE = 40


@torch.no_grad()
def plot_feats(
    image,
    target,
    pred,
    legend=["Image", "HR Features", "Pred Features"],
    save_path=None,
    return_array=False,
    show_legend=True,
    font_size=FONT_SIZE,
):
    """
    Create a plot_feats visualization.
    """
    # Ensure hr_or_seg is a list
    if not isinstance(pred, list):
        pred = [pred]

    # Prepare inputs for PCA
    feats_for_pca = [target.unsqueeze(0)] + [_.unsqueeze(0) for _ in pred]
    reduced_feats, _ = pca(feats_for_pca)  # pca outputs a list of reduced tensors

    target_imgs = reduced_feats[0]
    pred_imgs = reduced_feats[1:]

    # --- Plot ---
    # Determine number of columns based on whether image is provided
    n_cols = (1 if image is not None else 0) + 1 + len(pred_imgs)
    fig, ax = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    # Reduce space between images
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Handle single subplot case
    if n_cols == 1:
        ax = [ax]

    # Current axis index
    ax_idx = 0

    # Plot original image if provided
    if image is not None:
        if image.dim() == 3:
            ax[ax_idx].imshow(image.permute(1, 2, 0).detach().cpu())
        elif image.dim() == 2:
            ax[ax_idx].imshow(image.detach().cpu(), cmap="inferno")
        if show_legend:
            ax[ax_idx].set_title(legend[0], fontsize=font_size)
        ax_idx += 1

    # Plot the low-resolution features or segmentation mask
    ax[ax_idx].imshow(target_imgs[0].permute(1, 2, 0).detach().cpu())
    if show_legend:
        legend_idx = 1 if image is not None else 0
        ax[ax_idx].set_title(legend[legend_idx], fontsize=font_size)
    ax_idx += 1

    # Plot HR features or segmentation masks
    for idx, pred_img in enumerate(pred_imgs):
        ax[ax_idx].imshow(pred_img[0].permute(1, 2, 0).detach().cpu())
        if show_legend:
            legend_idx = (2 if image is not None else 1) + idx
            if len(legend) > legend_idx:
                ax[ax_idx].set_title(legend[legend_idx], fontsize=font_size)
            else:
                ax[ax_idx].set_title(f"HR Features {idx}", fontsize=font_size)
        ax_idx += 1

    remove_axes(ax)

    # Handle return_array case
    if return_array:
        # Turn off interactive mode temporarily
        was_interactive = plt.isinteractive()
        plt.ioff()

        # Convert figure to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        # Convert to PIL Image then to numpy array
        pil_img = Image.open(buf)
        img_array = np.array(pil_img)

        # Close the figure and buffer
        plt.close(fig)
        buf.close()

        # Restore interactive mode if it was on
        if was_interactive:
            plt.ion()

        return img_array

    # Standard behavior: save and/or show
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()

    return None


def remove_axes(axes):
    def _remove_axes(ax):
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.set_xticks([])
        ax.set_yticks([])

    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


def pca(image_feats_list, dim=3, fit_pca=None, max_samples=None):
    target_size = None
    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]

    def flatten(tensor, target_size=None):
        B, C, H, W = tensor.shape
        assert B == 1, "Batch size should be 1 for PCA flattening"
        if target_size is not None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear", align_corners=False)
        return rearrange(tensor, "b c h w -> (b h w) c").detach().cpu()

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        fit_pca = TorchPCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        B, C, H, W = feats.shape
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2))

    return reduced_feats, fit_pca


class TorchPCA(object):

    def __init__(self, n_components, skip=0):
        self.n_components = n_components
        self.skip = skip

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=20)
        self.components_ = V[:, self.skip :]
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_
        return projected
