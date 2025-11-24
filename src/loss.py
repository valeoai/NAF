import torch
import torch.nn as nn
from torch import nn
from torch.nn import functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, pred, target, normalize=False):
        if normalize:
            # If you must normalize (example: min-max scaling)
            min_val = torch.min(target, dim=1, keepdim=True).values
            max_val = torch.max(target, dim=1, keepdim=True).values
            pred_normalized = (pred - min_val) / (max_val - min_val + 1e-6)
            target_normalized = (target - min_val) / (max_val - min_val + 1e-6)
        else:
            pred_normalized = pred
            target_normalized = target

        return self.mse_loss(pred_normalized, target_normalized)


class Loss(nn.Module):

    def __init__(
        self,
        loss_type,
        dim=384,
    ):
        super().__init__()
        self.dim = dim

        if loss_type == "mse":
            loss = MSELoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")

        self.loss_func = loss

    def __call__(self, pred, target, *args, **kwargs):
        loss = self.loss_func(pred, target, *args, **kwargs)
        return {"total": loss}
