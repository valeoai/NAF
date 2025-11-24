# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal, Optional

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn


def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """
    - x: feature vector of shape [..., D] (e.g., [x0, x1, ..., x_{D-1}])
    - sin, cos: shape [..., D], e.g., [sin0, sin1, ..., sin_{D/2-1}, sin0, ..., sin_{D/2-1}]
    - rope_apply rotates the embedding by pairing the first half with the second half:
    - out = [ x0*cos0 - x_{D/2}*sin0,
            x1*cos1 - x_{D/2+1}*sin1,
            ...,
            x_{D/2}*cos0 + x0*sin0,
            x_{D/2+1}*cos1 + x1*sin1,
            ... ]
    """
    return (x * cos) + (rope_rotate_half(x) * sin)


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RoPE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        min_period: float = None,
        max_period: float = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float = None,
        jitter_coords: float = None,
        rescale_coords: float = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.num_heads = num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self._cached_coords = None
        self._cached_hw = None

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def create_coordinate(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        return coords

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2))
        else:
            periods = torch.logspace(math.log10(self.min_period), math.log10(self.max_period), steps=self.D_head // 4)
        self.periods.data = periods

    def rotate(self, x, coords):
        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        # Flatten u and v frequencies into a single vector per spatial location: [u1, ..., u_{D//4}, v1, ..., v_{D//4}]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        # Repeat the vector to match embedding dimension D: [u1,...,u_{D//4}, v1,...,v_{D//4}, u1,...,u_{D//4}, v1,...,v_{D//4}]
        angles = angles.tile(2)  # [HW, D]

        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        # x_u0 ... x_u{D//2-1} x_v0 ... x_v{D//2-1}

        # Apply RoPE: rotates each (u_i, v_i) pair
        # first half:  x[i]*cos[i] - x[i+D/2]*sin[i]
        # second half: x[i+D/2]*cos[i] + x[i]*sin[i]
        return rope_apply(x, sin, cos)

    def forward(self, x, layout: str = "spatial"):
        h, w = x.shape[-2:]
        x = rearrange(x, "b (n d) h w -> b n (h w) d", n=self.num_heads)

        if (h, w) != self._cached_hw:
            self._cached_coords = self.create_coordinate(H=h, W=w)
            self._cached_hw = (h, w)

        coords = self._cached_coords

        # Rotate
        x = self.rotate(x, coords)

        # Reshape
        if layout == "spatial":
            x = rearrange(x, f"b n (h w) d -> b (n d) h w", h=h, w=w)
        elif layout == "flatten":
            x = rearrange(x, f"b n (h w) d -> b h w n d", h=h, w=w)

        return x
