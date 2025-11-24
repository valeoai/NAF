# import cv2
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange


def create_coordinate(h, w, start=0, end=1, device="cuda", dtype=torch.float32):
    # Create a grid of coordinates
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    # Create a 2D map using meshgrid
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    # Stack the x and y coordinates to create the final map
    coord_map = torch.stack([xx, yy], axis=-1)[None, ...]
    coords = rearrange(coord_map, "b h w c -> b (h w) c", h=h, w=w)
    return coords


class PILToTensor:
    """Convert PIL Image to Tensor"""

    def __call__(self, image):
        image = T.functional.pil_to_tensor(image)
        return image
