import torch
import torch.nn.functional as F
from torch import nn

from src.model.base import BaseUpsampler


class AnyUpsampler(BaseUpsampler):
    def __init__(self, dim=256, radius=3, groups=8, *args, **kwargs):
        super().__init__()
        self.upsampler = torch.hub.load("wimmerth/anyup", "anyup")

    def forward(self, image, features, output_size, *args, **kwargs):
        image = F.interpolate(image, size=output_size, mode="bilinear", align_corners=False)
        return self.upsampler(image, features)
