import torch.nn.functional as F

from src.model.base import BaseUpsampler


class Nearest(BaseUpsampler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, image, features, output_size, *args, **kwargs):
        # Get upsampled features
        features = F.interpolate(features, size=output_size, mode="nearest-exact")
        return features
