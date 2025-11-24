import torch
import torch.nn.functional as F
from torch import nn

from src.layers import encoder
from src.model.base import BaseUpsampler
from src.model.featup import JBULearnedRange


class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=256, kernel_size=1, ks_res=1, groups=8):
        super().__init__()
        self.out_channels = dim
        self.encoder = encoder(in_channels, dim // 2, kernel_size, ks_res, num_groups=groups)
        self.sem_encoder = encoder(in_channels, dim // 2, 3, 3, num_groups=groups)

    def forward(self, img, output_size):
        sem_img = None

        pixel_img = self.encoder(img)
        sem_img = self.sem_encoder(img)
        x = torch.cat([pixel_img, sem_img], dim=1)
        x = F.adaptive_avg_pool2d(x, output_size=output_size)
        return x


class JBU(BaseUpsampler):
    def __init__(self, dim=256, radius=5, groups=8, name="jbu", combine=False, *args, **kwargs):
        super().__init__()
        self.name = name
        self.radius = radius
        guidance_dim = 3

        self.bilateral_filter = JBULearnedRange(guidance_dim=guidance_dim, key_dim=dim // 4, radius=radius, combine=combine)

    def forward(self, noisy_imgs_norm, noisy_imgs, output_size, *args, **kwargs):
        guidance_image = F.interpolate(noisy_imgs_norm, size=output_size, mode="bilinear", align_corners=False)
        noisy_imgs = F.interpolate(noisy_imgs, size=output_size, mode="bilinear", align_corners=False)

        filtered_features = self.bilateral_filter(noisy_imgs, guidance_image)
        return filtered_features
