import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.layers import CrossAttention, RoPE, encoder


class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        heads_rope=1,
        use_encoder=True,
        rope_base=None,
        rope_rescale=None,
        img_layers=2,
    ):
        super().__init__()
        self.use_encoder = use_encoder
        self.out_channels = out_channels

        self.encoder = encoder(in_channels, out_channels // 2, kernel_size=1, ks_res=1, num_layers=img_layers)
        self.sem_encoder = encoder(in_channels, out_channels // 2, kernel_size=3, ks_res=3, num_layers=img_layers)

        self.rope = RoPE(embed_dim=out_channels, num_heads=heads_rope, base=rope_base, rescale_coords=rope_rescale)

    def forward_encoder(self, x, output_size):
        if self.use_encoder:
            x = torch.cat([self.encoder(x), self.sem_encoder(x)], dim=1)
        x = F.adaptive_avg_pool2d(x, output_size=output_size)
        return x

    def forward(self, x, output_size):
        o_size = output_size
        if x.shape[-2] > 4 * o_size[0] or x.shape[-1] > 4 * o_size[1]:
            x = F.interpolate(
                x,
                size=(
                    min(x.shape[-2], 4 * o_size[0], 4 * o_size[1]),
                    min(x.shape[-1], 4 * o_size[1], 4 * o_size[0]),
                ),
                mode="bilinear",
                align_corners=False,
            )

        x = self.forward_encoder(x, o_size)
        x = self.rope(x)
        return x


class QueryEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        x = F.adaptive_avg_pool2d(x, output_size=features.shape[-2:])
        return x


class NAF(nn.Module):
    def __init__(
        self,
        dim=256,
        heads_attn=4,
        heads_rope=4,
        kernel_size=9,
        # ImageEncoder options
        use_encoder=True,
        rope_base=100.0,
        rope_rescale=2.0,
        img_layers=2,
        **kwargs,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            in_channels=3,
            out_channels=dim,
            heads_rope=heads_rope,
            use_encoder=use_encoder,
            rope_base=rope_base,
            img_layers=img_layers,
            rope_rescale=rope_rescale,
        )

        self.query_encoder = QueryEncoder()

        self.key_encoder = KeyEncoder()

        self.upsampler = CrossAttention(dim=dim, num_heads=heads_attn, kernel_size=(kernel_size, kernel_size))

    def forward(self, image, features, output_size, return_weights=False, *args, **kwargs):
        x = self.image_encoder(image, output_size=output_size)

        queries = self.query_encoder(x)
        keys = self.key_encoder(x, features)
        values = features

        if return_weights:
            out, attn_weights = self.upsampler(queries, keys, values, image, return_weights=True)
            return out, attn_weights
        else:
            out = self.upsampler(queries, keys, values, image)
            return out
