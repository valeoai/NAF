import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.layers import encoder

try:
    from utils.img import create_coordinate
except:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super().__init__()
        self.norm_q = nn.RMSNorm(query_dim)
        self.norm_k = nn.RMSNorm(key_dim)
        self.norm_v = nn.RMSNorm(value_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            vdim=value_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, query, key, value):
        query = self.norm_q(query)
        key = self.norm_k(key)

        _, attn_scores = self.attention(query, key, self.norm_v(value), average_attn_weights=True)
        attn_output = einsum("b i j, b j d -> b i d", attn_scores, value)

        return attn_output, attn_scores


class CrossAttentionBlock(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, num_heads, **kwargs):
        super().__init__()

        self.cross_attn = CrossAttention(
            query_dim,
            key_dim,
            value_dim,
            num_heads,
        )
        self.conv2d = nn.Conv2d(query_dim, query_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, **kwargs):
        q = self.conv2d(q)
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b (h w) c")
        v = rearrange(v, "b c h w -> b (h w) c")
        features, _ = self.cross_attn(q, k, v)

        return features


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self):
        # Create freqs in 1d
        freqs_1d = self.theta ** torch.linspace(0, -1, self.dim // 4)
        # duplicate freqs for rotation pairs of channels
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        # First half of channels do x, second half do y
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        # it's an angular freq here
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


class SFT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.gamma = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.beta = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=False)

    def forward(self, image, features):
        gamma = self.gamma(features)
        beta = self.beta(features)
        return gamma * self.norm(image) + beta  # Spatial modulation


class JAFAR(nn.Module):

    def __init__(
        self,
        dim=128,
        v_dim=384,
        feature_dim=None,
        kernel_size=1,
        num_heads=4,
        **kwargs,
    ):
        super().__init__()

        # Image encoder uses kernel_size=3 for spatial context
        self.image_encoder = encoder(3, dim, kernel_size=kernel_size, bias=False, residual=True)
        self.key_encoder = encoder(dim, dim, kernel_size=1, bias=False, residual=True)
        self.query_encoder = encoder(dim, dim, kernel_size=1, bias=False, residual=True)

        # Create Query features encoder
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim, affine=False)

        # Create Key features encoder
        self.key_features_encoder = encoder(v_dim, dim, kernel_size=1, bias=False, residual=True)
        self.cross_decode = CrossAttentionBlock(dim, dim, v_dim, num_heads)

        # SFT modulation for keys
        self.sft_key = SFT(dim, dim)

        self.rope = RoPE(dim)
        self.rope._device_weight_init()

    def upsample(self, encoded_image, features, output_size):
        _, _, h, w = features.shape

        # Process Queries
        queries = self.query_encoder(encoded_image)
        queries = F.adaptive_avg_pool2d(queries, output_size=output_size)
        queries = self.norm(queries)

        # Process Keys and Values.
        keys = self.key_encoder(encoded_image)
        keys = F.adaptive_avg_pool2d(keys, output_size=(h, w))
        keys = self.sft_key(keys, self.key_features_encoder(F.normalize(features, dim=1)))

        # Values
        values = features

        # Attention layer
        out = self.cross_decode(queries, keys, values)

        return out

    def forward(self, image, features, output_size, *args, **kwargs):
        # Extract high-level features of image.
        encoded_image = self.image_encoder(image)

        # Apply Positional Encoding
        coords = create_coordinate(encoded_image.shape[-2], encoded_image.shape[-1])
        _, _, h, _ = encoded_image.shape
        encoded_image = rearrange(encoded_image, "b c h w -> b (h w) c")
        encoded_image = self.rope(encoded_image, coords)
        encoded_image = rearrange(encoded_image, "b (h w) c -> b c h w", h=h)

        # Get upsampled feats
        features = self.upsample(encoded_image, features, output_size)
        features = rearrange(features, "b (h w) c -> b c h w", h=output_size[0])
        return features
