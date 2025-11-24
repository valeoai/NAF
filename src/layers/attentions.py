import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    NATTEN_RECENT = False
    from natten.functional import na2d_av, na2d_qk
except:
    NATTEN_RECENT = True
    from natten import na2d

from src.layers.rope import RoPE


def legacy_attention(q, k, v, kernel_size, dilation, scale=1, return_weights=False):
    q = rearrange(q, "b h w n d -> b n h w d")
    k = rearrange(k, "b h w n d -> b n h w d")
    v = rearrange(v, "b h w n d -> b n h w d")
    attn_scores = na2d_qk(q, k, kernel_size=kernel_size, dilation=dilation)
    attn_scores = attn_scores * scale

    attn_weights = attn_scores.softmax(dim=-1)
    features = na2d_av(attn_weights, v, kernel_size=kernel_size, dilation=dilation)
    features = rearrange(features, "b n h w d -> b h w n d")

    if return_weights:
        return features, attn_scores
    return features


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=(9, 9),
        **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.kernel_size = kernel_size

        self.scale = (dim // num_heads) ** -0.5

    def _resize(self, x, size, dtype):
        x = F.interpolate(x, size=size, mode="nearest-exact")
        x = rearrange(x, "b (n d) h w -> b h w n d", n=self.num_heads)
        return x.to(dtype)

    def forward(self, q, k, v, image=None, return_weights=False, **kwargs):
        hq, wq = q.shape[-2:]
        hk, wk = k.shape[-2:]
        dilation = (hq // hk, wq // wk)
        self.dilation = dilation

        q = rearrange(q, "b (n d) h w -> b h w n d", n=self.num_heads)
        k = self._resize(k, size=(hq, wq), dtype=q.dtype)
        v = self._resize(v, size=(hq, wq), dtype=q.dtype)

        # Use legacy attention pattern
        if return_weights:
            assert not NATTEN_RECENT, "Return weights not supported with recent natten versions"
            out, attn_weights = legacy_attention(q, k, v, self.kernel_size, dilation, scale=self.scale, return_weights=True)
            return rearrange(out, "b h w n d -> b (n d) h w"), attn_weights
        else:
            if NATTEN_RECENT:
                # Use modern na2d attention
                # Note: Modern na2d doesn't support position bias directly
                out = na2d(q, k, v, kernel_size=self.kernel_size, dilation=dilation, stride=1, backend="cutlass-fna")
            else:
                out = legacy_attention(q, k, v, self.kernel_size, dilation, scale=self.scale)
            return rearrange(out, "b h w n d -> b (n d) h w")
