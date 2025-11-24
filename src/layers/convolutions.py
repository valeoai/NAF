import torch.nn as nn
import torch.nn.functional as F


# Convolutions
class EncBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        norm_kwargs={},
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        bias=True,
        residual=False,
    ):
        super().__init__()
        self.use_conv_shortcut = use_conv_shortcut
        self.norm1 = norm_fn(**norm_kwargs)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=bias,
        )
        self.norm2 = norm_fn(**norm_kwargs)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=bias,
        )
        self.activation_fn = activation_fn()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                padding_mode=pad_mode,
                bias=bias,
            )
        self.residual = residual

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.use_conv_shortcut or residual.shape != x.shape:
            residual = self.shortcut(residual)
        if self.residual:
            return x + residual
        return x


def encoder(in_dim, hidden_dim, kernel_size=1, ks_res=1, num_layers=2, bias=True, num_groups=8, residual=False):
    return nn.Sequential(
        nn.Conv2d(
            in_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
            bias=bias,
        ),
        *[
            EncBlock(
                hidden_dim,
                hidden_dim,
                kernel_size=ks_res,
                pad_mode="reflect",
                norm_fn=nn.GroupNorm,
                norm_kwargs={"num_groups": num_groups, "num_channels": hidden_dim},
                activation_fn=nn.SiLU,
                use_conv_shortcut=False,
                bias=bias,
                residual=residual,
            )
            for _ in range(num_layers)
        ],
    )
