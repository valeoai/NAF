"""
https://github.com/yjn870/REDNet-pytorch/blob/master/model.py
"""

import math

import torch.nn as nn
import torch.nn.functional as F


class REDNet(nn.Module):
    def __init__(self, input_dim=3, num_layers=15, num_features=64, *args, **kwargs):
        super(REDNet, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(
            nn.Sequential(nn.Conv2d(input_dim, num_features, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        )
        for i in range(num_layers - 1):
            conv_layers.append(
                nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(inplace=True))
            )

        for i in range(num_layers - 1):
            deconv_layers.append(
                nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(inplace=True))
            )
        deconv_layers.append(nn.ConvTranspose2d(num_features, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, noisy_imgs_norm, noisy_imgs, output_size=None):
        noisy_imgs = F.interpolate(noisy_imgs, size=output_size, mode="bilinear", align_corners=False)

        residual = noisy_imgs.clone()
        x = noisy_imgs

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        noise = x
        return residual - noise
