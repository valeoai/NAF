try:
    import kornia
except ImportError:
    print("Kornia is not installed. Please install it to use the JBF upsampler.")
import torch
import torch.nn.functional as F

from src.model.base import BaseUpsampler


class JBF(BaseUpsampler):
    def __init__(self, kernel_size: int = 5, sigma_color: float = 0.1, sigma_spatial: float = 1.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_spatial = sigma_spatial

    def forward(self, noisy_img_norm, noisy_img, output_size, *args, **kwargs):

        noisy_img = F.interpolate(noisy_img, scale_factor=4, mode="bilinear", align_corners=False)
        noisy_img_norm = F.interpolate(noisy_img_norm, size=noisy_img.shape[-2:], mode="bilinear", align_corners=False)
        output_features = kornia.filters.joint_bilateral_blur(
            input=noisy_img,
            guidance=noisy_img_norm,
            kernel_size=(self.kernel_size, self.kernel_size),
            sigma_color=self.sigma_color,
            sigma_space=(self.sigma_spatial, self.sigma_spatial),
            border_type="reflect",
        )
        output_features = F.interpolate(output_features, size=output_size, mode="bilinear", align_corners=False)

        return output_features
