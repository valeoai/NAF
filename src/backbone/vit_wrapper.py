import re
import types
from typing import List, Tuple, Union

import timm
import timm.data
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision import transforms

# We provide a list of timm model names, more are available on their official repo.
MODEL_LIST = [
    # DINO
    "vit_base_patch16_224.dino",
    # DINOv2
    "vit_base_patch14_dinov2.lvd142m",
    # DINOv2-R
    "vit_base_patch14_reg4_dinov2",
    # Franca
    "franca_vitb14",
    # DINOv3-ViT
    "vit_base_patch16_dinov3.lvd1689m",
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_7b_patch16_dinov3.lvd1689m",
    # SigLIP2
    "vit_base_patch16_siglip_512.v2_webli",
    # PE Core
    "vit_pe_core_small_patch16_384.fb",
    # PE Spatial
    "vit_pe_spatial_tiny_patch16_512.fb",
    # RADIO
    "radio_v2.5-b",
    # CAPI
    "capi_vitl14_lvd",
    # MAE
    "vit_large_patch16_224.mae",
]

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class PretrainedViTWrapper(nn.Module):

    def __init__(
        self,
        name,
        norm: bool = True,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        **kwargs,
    ):
        super().__init__()
        # comment out the following line to test the models not in the list
        self.name = name

        load_weights = False
        if "dvt_" == name[:4]:
            load_weights = True
            load_tag = "dvt"
            name = name.replace("dvt_", "")
        if "fit3d_" == name[:6]:
            load_weights = True
            load_tag = "fit3d"
            name = name.replace("fit3d_", "")

        # Set patch size
        try:
            self.patch_size = int(re.search(r"patch(\d+)", name).group(1))
        except:
            self.patch_size = 16
        if "franca" in name or "capi" in name:
            self.patch_size = 14
        if "convnext" in name:
            self.patch_size = 32

        name, self.patch_size

        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.model, self.config = self.create_model(name, **kwargs)
        self.config["ps"] = self.patch_size
        self.embed_dim = self.model.embed_dim
        self.norm = norm

        if load_weights:
            ckpt = torch.load(f"/home/lchambon/workspace/JAFAR/ckpts/{load_tag}_{name}.pth", map_location="cpu")
            if load_tag == "dvt":
                self.load_state_dict(ckpt["model"], strict=True)
            elif load_tag == "fit3d":
                self.model.load_state_dict(ckpt, strict=True)

    def create_model(self, name: str, **kwargs) -> Tuple[VisionTransformer, transforms.Compose]:
        if "radio" in self.name:
            model = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                version=name,
                progress=True,
                skip_validation=True,
            )
            data_config = {
                "mean": torch.tensor([0.0, 0.0, 0.0]),
                "std": torch.tensor([1.0, 1.0, 1.0]),
                "input_size": (3, 512, 512),
            }

        elif "franca" in self.name:
            model = torch.hub.load("valeoai/Franca", name, use_rasa_head=True)
            data_config = {"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD, "input_size": (3, 448, 448)}

        elif "capi" in self.name:
            model = torch.hub.load("facebookresearch/capi:main", name, force_reload=False)

            data_config = {"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD, "input_size": (3, 448, 448)}

        else:
            timm_kwargs = dict(
                pretrained=True,
                num_classes=0,
                patch_size=self.patch_size,
            )

            if "sam" not in self.name and "convnext" not in self.name:
                timm_kwargs["dynamic_img_size"] = self.dynamic_img_size
                timm_kwargs["dynamic_img_pad"] = self.dynamic_img_pad

            timm_kwargs.update(kwargs)
            model = timm.create_model(name, **timm_kwargs)
            data_config = timm.data.resolve_model_data_config(model=model)

        model = model.eval()

        return model, data_config

    def forward(
        self,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
        return_prefix_tokens: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Intermediate layer accessor inspired by DINO / DINOv2 interface.
        Args:
            x: Input tensor.
            n: Take last n blocks if int, all if None, select matching indices if sequence
            reshape: Whether to reshape the output.
        """

        common_kwargs = dict(
            norm=self.norm,
            output_fmt="NCHW",
            intermediates_only=True,
        )

        if "sam" not in self.name and return_prefix_tokens:
            common_kwargs["return_prefix_tokens"] = return_prefix_tokens

        elif "franca" in self.name:
            B, C, H, W = x.shape
            feats = self.model.forward_features(x, use_rasa_head=True)
            out = feats["patch_token_rasa"]
            out = rearrange(out, "b (h w) c -> b c h w", h=H // self.patch_size, w=W // self.patch_size)

        elif "capi" in self.name:
            *_, out = self.model(x)
            out = out.permute(0, 3, 1, 2)

        else:
            out = self.model.forward_intermediates(x, n, **common_kwargs)

        # "sam" models return feats only, others may return (feats, prefix)
        if not isinstance(out, list) and not isinstance(out, tuple):
            out = [out]
            return out[0]
        else:
            assert len(out) == 1, f"Out contains {len(out)} elements, expected 1."
            return out[0]
