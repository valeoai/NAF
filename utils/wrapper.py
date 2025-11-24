import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import *


class ModelWrapper(nn.Module):
    def __init__(self, name, embed_dim=384, ratio=16, ckpt_path: str = None):
        super().__init__()

        self.name = name
        self.embed_dim = embed_dim
        self.ratio = ratio

        self.model = self._load_model()

        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            if name != "FeatUp":
                self.model.load_state_dict(state, strict=False)
            else:
                new_ckpts = {
                    k.replace("model.1.", "norm."): v
                    for k, v in state["state_dict"].items()
                    if "upsampler" in k or "model.1.norm" in k
                }
                self.model.model.load_state_dict(new_ckpts, strict=True)

    def _load_model(self):
        upsampler_map = {
            "AnyUp": lambda: AnyUpsampler(),
            "Bilinear": lambda: Bilinear(),
            "FeatUp": lambda: FeatUp(feature_dim=self.embed_dim, ratio=self.ratio),
            "IRCNN": lambda: IRCNN(),
            "JAFAR": lambda: JAFAR(v_dim=self.embed_dim),
            "JBF": lambda: JBF(),
            "JBU": lambda: JBU(),
            "NAF": lambda: NAF(),
            "Nearest": lambda: Nearest(),
            "REDNet": lambda: REDNet(),
            "Restormer": lambda: Restormer(),
        }

        if self.name not in upsampler_map:
            raise ValueError(f"Unknown upsampler: {self.name}")

        return upsampler_map[self.name]()

    def forward(self, image, features, output_size):
        out = self.model(image, features, output_size)
        return out
