dependencies = ["torch", "natten"]

import torch

from src.model.naf import NAF


def naf(pretrained: bool = True, device="cpu"):
    """
    NAF (Neighborhood Attention Filtering) model for feature upsampling.
    VFM-agnostic upsampler that works with any Vision Foundation Model without retraining.

    Args:
        pretrained (bool): If True, loads pretrained weights
        device (str): Device to load the model on ('cpu', 'cuda', etc.)

    Returns:
        NAF model instance

    Dependencies:
        - torch: PyTorch framework
        - natten: Neighborhood Attention Extension (required for cross-scale attention)

    Note: If you want to use visualization features, you may also need to install matplotlib and plotly.

    Installation:
        pip install natten -f https://shi-labs.com/natten/wheels
    """
    model = NAF().to(device)
    if pretrained:
        checkpoint = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location=device))
    return model
