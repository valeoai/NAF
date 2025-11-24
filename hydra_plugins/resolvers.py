import re

from omegaconf import OmegaConf


def get_feature(target: str) -> int:
    """Resolve feature dimensions from backbone name patterns"""
    model_name = target.lower()

    if "vits" in model_name or "small" in model_name:
        return 384
    elif "vitb" in model_name or "base" in model_name or model_name == "radio_v2.5-b":
        return 768
    elif "vitl" in model_name or "large" in model_name or model_name == "radio_v2.5-l":
        return 1024
    elif "tiny" in model_name:
        return 192
    else:
        print(f"Warning: get_feature() - Unsupported backbone: {model_name}. Returning default 0.")
        return 0


def get_patch_size(target: str) -> int:
    """Resolve patch size from backbone name patterns using regex"""
    model_name = target.lower()

    # Return a default value for consistency
    if "franca" in model_name:
        return 14

    # Use regex to find patchXX pattern
    match = re.search(r"patch(\d+)", model_name)
    if match:
        return int(match.group(1))

    # Default to 16 if no patch size found
    return 16


OmegaConf.register_new_resolver("get_feature", lambda name: get_feature(name))
OmegaConf.register_new_resolver("get_patch_size", lambda name: get_patch_size(name))
