"""
Shared utilities for test files to eliminate code duplication.
"""

import json
import sys
from pathlib import Path

import torch

# Add the project root to path dynamically
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Default values
DEFAULT_IMG_SIZE = 448
DEFAULT_EMBED_DIM = 384
DEFAULT_RATIO = 16
DEFAULT_LR_SIZE = DEFAULT_IMG_SIZE // DEFAULT_RATIO

# Test configuration constants
IMG_SIZES = [112, 224, 448, 896]
EMBED_DIMS = [128, 384, 768, 1024]
RATIOS = [2, 4, 8, 16, 32]
LR_SIZES = [32]


def setup_parametrization(metafunc):
    """
    Optimized pytest parametrization setup that eliminates repetitive if statements.
    """
    # Detect selected factor set
    sweep_options = {
        "embed_dim": (metafunc.config.getoption("--embed-dim"), EMBED_DIMS),
        "img_size": (metafunc.config.getoption("--img-size"), IMG_SIZES),
        "ratio": (metafunc.config.getoption("--ratio"), RATIOS),
        "lr_size": (metafunc.config.getoption("--lr-size"), LR_SIZES),
    }

    # Check only one sweep option is selected
    active_sweeps = [name for name, (enabled, _) in sweep_options.items() if enabled]
    if len(active_sweeps) > 1:
        raise ValueError("Only one fixture can be swept at a time")

    # Default parameter values
    defaults = {
        "embed_dim": DEFAULT_EMBED_DIM,
        "img_size": DEFAULT_IMG_SIZE,
        "ratio": DEFAULT_RATIO,
        "lr_size": DEFAULT_LR_SIZE,
    }

    # Set up parametrization
    if active_sweeps:
        sweep_param = active_sweeps[0]
        for param_name, default_value in defaults.items():
            if param_name == sweep_param:
                # Use the sweep values for the active parameter
                metafunc.parametrize(param_name, sweep_options[param_name][1])
            else:
                # Use default value for other parameters
                metafunc.parametrize(param_name, [default_value])
    else:
        # Use all default values when no sweep is active
        for param_name, default_value in defaults.items():
            metafunc.parametrize(param_name, [default_value])


def get_active_factor(request):
    """
    Determine which factor is being swept based on request config.
    """
    sweep_options = ["embed_dim", "img_size", "ratio", "lr_size"]
    active_sweeps = [opt for opt in sweep_options if request.config.getoption(f"--{opt.replace('_', '-')}")]
    return active_sweeps[0] if active_sweeps else "none (all defaults)"


def create_tensors(img_size, embed_dim, ratio, lr_size, device="cuda"):
    lr_feats = torch.randn(1, embed_dim, lr_size, lr_size, device=device)
    output_size = (ratio * lr_size, ratio * lr_size)
    img = torch.randn(1, 3, img_size, img_size, device=device)
    return img, lr_feats, output_size


def print_test_info(model_name, factor, embed_dim, img_size, lr_size, ratio, save=True, **extra_info):
    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print(f"Factor being swept: {factor}")
    print(f"Embed size: {embed_dim}")
    print(f"Image size: {img_size}")
    print(f"LR size: {lr_size}")
    print(f"Upsampling factor (ratio): {ratio}")
    print("-" * 60)

    for key, value in extra_info.items():
        print(f"{key}: {value}")

    print("=" * 60 + "\n")

    # Save results to JSON
    if save:
        save_test_results(model_name, factor, embed_dim, img_size, lr_size, ratio, **extra_info)


def save_test_results(model_name, factor, embed_dim, img_size, lr_size, ratio, **extra_info):
    """Save test results to a JSON file for later analysis."""
    # Create results directory if it doesn't exist
    results_dir = Path("./test")
    results_dir.mkdir(exist_ok=True)

    # Load existing results or create new list
    results_file = results_dir / "test_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Check if entry with same configuration already exists
    existing_entry = None
    for entry in results:
        if (
            entry["model"] == model_name
            and entry["factor_swept"] == factor
            and entry["embed_dim"] == embed_dim
            and entry["img_size"] == img_size
            and entry["lr_size"] == lr_size
            and entry["ratio"] == ratio
        ):
            existing_entry = entry
            break

    if existing_entry:
        # Merge metrics into existing entry
        existing_entry["metrics"].update(extra_info)
        print(f"Merged metrics into existing entry for {model_name} (ratio={ratio})")
    else:
        # Create new entry
        result_entry = {
            "model": model_name,
            "factor_swept": factor,
            "embed_dim": embed_dim,
            "img_size": img_size,
            "lr_size": lr_size,
            "ratio": ratio,
            "metrics": extra_info,
        }
        results.append(result_entry)
        print(f"Created new entry for {model_name} (ratio={ratio})")

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")
