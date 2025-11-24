import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from test_utils import create_tensors, get_active_factor, print_test_info, setup_parametrization

from utils.wrapper import ModelWrapper


@pytest.fixture(params=["FeatUp", "AnyUp", "JAFAR", "NAF"])
def model_name(request):
    return request.param


# Main test - optimized parametrization
def pytest_generate_tests(metafunc):
    setup_parametrization(metafunc)


def test_gpu_memory_usage_backward(model_name, img_size, embed_dim, ratio, lr_size, request):
    torch.cuda.empty_cache()

    # Determine which factor is being swept
    factor = get_active_factor(request)

    # Create model
    model = ModelWrapper(name=model_name, embed_dim=embed_dim, ratio=ratio).cuda()

    # Create input tensors
    img, lr_feats, output_size = create_tensors(img_size, embed_dim, ratio, lr_size)

    # Additional components for backward pass
    linear = nn.Conv2d(embed_dim, 1, 1).cuda()
    optimizer = optim.SGD(list(model.parameters()) + list(linear.parameters()), lr=0.01)

    # Measure GPU memory usage during backward pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = model(img, lr_feats, output_size)
    loss = linear(output).sum()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB

    # Print results using shared utility
    print_test_info(
        model_name,
        factor,
        embed_dim,
        img_size,
        lr_size,
        ratio,
        **{"Peak GPU memory usage (backward)": f"{peak_memory:.2f} MB"},
    )
