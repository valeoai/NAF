import pytest
import torch
from test_utils import create_tensors, get_active_factor, print_test_info, setup_parametrization

from utils.wrapper import ModelWrapper


@pytest.fixture(params=["FeatUp", "AnyUp", "JAFAR", "NAF"])
def model_name(request):
    return request.param


# Main test - optimized parametrization
def pytest_generate_tests(metafunc):
    setup_parametrization(metafunc)


def test_gpu_memory_usage_forward(model_name, img_size, embed_dim, ratio, lr_size, request):
    # Determine which factor is being swept
    factor = get_active_factor(request)

    # Create model
    model = ModelWrapper(name=model_name, embed_dim=embed_dim, ratio=ratio).cuda()

    # Create input tensors
    img, lr_feats, output_size = create_tensors(img_size, embed_dim, ratio, lr_size)

    # Measure memory usage
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = model(img, lr_feats, output_size)
    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB

    # Print results using shared utility
    print_test_info(
        model_name, factor, embed_dim, img_size, lr_size, ratio, **{"Peak GPU memory usage (forward)": f"{peak_memory:.2f} MB"}
    )
