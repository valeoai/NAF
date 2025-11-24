import pytest
import torch
from test_utils import create_tensors, get_active_factor, print_test_info, setup_parametrization

from utils.wrapper import ModelWrapper

NUM_RUNS = 10


@pytest.fixture(params=["FeatUp", "AnyUp", "JAFAR", "NAF"])
def model_name(request):
    return request.param


# Main test - optimized parametrization
def pytest_generate_tests(metafunc):
    setup_parametrization(metafunc)


def test_forward_speed(model_name, img_size, embed_dim, ratio, lr_size, request):
    # Determine which factor is being swept
    factor = get_active_factor(request)

    # Create model
    model = ModelWrapper(name=model_name, embed_dim=embed_dim, ratio=ratio).cuda()

    # Create input tensors
    img, lr_feats, output_size = create_tensors(img_size, embed_dim, ratio, lr_size)

    # Warmup runs
    for _ in range(5):
        with torch.no_grad():
            torch.cuda.empty_cache()
            _ = model(img, lr_feats, output_size)

    total_time = 0

    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(NUM_RUNS):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all previous operations are complete
        start_event.record()
        with torch.no_grad():
            _ = model(img, lr_feats, output_size)
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to complete
        total_time += start_event.elapsed_time(end_event)  # Time in milliseconds

    avg_time = total_time / NUM_RUNS

    # Print results using shared utility
    print_test_info(
        model_name,
        factor,
        embed_dim,
        img_size,
        lr_size,
        ratio,
        save=True,
        **{"Average forward pass time": f"{avg_time:.6f} ms"},
    )
