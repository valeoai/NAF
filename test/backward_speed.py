import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from test_utils import create_tensors, get_active_factor, print_test_info, setup_parametrization

from utils.wrapper import ModelWrapper

NUM_RUNS = 10


@pytest.fixture(params=["FeatUp", "AnyUp", "JAFAR", "NAF"])
def model_name(request):
    return request.param


# Main test - optimized parametrization
def pytest_generate_tests(metafunc):
    setup_parametrization(metafunc)


def test_backward_speed(model_name, img_size, embed_dim, ratio, lr_size, request):
    # Determine which factor is being swept
    factor = get_active_factor(request)

    # Create model
    model = ModelWrapper(name=model_name, embed_dim=embed_dim, ratio=ratio).cuda()

    # Create input tensors
    img, lr_feats, output_size = create_tensors(img_size, embed_dim, ratio, lr_size)

    # Additional components for backward pass
    linear = nn.Conv2d(embed_dim, 1, 1).cuda()
    optimizer = optim.SGD(list(model.parameters()) + list(linear.parameters()), lr=0.01)

    # Warmup runs
    for _ in range(5):
        torch.cuda.empty_cache()
        output = model(img, lr_feats, output_size)
        loss = linear(output).sum()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

    total_time = 0

    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(NUM_RUNS):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all previous operations are complete
        start_event.record()
        output = model(img, lr_feats, output_size)
        loss = linear(output).sum()
        loss.backward()
        optimizer.step()
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to complete
        torch.cuda.empty_cache()
        total_time += start_event.elapsed_time(end_event)  # Time in milliseconds

    avg_time = total_time / NUM_RUNS

    # Print results using shared utility
    print_test_info(
        model_name, factor, embed_dim, img_size, lr_size, ratio, **{"Average backward pass time": f"{avg_time:.6f} ms"}
    )
