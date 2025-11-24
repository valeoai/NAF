import pytest
import torch
from ptflops import get_model_complexity_info
from test_utils import get_active_factor, print_test_info, setup_parametrization

from utils.wrapper import ModelWrapper


@pytest.fixture(params=["FeatUp", "AnyUp", "JAFAR", "NAF"])
def model_name(request):
    return request.param


def prepare_input(input_params):
    img_size, embed_dim, ratio, lr_size = input_params
    features = torch.randn(1, embed_dim, lr_size, lr_size).cuda()
    output_size = (ratio * lr_size, ratio * lr_size)
    image = torch.randn(1, 3, *output_size).cuda()
    return dict(image=image, features=features, output_size=output_size)


# Main test - optimized parametrization
def pytest_generate_tests(metafunc):
    setup_parametrization(metafunc)


def test_flops(model_name, img_size, embed_dim, ratio, lr_size, request):
    # Determine which factor is being swept
    factor = get_active_factor(request)

    # Create model
    model = ModelWrapper(name=model_name, embed_dim=embed_dim, ratio=ratio).cuda()

    flops, params = get_model_complexity_info(
        model,
        input_res=(img_size, embed_dim, ratio, lr_size),
        input_constructor=prepare_input,
        backend="aten",
        as_strings=False,
        print_per_layer_stat=False,
    )

    gflops = (2 * flops) / 1e9

    # Print results using shared utility
    print_test_info(
        model_name,
        factor,
        embed_dim,
        img_size,
        lr_size,
        ratio,
        save=True,
        **{"GFLOPS": f"{gflops:.2f}", "# Params": f"{params}"},
    )
