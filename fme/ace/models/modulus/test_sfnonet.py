import os

import torch

from fme.core.device import get_device
from fme.core.testing import validate_tensor

from .sfnonet import SphericalFourierNeuralOperatorNet

DIR = os.path.abspath(os.path.dirname(__file__))


def test_sfnonet_output_is_unchanged():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    model = SphericalFourierNeuralOperatorNet(
        params=None,
        embed_dim=16,
        num_layers=2,
        img_shape=img_shape,
        in_chans=input_channels,
        out_chans=output_channels,
    ).to(device)
    # must initialize on CPU to get the same results on GPU
    x = torch.randn(n_samples, input_channels, *img_shape).to(device)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (n_samples, output_channels, *img_shape)
    validate_tensor(
        output,
        os.path.join(DIR, "testdata/test_sfnonet_output_is_unchanged.pt"),
    )
