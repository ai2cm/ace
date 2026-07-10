import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.testing import validate_tensor

DIR = os.path.abspath(os.path.dirname(__file__))

from fme.ace.models.ocean.m2lines.samudra import Samudra


@pytest.mark.parametrize("norm", ["batch", "layer", "instance", None, "group"])
def test_samudra_normalization(norm):
    # Model parameters
    input_channels = 4
    output_channels = 3
    batch_size = 2
    height = 64
    width = 64

    # Initialize model
    if norm == "group":
        with pytest.raises(NotImplementedError):
            model = Samudra(
                input_channels=input_channels,
                output_channels=output_channels,
                ch_width=[32, 64],
                dilation=[1, 2],
                n_layers=[1, 1],
                norm=norm,
            )
        return

    model = Samudra(
        input_channels=input_channels,
        output_channels=output_channels,
        ch_width=[32, 64],
        dilation=[1, 2],
        n_layers=[1, 1],
        norm=norm,
    )

    # Create dummy input
    x = torch.randn(batch_size, input_channels, height, width)

    # Forward pass
    output = model(x)

    # Check output shape
    expected_shape = (batch_size, output_channels, height, width)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"

    # Check output values
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_samudra_norm_kwargs():
    model = Samudra(
        input_channels=4,
        output_channels=3,
        ch_width=[32, 64],
        dilation=[1, 2],
        n_layers=[1, 1],
        norm="batch",
        norm_kwargs={"track_running_stats": False},
    )
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert not module.track_running_stats


@pytest.mark.parametrize("n_vector_outputs", [1, 12])
def test_samudra_vector_readout_head(n_vector_outputs):
    torch.manual_seed(0)
    input_channels = 4
    output_channels = 20
    batch_size = 2
    height, width = 16, 32
    model = Samudra(
        input_channels=input_channels,
        output_channels=output_channels,
        ch_width=[8, 16],
        dilation=[1, 2],
        n_layers=[1, 1],
        norm="instance",
        n_vector_outputs=n_vector_outputs,
    )
    x = torch.randn(batch_size, input_channels, height, width)
    out = model(x)

    assert out.shape == (batch_size, output_channels, height, width)
    assert not torch.isnan(out).any()

    # The last n_vector_outputs channels come from the MLP readout and are
    # broadcast across space, so they are spatially homogeneous by construction.
    vector_channels = out[:, output_channels - n_vector_outputs :]
    spatial_std = vector_channels.reshape(batch_size, n_vector_outputs, -1).std(dim=-1)
    torch.testing.assert_close(
        spatial_std, torch.zeros_like(spatial_std), atol=1e-5, rtol=0
    )
    # Any pixel recovers the same per-sample vector value.
    torch.testing.assert_close(vector_channels[..., 0, 0], vector_channels[..., -1, -1])


def test_samudra_vector_readout_requires_field_outputs():
    with pytest.raises(ValueError):
        Samudra(
            input_channels=4,
            output_channels=3,
            ch_width=[8, 16],
            dilation=[1, 2],
            n_layers=[1, 1],
            n_vector_outputs=3,  # must be < output_channels
        )


def test_samudra_output_is_unchanged():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()
    model = Samudra(
        input_channels=input_channels,
        output_channels=output_channels,
        ch_width=[3, 3],
        dilation=[1, 2],
        n_layers=[1, 1],
        norm="batch",
    ).to(device)
    # must initialize on CPU to get the same results on GPU
    x = torch.randn(n_samples, input_channels, *img_shape).to(device)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (n_samples, output_channels, *img_shape)
    validate_tensor(
        output,
        os.path.join(DIR, "testdata/test_samudra_output_is_unchanged.pt"),
    )
