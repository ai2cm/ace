import pytest
import torch

from fme.core.device import get_device
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.unets import NonDivisibleShapeError


def test_diffusion_unet_shapes():
    downscale_factor = 2
    coarse_shape = (8, 16)
    fine_shape = (
        coarse_shape[0] * downscale_factor,
        coarse_shape[1] * downscale_factor,
    )

    n_channels = 3

    unet = (
        DiffusionModuleRegistrySelector(
            "unet_diffusion_song",
            dict(
                model_channels=4,
            ),
        )
        .build(
            n_in_channels=n_channels,
            n_out_channels=n_channels,
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            sigma_data=1.0,
        )
        .to(get_device())
    )

    batch_size = 2
    # models expect interpolated data, so use fine_shape for inputs
    input = torch.randn(batch_size, n_channels, *fine_shape)
    latent = torch.randn(batch_size, n_channels, *fine_shape)
    noise_level = torch.randn(batch_size, 1, 1, 1)
    outputs = unet(latent, input, noise_level)

    assert (batch_size, n_channels, *fine_shape) == outputs.shape


def test_diffusion_invalid_shapes():
    downscale_factor = 2
    coarse_shape = (9, 18)
    fine_shape = (
        coarse_shape[0] * downscale_factor,
        coarse_shape[1] * downscale_factor,
    )
    n_channels = 3

    unet = (
        DiffusionModuleRegistrySelector(
            "unet_diffusion_song",
            dict(
                model_channels=4,
                attn_resolutions=[],
            ),
        )
        .build(
            n_in_channels=n_channels,
            n_out_channels=n_channels,
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            sigma_data=1.0,
        )
        .to(get_device())
    )

    batch_size = 2
    # models expect interpolated data, so use fine_shape for inputs
    input = torch.randn(batch_size, n_channels, *fine_shape)
    latent = torch.randn(batch_size, n_channels, *fine_shape)
    noise_level = torch.randn(batch_size, 1, 1, 1)
    with pytest.raises(NonDivisibleShapeError):
        unet(latent, input, noise_level)


def test_UNetDiffusionModule_forward_pass():
    downscale_factor = 2
    coarse_shape = (8, 16)
    fine_shape = coarse_shape[0] * downscale_factor, coarse_shape[1] * downscale_factor
    n_channels = 3

    module = DiffusionModuleRegistrySelector(
        "unet_diffusion_song", {"model_channels": 4}
    ).build(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        sigma_data=1.0,
    )

    batch_size = 1
    conditioning = torch.randn(batch_size, n_channels, *fine_shape)
    latent = torch.randn(batch_size, n_channels, *fine_shape)
    noise = torch.randn(batch_size, 1, 1, 1)

    assert (batch_size, n_channels, *fine_shape) == module(
        latent, conditioning, noise
    ).shape


def test_diffusion_module_has_channels_last_memory_format():
    """Test DiffusionModuleConfig builds models with channels_last memory format."""
    downscale_factor = 2
    coarse_shape = (8, 16)
    n_channels = 3

    module = DiffusionModuleRegistrySelector(
        "unet_diffusion_song", {"model_channels": 4}
    ).build(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        sigma_data=1.0,
    )

    # Check that 4D parameters (conv weights) are in channels_last format
    has_4d_params = False
    for name, param in module.named_parameters():
        if param.ndim == 4:
            has_4d_params = True
            assert param.is_contiguous(
                memory_format=torch.channels_last
            ), f"Parameter {name} is not in channels_last memory format"
    assert has_4d_params, "Model should have 4D parameters (conv weights) to test"


@pytest.mark.parametrize("use_channels_last", [True, False])
def test_diffusion_module_use_channels_last_flag(use_channels_last):
    """Test that use_channels_last flag correctly controls memory format."""
    downscale_factor = 2
    coarse_shape = (8, 16)
    fine_shape = coarse_shape[0] * downscale_factor, coarse_shape[1] * downscale_factor
    n_channels = 3

    module = DiffusionModuleRegistrySelector(
        "unet_diffusion_song",
        {"model_channels": 4},
        use_channels_last=use_channels_last,
    ).build(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        sigma_data=1.0,
    )

    # Check that 4D parameters match expected memory format
    for name, param in module.named_parameters():
        if param.ndim == 4:
            is_channels_last = param.is_contiguous(memory_format=torch.channels_last)
            if use_channels_last:
                assert (
                    is_channels_last
                ), f"Parameter {name} should be channels_last when flag is True"
            else:
                # When use_channels_last=False, params should be contiguous (NCHW)
                assert (
                    param.is_contiguous()
                ), f"Parameter {name} should be contiguous when flag is False"

    # Test forward pass works with both settings
    batch_size = 1
    conditioning = torch.randn(batch_size, n_channels, *fine_shape)
    latent = torch.randn(batch_size, n_channels, *fine_shape)
    noise = torch.randn(batch_size, 1, 1, 1)

    output = module(latent, conditioning, noise)
    assert output.shape == (batch_size, n_channels, *fine_shape)
