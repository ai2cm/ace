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
