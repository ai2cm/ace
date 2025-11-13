import pytest
import torch

from fme.core.device import get_device
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.unets import NonDivisibleShapeError, UNetBlock


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


def test_diffusion_unet_autocast():
    if get_device().type == "mps":
        pytest.skip("MPS does not support bfloat16 autocast.")

    downscale_factor = 2
    coarse_shape = (8, 16)
    fine_shape = (
        coarse_shape[0] * downscale_factor,
        coarse_shape[1] * downscale_factor,
    )

    n_channels = 3

    captured_dtypes = {}
    def hook_fn(module, input, output):
        captured_dtypes[module.__class__.__name__] = output.dtype

    unet = (
        DiffusionModuleRegistrySelector(
            "unet_diffusion_song",
            dict(
                model_channels=4,
                amp_mode=True,
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
    hooks = [
        m.register_forward_hook(hook_fn)
        for m in unet.modules()
        if isinstance(m, UNetBlock)
    ]
    
    batch_size = 2
    # models expect interpolated data, so use fine_shape for inputs
    input = torch.randn(batch_size, n_channels, *fine_shape)
    latent = torch.randn(batch_size, n_channels, *fine_shape)
    noise_level = torch.randn(batch_size, 1, 1, 1)
    _ = unet(latent, input, noise_level)

    for h in hooks:
        h.remove()
    assert any(dtype == torch.bfloat16 for dtype in captured_dtypes.values())