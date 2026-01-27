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


@pytest.mark.parametrize(
    "use_amp_bf16",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="Autocast with bfloat16 requires CUDA",
            ),
        ),
        False,
    ],
)
def test_UNetDiffusionModule_use_amp_precision(use_amp_bf16):
    """Test that use_amp_bf16 parameter correctly sets precision bfloat16 or float32."""
    downscale_factor = 2
    coarse_shape = (8, 16)
    fine_shape = coarse_shape[0] * downscale_factor, coarse_shape[1] * downscale_factor
    n_channels = 3

    # Build module with specified use_amp_bf16 value
    module = DiffusionModuleRegistrySelector(
        "unet_diffusion_song", {"model_channels": 4}
    ).build(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        sigma_data=1.0,
        use_amp_bf16=use_amp_bf16,
    )

    batch_size = 1
    conditioning = torch.randn(batch_size, n_channels, *fine_shape, dtype=torch.float32)
    latent = torch.randn(batch_size, n_channels, *fine_shape, dtype=torch.float32)
    noise = torch.randn(batch_size, 1, 1, 1, dtype=torch.float32)

    captured_dtypes = []

    def forward_hook(module, input, output):
        if len(input) > 0 and isinstance(input[0], torch.Tensor):
            captured_dtypes.append(input[0].dtype)
        if isinstance(output, torch.Tensor):
            captured_dtypes.append(output.dtype)

    # Register hook on the SongUNet model inside EDMPrecond
    edm_precond = module.unet
    edm_precond.model.register_forward_hook(forward_hook)

    module(latent, conditioning, noise)

    assert len(captured_dtypes) > 0, "No dtypes captured from forward hook"
    if use_amp_bf16:
        assert torch.bfloat16 in captured_dtypes, (
            "Autocast to bfloat16 was set but did not find bfloat16 in captured "
            f"dtypes {captured_dtypes}."
        )
    else:
        assert all(
            dtype == torch.float32 for dtype in captured_dtypes
        ), "Expected all dtypes to be float32 when use_amp_bf16=False, "
        f"but got captured dtypes {captured_dtypes}."
        assert torch.bfloat16 not in captured_dtypes, (
            "Expected no bfloat16 when use_amp_bf16=False, "
            f"but found bfloat16 in captured dtypes {captured_dtypes}."
        )
