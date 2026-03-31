import pytest
import torch

from fme.core.device import get_device
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.physicsnemo_unets_v2.group_norm import apex_available
from fme.downscaling.modules.utils import NonDivisibleShapeError


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
                get_device().type == "mps",
                reason="MPS does not support bfloat16 autocast.",
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
        assert all(dtype == torch.float32 for dtype in captured_dtypes), (
            "Expected all dtypes to be float32 when use_amp_bf16=False, "
            f"but got captured dtypes {captured_dtypes}."
        )
        assert torch.bfloat16 not in captured_dtypes, (
            "Expected no bfloat16 when use_amp_bf16=False, "
            f"but found bfloat16 in captured dtypes {captured_dtypes}."
        )


@pytest.mark.parametrize(
    "use_apex_gn",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(not apex_available(), reason="Apex not available"),
        ),
        False,
    ],
)
def test_songunetv2_channels_last_if_using_apex_gn(use_apex_gn):
    """Test that use_apex_gn flag correctly controls memory format."""
    downscale_factor = 2
    coarse_shape = (8, 16)
    fine_shape = coarse_shape[0] * downscale_factor, coarse_shape[1] * downscale_factor
    n_channels = 3

    module = DiffusionModuleRegistrySelector(
        "unet_diffusion_song_v2",
        {
            "model_channels": 4,
            "use_apex_gn": use_apex_gn,
            "attn_resolutions": [],
        },
    ).build(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        sigma_data=1.0,
    )

    # Check that model parameters match expected memory format
    for name, param in module.named_parameters():
        if param.ndim == 4:
            is_channels_last = param.is_contiguous(memory_format=torch.channels_last)
            if use_apex_gn:
                assert (
                    is_channels_last
                ), f"Parameter {name} should be channels_last when flag is True"
            else:
                # When use_apex_gn=False, params should be contiguous (NCHW)
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

    if use_apex_gn:
        assert output.is_contiguous(memory_format=torch.channels_last)
    else:
        assert output.is_contiguous()
