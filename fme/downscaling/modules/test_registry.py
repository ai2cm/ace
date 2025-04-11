import pytest
import torch

from fme.core.device import get_device
from fme.downscaling.modules.registry import (
    InterpolateConfig,
    ModuleRegistrySelector,
    SwinirConfig,
)


@pytest.mark.parametrize(
    "coarse_shape, downscale_factor, fine_shape",
    [
        pytest.param(
            (45, 90),
            4,
            (180, 360),
            id="45x90_to_180x360",
        ),
        pytest.param(
            (4, 4),
            4,
            (16, 16),
            id="4x4_to_16x16",
        ),
        pytest.param(
            (8, 8),
            2,
            (16, 16),
            id="8x8_to_16x16",
        ),
    ],
)
@pytest.mark.parametrize("n_in_channels", [1, 2, 3])
@pytest.mark.parametrize("n_out_channels", [1, 2])
@pytest.mark.parametrize("window_size", [1, 4])
def test_swinir_output_shapes(
    coarse_shape,
    downscale_factor,
    fine_shape,
    n_in_channels,
    n_out_channels,
    window_size,
):
    swinir = SwinirConfig(
        depths=(6,),
        window_size=window_size,
        embed_dim=30,
        num_heads=(6,),
    ).build(
        n_in_channels=n_in_channels,
        n_out_channels=n_out_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
    )
    batch_size = 2
    inputs = torch.rand(batch_size, n_in_channels, *coarse_shape)
    outputs = swinir(inputs)
    assert outputs.shape == (batch_size, n_out_channels, *fine_shape)


@pytest.mark.parametrize(
    "coarse_shape, downscale_factor, fine_shape",
    [
        pytest.param(
            (4, 4),
            4,
            (16, 16),
            id="4x4_to_16x16",
        ),
    ],
)
@pytest.mark.parametrize("n_in_channels", [1, 2, 3])
@pytest.mark.parametrize("n_out_channels", [1, 2])
@pytest.mark.parametrize("window_size", [4])
@pytest.mark.parametrize(
    "upsampler",
    ["pixelshuffle", "pixelshuffledirect", "nearest+conv"],
)
def test_swinir_downscaling_options(
    coarse_shape,
    downscale_factor,
    fine_shape,
    n_in_channels,
    n_out_channels,
    window_size,
    upsampler,
):
    """Tests the various upsampling options for SwinIR and checks that they have
    the expected output shapes."""
    swinir = SwinirConfig(
        depths=(6,),
        window_size=window_size,
        embed_dim=30,
        num_heads=(6,),
        upsampler=upsampler,
    ).build(
        n_in_channels=n_in_channels,
        n_out_channels=n_out_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
    )
    batch_size = 2
    inputs = torch.rand(batch_size, n_in_channels, *coarse_shape)
    outputs = swinir(inputs)
    assert outputs.shape == (batch_size, n_out_channels, *fine_shape)


@pytest.mark.parametrize(
    "coarse_shape, downscale_factor, fine_shape",
    [
        pytest.param(
            (4, 4),
            4,
            (16, 16),
            id="4x4_to_16x16",
        ),
    ],
)
@pytest.mark.parametrize("n_channels, expected_sum", [(1, 14.1226), (2, 7.5227)])
@pytest.mark.parametrize("window_size", [4])
@pytest.mark.parametrize("upsampler", ["pixelshuffle"])
def test_swinir_values(
    coarse_shape,
    downscale_factor,
    fine_shape,
    n_channels,
    window_size,
    upsampler,
    expected_sum,
):
    """Regression test to check that the output values match the original
    implementation."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    swinir = SwinirConfig(
        depths=(6,),
        window_size=window_size,
        embed_dim=30,
        num_heads=(6,),
        upsampler=upsampler,
    ).build(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
    )
    batch_size = 2
    inputs = torch.rand(batch_size, n_channels, *coarse_shape)
    outputs = swinir(inputs)
    assert outputs.shape == (batch_size, n_channels, *fine_shape)
    torch.testing.assert_close(
        float(torch.sum(outputs)), expected_sum, atol=1e-2, rtol=1e-3
    )


@pytest.mark.parametrize("mode", ["bicubic", "nearest"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_in_channels, n_out_channels", [(1, 1), (2, 1)])
@pytest.mark.parametrize("downscale_factor", [1, 2, 4])
def test_interpolate(
    mode,
    batch_size,
    n_in_channels,
    n_out_channels,
    downscale_factor,
    coarse_shape=(4, 8),
):
    config = InterpolateConfig(mode)
    interpolate = config.build(
        n_in_channels, n_out_channels, coarse_shape, downscale_factor
    )
    inputs = torch.rand(batch_size, n_in_channels, *coarse_shape)
    outputs = interpolate(inputs)
    fine_shape = tuple(s * downscale_factor for s in coarse_shape)
    # Note: interpolate models ignore `n_out_channels`
    assert outputs.shape == (batch_size, n_in_channels, *fine_shape)


@pytest.mark.parametrize("type_", ["unet_regression_song", "unet_regression_dhariwal"])
def test_unets_output_shape(type_):
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    downscale_factor = 2
    unet = (
        ModuleRegistrySelector(
            type_,
            dict(
                model_channels=4,
                attn_resolutions=[],
            ),
        )
        .build(
            n_in_channels=3,
            n_out_channels=3,
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
        )
        .to(get_device())
    )
    inputs = torch.rand(2, 3, *fine_shape).to(get_device())
    outputs = unet(inputs)
    assert outputs.shape == (2, 3, *fine_shape)


@pytest.mark.parametrize("type_", ["unet_regression_song", "unet_regression_dhariwal"])
@pytest.mark.parametrize(
    "channel_mult,attn_resolution",
    [
        pytest.param([1, 2, 3, 4, 5], [], id="too_may_levels"),
        pytest.param([1, 2], [4, 1], id="missing_requested_attn"),
    ],
)
def test_unets_invalid_config(type_, channel_mult, attn_resolution):
    # determined by min coarse dimension
    coarse_shape = (8, 16)
    downscale_factor = 1
    with pytest.raises(ValueError):
        ModuleRegistrySelector(
            type_,
            dict(
                model_channels=4,
                attn_resolutions=attn_resolution,
                channel_mult=channel_mult,
            ),
        ).build(
            n_in_channels=3,
            n_out_channels=3,
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
        )
