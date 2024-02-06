import pytest
import torch

from fme.downscaling.modules.registry import ModuleRegistrySelector, SwinirConfig


def test_module_registry_selector_build():
    n_in_channels = 3
    n_out_channels = 3
    lowres_shape = (16, 16)
    upscale_factor = 4

    selector = ModuleRegistrySelector(
        type="swinir",
        config={
            "depths": (6,),
            "embed_dim": 30,
            "num_heads": (6,),
        },
    )

    module = selector.build(
        n_in_channels=n_in_channels,
        n_out_channels=n_out_channels,
        lowres_shape=lowres_shape,
        upscale_factor=upscale_factor,
    )

    assert isinstance(module, torch.nn.Module)


@pytest.mark.parametrize(
    "lowres_shape, upscale_factor, highres_shape",
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
@pytest.mark.parametrize("n_channels", [2, 3, 4])
@pytest.mark.parametrize("window_size", [1, 4])
def test_swinir_output_shapes(
    lowres_shape, upscale_factor, highres_shape, n_channels, window_size
):
    swinir = SwinirConfig(
        depths=(6,),
        window_size=window_size,
        embed_dim=30,
        num_heads=(6,),
    ).build(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        lowres_shape=lowres_shape,
        upscale_factor=upscale_factor,
    )
    batch_size = 2
    inputs = torch.rand(batch_size, n_channels, *lowres_shape)
    outputs = swinir(inputs)
    assert outputs.shape == (batch_size, n_channels, *highres_shape)
