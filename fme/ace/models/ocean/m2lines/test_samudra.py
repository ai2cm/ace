import os

import pytest
import torch

from fme.ace.models.ocean.m2lines.layers import (
    BilinearUpsample,
    ConvNeXtBlock,
    NoiseConditioning,
    ZonallyPeriodicBilinearUpsample,
)
from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.testing import validate_tensor

DIR = os.path.abspath(os.path.dirname(__file__))

from fme.ace.models.ocean.m2lines.samudra import Samudra


@pytest.mark.parametrize(
    "img_shape",
    [(8, 16), (9, 18), (180, 360)],
)
def test_zonally_periodic_upsample_matches_bilinear_shape(img_shape):
    x = torch.randn(2, 3, *img_shape)
    periodic = ZonallyPeriodicBilinearUpsample()(x)
    plain = BilinearUpsample()(x)
    assert periodic.shape == plain.shape


@pytest.mark.parametrize("shift", [1, 3, 7])
def test_zonally_periodic_upsample_is_zonally_periodic(shift):
    """Upsampling commutes with circular shifts in longitude only if the
    upsampler is periodic along that axis. The plain bilinear upsampler is not,
    which is the source of the lon=0 seam.
    """
    x = torch.randn(2, 3, 8, 16)
    periodic = ZonallyPeriodicBilinearUpsample()
    shifted_then_up = periodic(torch.roll(x, shifts=shift, dims=-1))
    up_then_shifted = torch.roll(periodic(x), shifts=2 * shift, dims=-1)
    assert torch.allclose(shifted_then_up, up_then_shifted, atol=1e-5)

    plain = BilinearUpsample()
    assert not torch.allclose(
        plain(torch.roll(x, shifts=shift, dims=-1)),
        torch.roll(plain(x), shifts=2 * shift, dims=-1),
        atol=1e-5,
    )


def test_samudra_zonally_periodic_upsample_runs_and_differs():
    input_channels, output_channels = 2, 3
    img_shape = (9, 18)
    n_samples = 4

    def build(zonally_periodic_upsample: bool) -> Samudra:
        torch.manual_seed(0)
        return Samudra(
            input_channels=input_channels,
            output_channels=output_channels,
            ch_width=[3, 3],
            dilation=[1, 2],
            n_layers=[1, 1],
            norm="batch",
            zonally_periodic_upsample=zonally_periodic_upsample,
        )

    periodic_model = build(zonally_periodic_upsample=True)
    default_model = build(zonally_periodic_upsample=False)

    x = torch.randn(n_samples, input_channels, *img_shape)
    with torch.no_grad():
        periodic_out = periodic_model(x)
        default_out = default_model(x)

    assert periodic_out.shape == (n_samples, output_channels, *img_shape)
    assert not torch.isnan(periodic_out).any()
    assert not torch.isinf(periodic_out).any()
    # the periodic upsampling changes the result relative to the default
    assert not torch.allclose(periodic_out, default_out)


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


def _samudra(**kwargs):
    return Samudra(
        input_channels=4,
        output_channels=3,
        ch_width=[8, 12],
        dilation=[1, 2],
        n_layers=[1, 1],
        **kwargs,
    )


def _context(n_noise, n_samples, img_shape):
    return Context(
        embedding_scalar=None,
        embedding_pos=None,
        labels=None,
        noise=torch.randn(n_samples, n_noise, *img_shape),
    )


def _noise_context_config(n_noise):
    return ContextConfig(
        embed_dim_scalar=0,
        embed_dim_labels=0,
        embed_dim_noise=n_noise,
        embed_dim_pos=0,
    )


def test_unconditioned_convnext_block_state_dict_has_no_conditioning_keys():
    """Checkpoints predating conditioning must still load, so an unconditioned
    block's state dict must be byte-for-byte the same set of keys."""
    block = ConvNeXtBlock(in_channels=4, out_channels=4, upscale_factor=2)
    keys = set(block.state_dict())
    assert not any("noise_conditioning" in key for key in keys)
    assert keys == {
        "convblock.0.weight",
        "convblock.0.bias",
        "convblock.2.cap",
        "convblock.3.weight",
        "convblock.3.bias",
        "convblock.5.cap",
        "convblock.6.weight",
        "convblock.6.bias",
    }


def test_convnext_block_conditioning_requires_a_norm():
    with pytest.raises(ValueError, match="requires a normalization layer"):
        ConvNeXtBlock(
            in_channels=4,
            out_channels=4,
            norm=None,
            context_config=_noise_context_config(4),
        )


def test_convnext_block_conditioning_requires_context_at_forward():
    block = ConvNeXtBlock(
        in_channels=4, out_channels=4, context_config=_noise_context_config(4)
    )
    with pytest.raises(ValueError, match="requires a"):
        block(torch.randn(2, 4, 8, 16))


def test_convnext_block_conditioning_resamples_coarser_noise():
    """Inside the U-Net a conditioned block runs at coarser resolution than the
    input-resolution noise field, so the conditioning must resample."""
    n_noise = 4
    block = ConvNeXtBlock(
        in_channels=4, out_channels=4, context_config=_noise_context_config(n_noise)
    )
    for module in block.noise_conditioning.values():
        torch.nn.init.normal_(module.W_scale.weight, std=0.1)
        torch.nn.init.normal_(module.W_bias.weight, std=0.1)
    x = torch.randn(2, 4, 4, 8)
    fine = torch.randn(2, n_noise, 16, 32)
    coarse = torch.nn.functional.adaptive_avg_pool2d(fine, (4, 8))
    with torch.no_grad():
        from_fine = block(
            x,
            Context(embedding_scalar=None, embedding_pos=None, labels=None, noise=fine),
        )
        from_coarse = block(
            x,
            Context(
                embedding_scalar=None, embedding_pos=None, labels=None, noise=coarse
            ),
        )
        # a resampling that keeps the shape but not the area average
        from_subsample = block(
            x,
            Context(
                embedding_scalar=None,
                embedding_pos=None,
                labels=None,
                noise=fine[..., ::4, ::4],
            ),
        )
    assert from_fine.shape == x.shape
    # the resampling is the area average, not just any shape-matching reduction
    torch.testing.assert_close(from_fine, from_coarse, rtol=0.0, atol=0.0)
    assert not torch.allclose(from_fine, from_subsample)


@pytest.mark.parametrize("noise_injection", ["bottleneck", "all_blocks"])
def test_samudra_noise_injection_conditions_expected_blocks(noise_injection):
    n_noise = 4
    model = _samudra(
        context_config=_noise_context_config(n_noise),
        noise_injection=noise_injection,
    )
    blocks = [layer for layer in model.layers if isinstance(layer, ConvNeXtBlock)]
    # 2 encoder blocks + bottleneck + 2 decoder blocks
    assert len(blocks) == 5
    conditioned = [len(block.noise_conditioning) > 0 for block in blocks]
    if noise_injection == "bottleneck":
        assert conditioned == [False, False, True, False, False]
    else:
        assert conditioned == [True] * 5


@pytest.mark.parametrize("noise_injection", ["bottleneck", "all_blocks"])
def test_samudra_conditioning_is_zero_init_so_noise_is_inert(noise_injection):
    """Zero-initialized conditioning weights make training start deterministic:
    two different noise draws must give bit-identical output, and that output
    must match the unconditioned model with the same parameters."""
    torch.manual_seed(0)
    n_noise = 4
    img_shape = (16, 32)
    conditioned = _samudra(
        context_config=_noise_context_config(n_noise),
        noise_injection=noise_injection,
    )
    x = torch.randn(2, 4, *img_shape)
    with torch.no_grad():
        first = conditioned(x, _context(n_noise, 2, img_shape))
        second = conditioned(x, _context(n_noise, 2, img_shape))
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    plain = _samudra()
    plain.load_state_dict(
        {
            key: value
            for key, value in conditioned.state_dict().items()
            if "noise_conditioning" not in key
        }
    )
    with torch.no_grad():
        torch.testing.assert_close(plain(x), first, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("noise_injection", ["bottleneck", "all_blocks"])
def test_samudra_noise_changes_output_once_conditioning_is_trained(noise_injection):
    torch.manual_seed(0)
    n_noise = 4
    img_shape = (16, 32)
    model = _samudra(
        context_config=_noise_context_config(n_noise),
        noise_injection=noise_injection,
    )
    for module in model.modules():
        if isinstance(module, NoiseConditioning):
            torch.nn.init.normal_(module.W_scale.weight, std=0.1)
            torch.nn.init.normal_(module.W_bias.weight, std=0.1)
    x = torch.randn(2, 4, *img_shape)
    with torch.no_grad():
        first = model(x, _context(n_noise, 2, img_shape))
        second = model(x, _context(n_noise, 2, img_shape))
    assert first.shape == (2, 3, *img_shape)
    assert not torch.allclose(first, second)


def test_samudra_conditioning_gradients_reach_conditioning_weights():
    n_noise = 4
    img_shape = (16, 32)
    model = _samudra(
        context_config=_noise_context_config(n_noise), noise_injection="all_blocks"
    )
    model(
        torch.randn(2, 4, *img_shape), _context(n_noise, 2, img_shape)
    ).sum().backward()
    for module in model.modules():
        if isinstance(module, NoiseConditioning):
            assert module.W_bias.weight.grad is not None
            assert torch.any(module.W_bias.weight.grad != 0.0)


def test_samudra_checkpointing_with_conditioning():
    n_noise = 4
    img_shape = (16, 32)
    model = _samudra(
        context_config=_noise_context_config(n_noise),
        noise_injection="all_blocks",
        checkpoint_strategy="all",
    )
    x = torch.randn(2, 4, *img_shape, requires_grad=True)
    out = model(x, _context(n_noise, 2, img_shape))
    assert out.shape == (2, 3, *img_shape)
    out.sum().backward()
    assert x.grad is not None


def test_samudra_rejects_inconsistent_conditioning_config():
    with pytest.raises(ValueError, match="noise_injection to be set"):
        _samudra(context_config=_noise_context_config(4))
    with pytest.raises(ValueError, match="requires context_config"):
        _samudra(noise_injection="bottleneck")
    with pytest.raises(ValueError, match="embed_dim_noise > 0"):
        _samudra(context_config=_noise_context_config(0), noise_injection="bottleneck")
    with pytest.raises(ValueError, match="only supports noise conditioning"):
        _samudra(
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_labels=0,
                embed_dim_noise=4,
                embed_dim_pos=8,
            ),
            noise_injection="bottleneck",
        )
