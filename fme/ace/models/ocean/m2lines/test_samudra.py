import json
import math
import os

import pytest
import torch

from fme.ace.models.ocean.m2lines.layers import (
    BilinearUpsample,
    ConvNeXtBlock,
    MultiResolutionFiLM,
    ZonallyPeriodicBilinearUpsample,
)
from fme.ace.models.ocean.m2lines.samudra import Samudra
from fme.ace.registry.registry import ModuleSelector
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.testing import validate_tensor

DIR = os.path.abspath(os.path.dirname(__file__))


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


def test_released_checkpoint_still_loads():
    """A released checkpoint must keep loading into this module.

    Checked against a committed manifest of the SamudrACE-E3SMv3 ocean
    checkpoint -- its builder config and the name and shape of every parameter
    -- rather than the 327 MB artifact. Matching the rebuilt state_dict against
    the recorded parameters is the condition ``load_state_dict(strict=True)``
    needs, and is the property structural edits to ``ConvNeXtBlock`` break;
    numerics are pinned separately by ``test_samudra_output_is_unchanged``.
    See the manifest's ``_comment`` for how to regenerate it.
    """
    path = os.path.join(DIR, "testdata/samudrace_e3smv3_ocean_manifest.json")
    with open(path) as f:
        manifest = json.load(f)

    # via ModuleSelector rather than the builder directly, so the
    # dacite(strict=True) deserialization of the stored config dict -- its own
    # compatibility surface -- is covered too
    selector = ModuleSelector(**manifest["builder"])
    module = selector.build(
        manifest["n_in_channels"],
        manifest["n_out_channels"],
        DatasetInfo(img_shape=tuple(manifest["img_shape"])),
    )
    built = {k: list(v.shape) for k, v in module.torch_module.state_dict().items()}
    recorded = manifest["parameters"]

    assert set(built) == set(recorded), (
        f"state dict keys drifted from the released checkpoint; "
        f"missing {sorted(set(recorded) - set(built))}, "
        f"unexpected {sorted(set(built) - set(recorded))}"
    )
    mismatched = {
        k: (recorded[k], built[k]) for k in recorded if recorded[k] != built[k]
    }
    assert not mismatched, f"parameter shapes drifted: {mismatched}"


def _samudra(**kwargs):
    return Samudra(
        input_channels=4,
        output_channels=3,
        ch_width=[8, 12],
        dilation=[1, 2],
        n_layers=[1, 1],
        **kwargs,
    )


def _noise_only_context(noise):
    return Context(embedding_scalar=None, embedding_pos=None, labels=None, noise=noise)


def _context(n_noise, n_samples, img_shape):
    return _noise_only_context(torch.randn(n_samples, n_noise, *img_shape))


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
    input-resolution noise field, so the conditioning must resample. The
    resampling is the area average scaled back up to unit variance -- not the
    plain average (which would attenuate it) and not a subsample (which would
    tie the coarse value to an arbitrary corner of the fine patch).
    """
    n_noise = 4
    block = ConvNeXtBlock(
        in_channels=4, out_channels=4, context_config=_noise_context_config(n_noise)
    )
    for module in block.noise_conditioning.values():
        torch.nn.init.normal_(module.W_scale.weight, std=0.1)
        torch.nn.init.normal_(module.W_bias.weight, std=0.1)
    x = torch.randn(2, 4, 4, 8)
    fine = torch.randn(2, n_noise, 16, 32)
    averaged = torch.nn.functional.adaptive_avg_pool2d(fine, (4, 8))
    rescaled = averaged * math.sqrt((16 * 32) / (4 * 8))
    subsampled = fine[..., ::4, ::4]
    with torch.no_grad():
        from_fine = block(x, _noise_only_context(fine))
        from_rescaled = block(x, _noise_only_context(rescaled))
        from_averaged = block(x, _noise_only_context(averaged))
        from_subsampled = block(x, _noise_only_context(subsampled))
    assert from_fine.shape == x.shape
    torch.testing.assert_close(from_fine, from_rescaled, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(from_fine, from_averaged)
    assert not torch.allclose(from_fine, from_subsampled)


def test_conditioning_resampling_uses_every_source_cell():
    """The area average is what distinguishes this from a subsample: perturbing
    any single fine cell must move the coarse cell that contains it. A
    subsample ignores all but one cell per patch, so this fails for it.
    """
    film = MultiResolutionFiLM(n_channels=1, embed_dim=1)
    target = (2, 3)
    source = torch.zeros(1, 1, 8, 9)
    baseline = film._resample(source, target)
    for row, col in [(0, 0), (1, 1), (3, 4), (7, 8), (2, 7)]:
        bumped = source.clone()
        bumped[0, 0, row, col] = 1.0
        moved = film._resample(bumped, target) - baseline
        assert (moved != 0).sum() == 1, (row, col)


def test_conditioning_noise_keeps_unit_variance_at_every_depth():
    """Conditioning weights are shared-LR and zero-initialized, so the noise
    reaching each block has to be the same size at every depth. The bare area
    average would hand the bottleneck ~1/16 amplitude on the 4-degree grid and
    spread it 17-fold across the blocks ``all_blocks`` conditions, which is why
    the average is rescaled by the square root of the area ratio.
    """
    torch.manual_seed(0)
    film = MultiResolutionFiLM(n_channels=1, embed_dim=1)
    # the 4-degree ocean grid and the block resolutions Samudra's AvgPools give.
    # 45 -> 22 and 45 -> 11 are the non-divisible ratios, where adaptive pooling
    # builds ragged windows and a single global rescale would leave ~0.83.
    targets = [(45, 90), (22, 45), (11, 22), (5, 11), (2, 5)]
    noise = torch.randn(20000, 1, 45, 90)
    for target in targets:
        resampled = film._resample(noise, target)
        assert resampled.shape[-2:] == target
        # every cell individually, not just the field as a whole: a global
        # rescale gets the mean right while leaving cells with 2-row and 3-row
        # windows at different amplitudes
        per_cell = resampled.std(dim=0)
        assert per_cell.min().item() > 0.97, target
        assert per_cell.max().item() < 1.03, target


def test_conditioning_preserve_variance_false_is_the_plain_average():
    """A smooth conditioning field should coarsen to its own magnitude, not be
    scaled up as an iid field is."""
    torch.manual_seed(0)
    film = MultiResolutionFiLM(n_channels=1, embed_dim=1, preserve_variance=False)
    field = torch.randn(8, 1, 16, 32)
    torch.testing.assert_close(
        film._resample(field, (4, 8)),
        torch.nn.functional.adaptive_avg_pool2d(field, (4, 8)),
    )


@pytest.mark.parametrize("conditioned_blocks", ["bottleneck", "all_blocks"])
def test_samudra_conditioned_blocks_conditions_expected_blocks(conditioned_blocks):
    n_noise = 4
    model = _samudra(
        context_config=_noise_context_config(n_noise),
        conditioned_blocks=conditioned_blocks,
    )
    blocks = [layer for layer in model.layers if isinstance(layer, ConvNeXtBlock)]
    # 2 encoder blocks + bottleneck + 2 decoder blocks
    assert len(blocks) == 5
    conditioned = [len(block.noise_conditioning) > 0 for block in blocks]
    if conditioned_blocks == "bottleneck":
        assert conditioned == [False, False, True, False, False]
    else:
        assert conditioned == [True] * 5


@pytest.mark.parametrize("conditioned_blocks", ["bottleneck", "all_blocks"])
def test_samudra_conditioning_is_zero_init_so_noise_is_inert(conditioned_blocks):
    """Zero-initialized conditioning weights make training start deterministic:
    two different noise draws must give bit-identical output, and that output
    must match the unconditioned model with the same parameters."""
    torch.manual_seed(0)
    n_noise = 4
    img_shape = (16, 32)
    conditioned = _samudra(
        context_config=_noise_context_config(n_noise),
        conditioned_blocks=conditioned_blocks,
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


@pytest.mark.parametrize("conditioned_blocks", ["bottleneck", "all_blocks"])
def test_samudra_noise_changes_output_once_conditioning_is_trained(conditioned_blocks):
    torch.manual_seed(0)
    n_noise = 4
    img_shape = (16, 32)
    model = _samudra(
        context_config=_noise_context_config(n_noise),
        conditioned_blocks=conditioned_blocks,
    )
    for module in model.modules():
        if isinstance(module, MultiResolutionFiLM):
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
        context_config=_noise_context_config(n_noise), conditioned_blocks="all_blocks"
    )
    model(
        torch.randn(2, 4, *img_shape), _context(n_noise, 2, img_shape)
    ).sum().backward()
    for module in model.modules():
        if isinstance(module, MultiResolutionFiLM):
            assert module.W_bias.weight.grad is not None
            assert torch.any(module.W_bias.weight.grad != 0.0)


def test_samudra_checkpointing_with_conditioning():
    n_noise = 4
    img_shape = (16, 32)
    model = _samudra(
        context_config=_noise_context_config(n_noise),
        conditioned_blocks="all_blocks",
        checkpoint_strategy="all",
    )
    x = torch.randn(2, 4, *img_shape, requires_grad=True)
    out = model(x, _context(n_noise, 2, img_shape))
    assert out.shape == (2, 3, *img_shape)
    out.sum().backward()
    assert x.grad is not None


def test_samudra_rejects_inconsistent_conditioning_config():
    with pytest.raises(ValueError, match="conditioned_blocks to be set"):
        _samudra(context_config=_noise_context_config(4))
    with pytest.raises(ValueError, match="requires context_config"):
        _samudra(conditioned_blocks="bottleneck")
    with pytest.raises(ValueError, match="embed_dim_noise > 0"):
        _samudra(
            context_config=_noise_context_config(0), conditioned_blocks="bottleneck"
        )
    with pytest.raises(ValueError, match="only supports noise conditioning"):
        _samudra(
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_labels=0,
                embed_dim_noise=4,
                embed_dim_pos=8,
            ),
            conditioned_blocks="bottleneck",
        )


def test_conditioning_rejects_noise_coarser_than_the_block():
    conditioning = MultiResolutionFiLM(n_channels=3, embed_dim=2)
    with pytest.raises(ValueError, match="coarser than the block grid"):
        conditioning(torch.zeros(1, 3, 16, 32), torch.randn(1, 2, 4, 8))
