import itertools
import json
import math
import os

import pytest
import torch

from fme.ace.models.ocean.m2lines.layers import (
    AvgPool,
    BilinearUpsample,
    ConvNeXtBlock,
    LatPad,
    MultiResolutionFiLM,
    ZonallyPeriodicBilinearUpsample,
    _rotate_half_longitude,
    pad_latitude,
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


@pytest.mark.parametrize("upsampling", [2, 3])
@pytest.mark.parametrize("shift", [1, 3, 7])
def test_zonally_periodic_upsample_is_zonally_periodic(shift, upsampling):
    """Upsampling commutes with circular shifts in longitude only if the
    upsampler is periodic along that axis. The plain bilinear upsampler is not,
    which is the source of the lon=0 seam.
    """
    x = torch.randn(2, 3, 8, 16)
    periodic = ZonallyPeriodicBilinearUpsample(upsampling=upsampling)
    shifted_then_up = periodic(torch.roll(x, shifts=shift, dims=-1))
    up_then_shifted = torch.roll(periodic(x), shifts=upsampling * shift, dims=-1)
    assert torch.allclose(shifted_then_up, up_then_shifted, atol=1e-5)

    plain = BilinearUpsample(upsampling=upsampling)
    assert not torch.allclose(
        plain(torch.roll(x, shifts=shift, dims=-1)),
        torch.roll(plain(x), shifts=upsampling * shift, dims=-1),
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


def _reference_convnext_forward(
    block: ConvNeXtBlock, x: torch.Tensor, lat_pad: LatPad
) -> torch.Tensor:
    """ConvNeXtBlock.forward (no conditioning, no LayerNorm) with ``lat_pad``
    written out at each conv."""
    skip = block.skip_module(x)
    for layer in block.convblock:
        if isinstance(layer, torch.nn.Conv2d) and layer.kernel_size[0] != 1:
            x = pad_latitude(x, block.N_pad, block.N_pad, lat_pad)
            x = torch.nn.functional.pad(
                x, (block.N_pad, block.N_pad, 0, 0), mode=block.pad
            )
        x = layer(x)
    return skip + x


def _reference_upsample(
    x: torch.Tensor, zonally_periodic: bool, lat_pad: LatPad
) -> torch.Tensor:
    """Samudra's 2x upsampler, with ``lat_pad`` written out when periodic."""
    if not zonally_periodic:
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
    height, width = x.shape[-2:]
    n_lat = 0 if lat_pad == "constant" else 1
    x = pad_latitude(x, n_lat, n_lat, lat_pad)
    x = torch.nn.functional.pad(x, (1, 1, 0, 0), mode="circular")
    x = torch.nn.functional.interpolate(
        x, scale_factor=2, mode="bilinear", align_corners=False
    )
    return x[..., 2 * n_lat : 2 * (n_lat + height), 2 : 2 * (1 + width)]


def _reference_samudra_forward(
    model: Samudra, fts: torch.Tensor, lat_pad: LatPad
) -> torch.Tensor:
    """Samudra.forward with ``lat_pad`` written out at every site: block convs,
    the final conv, the upsampler, and the decoder refill. Under "constant" it
    is the original forward, since zero rows pad the same before or after
    the longitude padding."""
    temp: list[torch.Tensor] = []
    count = 0
    for layer in model.layers:
        if isinstance(layer, torch.nn.Conv2d):
            fts = pad_latitude(fts, model.N_pad, model.N_pad, lat_pad)
            fts = torch.nn.functional.pad(
                fts, (model.N_pad, model.N_pad, 0, 0), mode=model.pad
            )
        if isinstance(layer, ConvNeXtBlock):
            fts = _reference_convnext_forward(layer, fts, lat_pad)
        elif isinstance(layer, BilinearUpsample | ZonallyPeriodicBilinearUpsample):
            fts = _reference_upsample(fts, model.zonally_periodic_upsample, lat_pad)
        else:
            fts = layer(fts)
        if count < model.num_steps:
            if isinstance(layer, ConvNeXtBlock):
                temp.append(fts)
                count += 1
        elif isinstance(layer, BilinearUpsample | ZonallyPeriodicBilinearUpsample):
            skip = temp[2 * model.num_steps - count - 1]
            pad_h = skip.shape[2] - fts.shape[2]
            pad_w = skip.shape[3] - fts.shape[3]
            fts = pad_latitude(fts, pad_h // 2, pad_h - pad_h // 2, lat_pad)
            fts = torch.nn.functional.pad(
                fts, (pad_w // 2, pad_w - pad_w // 2, 0, 0), mode=model.pad
            )
            fts = fts + skip
            count += 1
    return fts


def _small_samudra(**kwargs) -> Samudra:
    """Three levels on a (22, 36) grid, so some levels pool an odd height
    or width, as the 180x360 grid does."""
    torch.manual_seed(0)
    return Samudra(
        input_channels=2,
        output_channels=3,
        ch_width=[4, 4, 4],
        dilation=[1, 2, 1],
        n_layers=[1, 1, 1],
        norm="batch",
        upscale_factor=2,
        **kwargs,
    )


_SMALL_SHAPE = (22, 36)


@pytest.mark.parametrize("zonally_periodic_upsample", [False, True])
def test_samudra_default_lat_pad_matches_original_forward_bitwise(
    zonally_periodic_upsample: bool,
):
    """The default reproduces the original forward exactly."""
    model = _small_samudra(zonally_periodic_upsample=zonally_periodic_upsample)
    explicit = _small_samudra(
        zonally_periodic_upsample=zonally_periodic_upsample,
        lat_pad="constant",
    )
    torch.manual_seed(1)
    x = torch.randn(2, 2, *_SMALL_SHAPE)
    with torch.no_grad():
        reference = _reference_samudra_forward(model, x, "constant")
        assert torch.equal(model(x), reference)
        assert torch.equal(explicit(x), reference)


@pytest.mark.parametrize("zonally_periodic_upsample", [False, True])
def test_samudra_pole_lat_pad_matches_reference_forward_bitwise(
    zonally_periodic_upsample: bool,
):
    """Pole padding reaches every site: block convs, the final conv, the
    decoder refill, and the periodic upsampler."""
    model = _small_samudra(
        zonally_periodic_upsample=zonally_periodic_upsample, lat_pad="pole"
    )
    torch.manual_seed(1)
    x = torch.randn(2, 2, *_SMALL_SHAPE)
    with torch.no_grad():
        assert torch.equal(model(x), _reference_samudra_forward(model, x, "pole"))


def test_pad_latitude_pole_rows():
    # rows 0..3 (south to north), columns 0..5: value = 10 * row + column
    x = (10 * torch.arange(4)[:, None] + torch.arange(6)[None, :]).float()
    padded = pad_latitude(x[None, None], 2, 1, "pole")[0, 0]
    assert padded.shape == (7, 6)
    torch.testing.assert_close(padded[2:6], x, rtol=0, atol=0)
    rolled = torch.roll(x, shifts=3, dims=-1)
    # beyond the south pole, outermost first: rows 1 then 0, rotated 180 deg
    torch.testing.assert_close(padded[0], rolled[1], rtol=0, atol=0)
    torch.testing.assert_close(padded[1], rolled[0], rtol=0, atol=0)
    # beyond the north pole: row 3, rotated 180 deg
    torch.testing.assert_close(padded[6], rolled[3], rtol=0, atol=0)
    assert padded[6, 0].item() == 33.0


def test_pad_latitude_pole_is_the_continuation_across_the_pole():
    """Pole padding of the cell centers' Cartesian coordinates matches the
    coordinates evaluated past the pole (latitude -90 - d, longitude L is
    the point -90 + d, L + 180)."""
    n_lat, n_lon, n_pad = 6, 8, 2

    def cartesian(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
        lat = torch.deg2rad(lat_deg)[:, None]
        lon = torch.deg2rad(lon_deg)[None, :]
        return torch.stack(
            [
                torch.cos(lat) * torch.cos(lon),
                torch.cos(lat) * torch.sin(lon),
                torch.sin(lat).expand(-1, lon.shape[-1]),
            ]
        )

    dlat = 180.0 / n_lat
    lon = (torch.arange(n_lon, dtype=torch.float64) + 0.5) * 360.0 / n_lon
    lat = -90.0 + (torch.arange(n_lat, dtype=torch.float64) + 0.5) * dlat
    lat_extended = (
        -90.0 + (torch.arange(-n_pad, n_lat + n_pad, dtype=torch.float64) + 0.5) * dlat
    )
    padded = pad_latitude(cartesian(lat, lon)[None], n_pad, n_pad, "pole")[0]
    torch.testing.assert_close(padded, cartesian(lat_extended, lon))


def test_pad_latitude_pole_odd_width_averages_the_two_antipodal_columns():
    x = torch.arange(5, dtype=torch.float32)[None, :].expand(2, -1)
    padded = pad_latitude(x[None, None], 1, 0, "pole")[0, 0]
    # antipode of column j is j + 2.5: the mean of columns j + 2 and j + 3
    expected = 0.5 * (torch.roll(x[0], -2) + torch.roll(x[0], -3))
    torch.testing.assert_close(padded[0], expected)


def test_pad_latitude_constant():
    x = torch.randn(1, 1, 5, 4)
    constant = pad_latitude(x, 2, 1, "constant")
    assert torch.equal(constant[..., [0, 1, -1], :], torch.zeros(1, 1, 3, 4))


def test_pad_latitude_pole_rejects_padding_beyond_the_height():
    with pytest.raises(ValueError, match="exceeds the height"):
        pad_latitude(torch.zeros(1, 1, 2, 4), 3, 0, "pole")


_LAT_PAD_COMBINATIONS = list(itertools.product(["constant", "pole"], [False, True]))


@pytest.mark.parametrize("lat_pad, zonally_periodic_upsample", _LAT_PAD_COMBINATIONS)
def test_samudra_lat_pad_options_keep_output_shape(lat_pad, zonally_periodic_upsample):
    model = _small_samudra(
        lat_pad=lat_pad,
        zonally_periodic_upsample=zonally_periodic_upsample,
    )
    x = torch.randn(2, 2, *_SMALL_SHAPE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3, *_SMALL_SHAPE)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("lat_pad, zonally_periodic_upsample", _LAT_PAD_COMBINATIONS)
def test_samudra_lat_pad_options_keep_output_shape_on_1deg_grid(
    lat_pad, zonally_periodic_upsample
):
    """The default four levels on the 180x360 grid."""
    torch.manual_seed(0)
    model = Samudra(
        input_channels=1,
        output_channels=1,
        ch_width=[2, 2, 2, 2],
        dilation=[1, 2, 4, 8],
        n_layers=[1, 1, 1, 1],
        norm="batch",
        upscale_factor=1,
        lat_pad=lat_pad,
        zonally_periodic_upsample=zonally_periodic_upsample,
    )
    with torch.no_grad():
        out = model(torch.randn(1, 1, 180, 360))
    assert out.shape == (1, 1, 180, 360)


def test_samudra_pole_lat_pad_changes_the_output():
    x = torch.randn(2, 2, *_SMALL_SHAPE)
    with torch.no_grad():
        default = _small_samudra()(x)
        changed = _small_samudra(lat_pad="pole")(x)
    assert not torch.allclose(default, changed)


@pytest.mark.parametrize("lat_pad, zonally_periodic_upsample", _LAT_PAD_COMBINATIONS)
def test_samudra_lat_pad_options_keep_state_dict(lat_pad, zonally_periodic_upsample):
    """A default checkpoint loads strictly into a model with any mode."""
    default = _small_samudra()
    model = _small_samudra(
        lat_pad=lat_pad,
        zonally_periodic_upsample=zonally_periodic_upsample,
    )
    assert {k: v.shape for k, v in default.state_dict().items()} == {
        k: v.shape for k, v in model.state_dict().items()
    }
    model.load_state_dict(default.state_dict(), strict=True)


def test_zonally_periodic_upsample_pole_lat_pad():
    x = torch.randn(2, 3, 9, 18)
    default = ZonallyPeriodicBilinearUpsample()(x)
    padded = ZonallyPeriodicBilinearUpsample(lat_pad="pole")(x)
    assert padded.shape == default.shape
    # only the outermost upsampled row at each latitude edge reads the padding
    torch.testing.assert_close(padded[..., 1:-1, :], default[..., 1:-1, :])
    assert not torch.allclose(padded[..., 0, :], default[..., 0, :])
    assert not torch.allclose(padded[..., -1, :], default[..., -1, :])
    assert torch.equal(ZonallyPeriodicBilinearUpsample(lat_pad="constant")(x), default)


def test_samudra_pole_lat_pad_with_noise_conditioning():
    n_noise = 4
    model = _samudra(
        context_config=_noise_context_config(n_noise),
        conditioned_blocks="all_blocks",
        lat_pad="pole",
    )
    img_shape = (18, 32)
    out = model(torch.randn(2, 4, *img_shape), _context(n_noise, 2, img_shape))
    assert out.shape == (2, 3, *img_shape)


def test_samudra_rejects_unknown_lat_pad():
    with pytest.raises(ValueError, match="unknown lat_pad"):
        _samudra(lat_pad="circular")


# each product divides both 180 and 360
_EXACT_1DEG = [[2, 2, 3], [2, 3, 3], [2, 2, 3, 3], [2, 2, 3, 1]]
_1DEG_SHAPE = (180, 360)


def _exact_samudra(pooling: list[int], **kwargs) -> Samudra:
    """One channel per level, dilations that fit every level of the 180x360
    grid under each pooling in ``_EXACT_1DEG``."""
    n_levels = len(pooling)
    torch.manual_seed(0)
    return Samudra(
        input_channels=1,
        output_channels=1,
        ch_width=[2] * n_levels,
        dilation=[1, 2, 4, 4][:n_levels],
        n_layers=[1] * n_levels,
        norm="batch",
        upscale_factor=1,
        pooling=pooling,
        **kwargs,
    )


@pytest.mark.parametrize("lat_pad, zonally_periodic_upsample", _LAT_PAD_COMBINATIONS)
def test_samudra_explicit_twos_match_default_bitwise(
    lat_pad, zonally_periodic_upsample
):
    """``pooling=[2] * L`` on a grid 2^L divides is the default network."""
    default = _small_samudra(
        lat_pad=lat_pad, zonally_periodic_upsample=zonally_periodic_upsample
    )
    explicit = _small_samudra(
        lat_pad=lat_pad,
        zonally_periodic_upsample=zonally_periodic_upsample,
        pooling=[2, 2, 2],
    )
    explicit.load_state_dict(default.state_dict(), strict=True)
    torch.manual_seed(1)
    x = torch.randn(2, 2, 16, 32)
    with torch.no_grad():
        assert torch.equal(explicit(x), default(x))


def test_samudra_pooling_validation():
    with pytest.raises(ValueError, match="expected one per level"):
        _small_samudra(pooling=[2, 2])
    with pytest.raises(ValueError, match="positive integers"):
        _small_samudra(pooling=[2, 0, 2])
    with pytest.raises(ValueError, match="positive integers"):
        _small_samudra(pooling=[2, -2, 2])
    x = torch.randn(1, 2, *_SMALL_SHAPE)
    with pytest.raises(ValueError, match="does not divide the grid shape"):
        _small_samudra(pooling=[2, 2, 2])(x)
    # 8 divides (16, 24); the bottleneck width 3 is odd
    with pytest.raises(ValueError, match="even width at every level"):
        _small_samudra(pooling=[2, 2, 2], lat_pad="pole")(torch.randn(1, 2, 16, 24))
    with torch.no_grad():
        out = _small_samudra(pooling=[2, 1, 2])(torch.randn(1, 2, 16, 24))
    assert out.shape == (1, 3, 16, 24)


@pytest.mark.parametrize("pooling", _EXACT_1DEG)
def test_samudra_exact_pooling_level_shapes(pooling):
    """Every encoder level has shape (H_l, W_l), and every decoder upsample
    returns its skip connection's shape, so no row or column is refilled."""
    model = _exact_samudra(pooling, lat_pad="pole", zonally_periodic_upsample=True)
    blocks = [layer for layer in model.layers if isinstance(layer, ConvNeXtBlock)]
    upsamplers = [
        layer
        for layer in model.layers
        if isinstance(layer, BilinearUpsample | ZonallyPeriodicBilinearUpsample)
    ]
    shapes: dict[torch.nn.Module, tuple[int, ...]] = {}
    handles = [
        module.register_forward_hook(
            lambda module, _, out: shapes.__setitem__(module, tuple(out.shape[2:]))
        )
        for module in blocks + upsamplers
    ]
    with torch.no_grad():
        out = model(torch.randn(1, 1, *_1DEG_SHAPE))
    for handle in handles:
        handle.remove()
    assert out.shape == (1, 1, *_1DEG_SHAPE)

    height, width = _1DEG_SHAPE
    level_shapes = [(height, width)]
    for factor in pooling:
        height, width = height // factor, width // factor
        level_shapes.append((height, width))
    n_levels = len(pooling)
    encoder, bottleneck = blocks[:n_levels], blocks[n_levels]
    assert [shapes[block] for block in encoder] == level_shapes[:-1]
    assert shapes[bottleneck] == level_shapes[-1]
    assert [shapes[up] for up in upsamplers] == level_shapes[-2::-1]


def _pool_chain(model: Samudra):
    pools = [layer for layer in model.layers if isinstance(layer, AvgPool)]

    def chain(x: torch.Tensor) -> torch.Tensor:
        for pool in pools:
            x = pool(x)
        return x

    return chain


@pytest.mark.parametrize("pooling", _EXACT_1DEG)
def test_exact_pool_chain_value_properties(pooling):
    """The pool chain of a built network averages disjoint P x P blocks that
    tile the 180x360 grid: every input cell reaches the bottleneck with weight
    1 / P^2, a constant field is preserved, the 180-degree rotation at the pole
    is exact, and both polar bottleneck rows draw on P input rows."""
    chain = _pool_chain(_exact_samudra(pooling))
    product = math.prod(pooling)
    height, width = _1DEG_SHAPE
    dtype = torch.float64

    x = torch.randn(1, 1, height, width, dtype=dtype, requires_grad=True)
    chain(x).sum().backward()
    assert x.grad is not None
    # coverage and uniform weight
    torch.testing.assert_close(
        x.grad, torch.full_like(x.grad, 1.0 / product**2), rtol=0, atol=1e-15
    )

    # constant field
    ones = torch.ones(1, 1, height, width, dtype=dtype)
    torch.testing.assert_close(chain(ones), torch.ones_like(chain(ones)))

    # pole roll: the bottleneck column centred at lon maps to lon + 180
    lon = torch.deg2rad((torch.arange(width, dtype=dtype) + 0.5) * 360.0 / width)
    for field in (torch.cos(lon), torch.sin(lon)):
        pooled = chain(field.expand(1, 1, height, width))
        torch.testing.assert_close(_rotate_half_longitude(pooled), -pooled)

    # pole symmetry
    for row, expected_rows in [
        (0, range(product)),
        (-1, range(height - product, height)),
    ]:
        x = torch.randn(1, 1, height, width, dtype=dtype, requires_grad=True)
        chain(x)[..., row, :].sum().backward()
        assert x.grad is not None
        rows_used = torch.nonzero(x.grad[0, 0].abs().sum(dim=-1)).flatten()
        assert rows_used.tolist() == list(expected_rows)


@pytest.mark.parametrize("lat_pad", ["constant", "pole"])
@pytest.mark.parametrize("pooling", _EXACT_1DEG)
def test_samudra_exact_pooling_is_zonally_equivariant(pooling, lat_pad):
    """Rolling the input by P columns rolls the output by P columns."""
    model = _exact_samudra(pooling, lat_pad=lat_pad, zonally_periodic_upsample=True)
    product = math.prod(pooling)
    torch.manual_seed(1)
    x = torch.randn(1, 1, *_1DEG_SHAPE)
    with torch.no_grad():
        rolled_then_run = model(torch.roll(x, shifts=product, dims=-1))
        run_then_rolled = torch.roll(model(x), shifts=product, dims=-1)
    torch.testing.assert_close(rolled_then_run, run_then_rolled, rtol=0, atol=1e-5)


def test_samudra_exact_pooling_state_dict_matches_default():
    """Pooling adds no parameters: a default checkpoint loads strictly."""
    default = _small_samudra()
    exact = _small_samudra(pooling=[2, 3, 1])
    assert {k: v.shape for k, v in default.state_dict().items()} == {
        k: v.shape for k, v in exact.state_dict().items()
    }
    exact.load_state_dict(default.state_dict(), strict=True)


def test_samudra_exact_pooling_with_noise_conditioning():
    n_noise = 2
    model = _exact_samudra(
        [2, 2, 3],
        lat_pad="pole",
        zonally_periodic_upsample=True,
        context_config=_noise_context_config(n_noise),
        conditioned_blocks="all_blocks",
    )
    with torch.no_grad():
        out = model(torch.randn(2, 1, *_1DEG_SHAPE), _context(n_noise, 2, _1DEG_SHAPE))
    assert out.shape == (2, 1, *_1DEG_SHAPE)
