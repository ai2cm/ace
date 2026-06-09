import dataclasses

import pytest
import torch

from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.models.conditional_sfno.s2convolutions import SpectralConvS2

from .s2convolutions import _contract_dhconv


@dataclasses.dataclass
class BenchmarkResult:
    ms_total: float
    ms_per: float
    max_alloc: int
    max_reserved: int
    y_shape: tuple
    y_dtype: torch.dtype


def benchmark(fn, iters=10, warmup=1) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    for _ in range(iters):
        y = fn()
    ender.record()
    torch.cuda.synchronize()

    ms = starter.elapsed_time(ender)
    return BenchmarkResult(
        ms_total=ms,
        ms_per=ms / iters,
        max_alloc=torch.cuda.max_memory_allocated(),
        max_reserved=torch.cuda.max_memory_reserved(),
        y_shape=tuple(y.shape),
        y_dtype=y.dtype,
    )


@pytest.mark.skipif(
    get_device().type != "cuda",
    reason=(
        "This test is only relevant for CUDA since "
        "it's testing speed of DHConv groups on GPU."
    ),
)  # noqa: E501
@pytest.mark.serial
def test_contract_dhconv_groups_are_faster():
    B = 2
    C = 512
    H = 180
    L = 360
    G = 8
    x = torch.randn(B, 1, C, H, L, dtype=torch.complex64, device=get_device())
    w = torch.randn(1, H, C, C, 2, dtype=torch.float32, device=get_device())

    def contract_ungrouped():
        return _contract_dhconv(x, w)

    ungrouped_result = benchmark(contract_ungrouped)

    x_grouped = x.reshape(B, G, C // G, H, L)
    w_grouped = torch.randn(
        G, H, C // G, C // G, 2, dtype=torch.float32, device=get_device()
    )

    def contract_grouped():
        return _contract_dhconv(x_grouped, w_grouped)

    grouped_result = benchmark(contract_grouped)

    assert grouped_result.ms_per < 2 / G * ungrouped_result.ms_per, (
        "Expected grouped DHConv to be faster than ungrouped, but got "
        f"{grouped_result.ms_per:.6f} seconds for grouped and "
        f"{ungrouped_result.ms_per:.6f} seconds for ungrouped."
    )
    assert grouped_result.max_alloc < 1.05 * ungrouped_result.max_alloc, (
        "Did not expect grouped DHConv to use significantly more memory "
        "than ungrouped, but got "
        f"{grouped_result.max_alloc/1024/1024:.2f} MB for grouped and "
        f"{ungrouped_result.max_alloc/1024/1024:.2f} MB for ungrouped."
    )


def _make_conv(embed_dim, n_lat=12, n_lon=24, **kwargs):
    operations = LatLonOperations(
        area_weights=torch.ones(n_lat, n_lon),
        grid="legendre-gauss",
    )
    sht = operations.get_real_sht()
    isht = operations.get_real_isht()
    return SpectralConvS2(
        forward_transform=sht,
        inverse_transform=isht,
        in_channels=embed_dim,
        out_channels=embed_dim,
        **kwargs,
    )


def test_spectral_ratio_reduces_internal_weight_shape():
    embed_dim = 16
    n_lat, n_lon = 12, 24
    full = _make_conv(embed_dim, n_lat, n_lon)
    half = _make_conv(embed_dim, n_lat, n_lon, spectral_ratio=0.5)

    assert full.spectral_channels == embed_dim
    assert half.spectral_channels == embed_dim // 2
    # weight is (num_groups, modes_lat, out_ch // g, in_ch // g, 2)
    assert full.weight.shape == (1, n_lat, embed_dim, embed_dim, 2)
    assert half.weight.shape == (
        1,
        n_lat,
        embed_dim // 2,
        embed_dim // 2,
        2,
    )
    # Projections present iff spectral_ratio < 1
    assert full.pre_proj is None and full.post_proj is None
    assert half.pre_proj is not None and half.post_proj is not None
    assert half.pre_proj.weight.shape == (embed_dim // 2, embed_dim, 1, 1)
    assert half.post_proj.weight.shape == (embed_dim, embed_dim // 2, 1, 1)


def test_spectral_ratio_forward_pass_and_shape():
    embed_dim = 16
    n_lat, n_lon = 12, 24
    conv = _make_conv(embed_dim, n_lat, n_lon, spectral_ratio=0.5)
    x = torch.randn(2, embed_dim, n_lat, n_lon)
    y, residual = conv(x)
    assert y.shape == (2, embed_dim, n_lat, n_lon)
    # No round-trip residual (grids match, no filter_residual), so residual
    # should be the original input passed through unchanged.
    assert torch.equal(residual, x)


def test_spectral_ratio_matches_manual_qwp_contraction():
    """Verify that the spectral_ratio < 1 forward pass is equivalent to a
    sandwich Q @ inv_F( W'_l @ F(P x) )."""
    torch.manual_seed(0)
    embed_dim = 8
    n_lat, n_lon = 12, 24
    conv = _make_conv(embed_dim, n_lat, n_lon, spectral_ratio=0.5)
    x = torch.randn(2, embed_dim, n_lat, n_lon)
    with torch.no_grad():
        y, _ = conv(x)

        # manual reference
        sht = conv.forward_transform
        isht = conv.inverse_transform
        xp = conv.pre_proj(x.float())  # (B, C_spec, H, W)
        xs = sht(xp.float())  # (B, C_spec, lmax, mmax) complex
        # apply per-mode weight: w shape (1, modes_lat, out, in, 2)
        w = torch.view_as_complex(conv.weight)[0]  # (modes_lat, out, in)
        # contract over in -> out, per (lat)
        ys = torch.einsum("loi,bily->boly", w, xs)
        y_ref = isht(ys.contiguous())
        y_ref = conv.post_proj(y_ref)

    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)


def test_spectral_ratio_default_unchanged():
    """spectral_ratio=1.0 must not add projections or change params."""
    torch.manual_seed(0)
    embed_dim = 8
    baseline = _make_conv(embed_dim)
    with_default = _make_conv(embed_dim, spectral_ratio=1.0)
    baseline_params = {k: v.shape for k, v in baseline.state_dict().items()}
    default_params = {k: v.shape for k, v in with_default.state_dict().items()}
    assert baseline_params == default_params


def test_spectral_ratio_validation():
    embed_dim = 8
    with pytest.raises(ValueError, match="spectral_ratio"):
        _make_conv(embed_dim, spectral_ratio=0.0)
    with pytest.raises(ValueError, match="spectral_ratio"):
        _make_conv(embed_dim, spectral_ratio=1.5)
    # 8 * 0.25 -> 2 spectral channels, not divisible by num_groups=4
    with pytest.raises(ValueError, match="num_groups"):
        _make_conv(embed_dim, spectral_ratio=0.25, num_groups=4)
    with pytest.raises(NotImplementedError, match="preserve_global_mean"):
        _make_conv(embed_dim, spectral_ratio=0.5, preserve_global_mean=True)


def test_spectral_conv_s2_lora():
    in_channels = 8
    out_channels = in_channels
    n_lat = 12
    n_lon = 24
    operations = LatLonOperations(
        area_weights=torch.ones(n_lat, n_lon),
        grid="legendre-gauss",
    )
    sht = operations.get_real_sht()
    isht = operations.get_real_isht()

    conv1 = SpectralConvS2(
        forward_transform=sht,
        inverse_transform=isht,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    assert conv1.lora_A is None
    assert conv1.lora_B is None
    conv2 = SpectralConvS2(
        forward_transform=sht,
        inverse_transform=isht,
        in_channels=in_channels,
        out_channels=out_channels,
        lora_rank=4,
        lora_alpha=8,
    )
    assert conv2.lora_A is not None
    assert conv2.lora_B is not None

    conv2.load_state_dict(conv1.state_dict(), strict=False)
    x = torch.randn(2, in_channels, n_lat, n_lon)
    y1, residual1 = conv1(x)
    y2, residual2 = conv2(x)

    # initial outputs should be identical since LoRA starts at 0
    assert torch.allclose(y1, y2, atol=1e-6)
    assert torch.allclose(residual1, residual2, atol=1e-6)
