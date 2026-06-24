import numpy as np
import torch

from fme.core.spherical_lowpass import SphericalLowPass


def _checkerboard(nlat: int, nlon: int) -> torch.Tensor:
    i = torch.arange(nlat)[:, None]
    j = torch.arange(nlon)[None, :]
    return ((-1.0) ** (i + j)).float()


def test_preserves_shape_and_global_mean():
    lp = SphericalLowPass(32, 64, "equiangular")
    x = torch.randn(2, 3, 32, 64) + 5.0
    y = lp(x, cutoff_degree=8)
    assert y.shape == x.shape
    # degree-0 is retained, so the spherical (area-weighted) mean is preserved;
    # the grid-point arithmetic mean is preserved only approximately because an
    # equiangular grid oversamples the poles.
    torch.testing.assert_close(
        y.mean(dim=(-2, -1)), x.mean(dim=(-2, -1)), rtol=2e-2, atol=5e-2
    )


def test_removes_grid_scale_checkerboard():
    nlat, nlon = 32, 64
    lp = SphericalLowPass(nlat, nlon, "equiangular")
    x = _checkerboard(nlat, nlon)
    y = lp(x, cutoff_degree=8)
    # the 2-gridpoint checkerboard is the highest wavenumbers; low-pass should
    # remove almost all of its variance
    assert y.std() < 0.1 * x.std()


def test_larger_cutoff_retains_more_variance():
    nlat, nlon = 48, 96
    lp = SphericalLowPass(nlat, nlon, "equiangular")
    torch.manual_seed(0)
    x = torch.randn(nlat, nlon)
    stds = [lp(x, cutoff_degree=c).std().item() for c in (4, 12, 24)]
    assert stds[0] < stds[1] < stds[2]


def test_cutoff_clamped_and_caches():
    lp = SphericalLowPass(16, 32, "equiangular")
    x = torch.randn(16, 32)
    # cutoff above max_degree is clamped (no error), below 1 is clamped to 1
    assert lp(x, cutoff_degree=10_000).shape == x.shape
    assert lp(x, cutoff_degree=0).shape == x.shape
    # operators are cached per cutoff/device
    assert len(lp._cache) == 2
    lp(x, cutoff_degree=0)
    assert len(lp._cache) == 2


def test_low_degree_field_passes_through():
    # a smooth, large-scale field should be largely unchanged by a modest cutoff
    nlat, nlon = 64, 128
    lp = SphericalLowPass(nlat, nlon, "equiangular")
    lat = torch.linspace(-np.pi / 2, np.pi / 2, nlat)[:, None]
    lon = torch.linspace(0, 2 * np.pi, nlon)[None, :]
    x = torch.cos(lat) * torch.cos(lon)  # degree-1-ish structure
    y = lp(x, cutoff_degree=12)
    assert (y - x).std() < 0.05 * x.std()
