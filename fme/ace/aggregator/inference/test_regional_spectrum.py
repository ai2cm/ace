import math

import matplotlib.pyplot as plt
import pytest
import torch

from fme.ace.aggregator.inference.regional_spectrum import (
    PairedRegionalSpectrumAggregator,
    RegionalSpectrumAggregator,
    compute_isotropic_spectrum,
)
from fme.core.coordinates import LatLonRegion


def test_compute_isotropic_spectrum():
    """3D (B, H, W) input with detrend='linear', window='hann', truncate=True."""
    H, W = 64, 64
    B = 3
    kx_cycles = 10
    expected_k = kx_cycles / W

    x = torch.arange(W, dtype=torch.float64)
    y = torch.arange(H, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    signal = torch.sin(2 * math.pi * kx_cycles * X / W)
    data = signal.unsqueeze(0).expand(B, -1, -1).clone()

    k_bins, spectrum = compute_isotropic_spectrum(
        data, detrend="linear", window="hann", truncate=True
    )

    num_bins = min(H, W) // 2 - 1
    assert k_bins.shape == (num_bins,)
    assert spectrum.shape == (B, num_bins)

    torch.testing.assert_close(spectrum[0], spectrum[1])
    torch.testing.assert_close(spectrum[0], spectrum[2])

    spec = spectrum[0].clone()
    spec[torch.isnan(spec)] = 0.0
    peak_idx = spec.argmax()
    peak_k = k_bins[peak_idx].item()
    bin_width = (k_bins[1] - k_bins[0]).item()
    assert abs(peak_k - expected_k) <= 0.5 * bin_width


def _make_region(nlat, nlon, sub_lat, sub_lon):
    lat = torch.linspace(-90, 90, nlat)
    lon = torch.linspace(0, 360, nlon + 1)[:nlon]
    lat_hi = lat[sub_lat - 1].item()
    lon_hi = lon[sub_lon - 1].item()
    return LatLonRegion(
        lat=lat,
        lon=lon,
        lat_bounds=(lat[0].item(), lat_hi),
        lon_bounds=(lon[0].item(), lon_hi),
    )


def test_regional_power_spectrum_aggregator():
    nlat, nlon = 32, 64
    sub_lat, sub_lon = 16, 32
    region = _make_region(nlat, nlon, sub_lat, sub_lon)
    agg = RegionalSpectrumAggregator(region)

    data1 = {"a": torch.randn(2, 2, nlat, nlon)}
    data2 = {"a": torch.randn(2, 3, nlat, nlon)}
    agg.record_batch(data1)
    agg.record_batch(data2)
    result = agg.get_mean()

    assert "a" in result
    num_bins = min(sub_lat, sub_lon) // 2 - 1
    assert result["a"].shape == (num_bins,)

    data_concat = torch.cat([data1["a"], data2["a"]], dim=1)
    subsetted = region.subset(data_concat)
    _, expected_spectrum = compute_isotropic_spectrum(subsetted)
    expected_mean = torch.mean(expected_spectrum, dim=(0, 1))
    torch.testing.assert_close(result["a"], expected_mean)


@pytest.mark.parametrize("report_plot", [True, False])
def test_paired_regional_power_spectrum_aggregator(report_plot: bool):
    nlat, nlon = 32, 64
    sub_lat, sub_lon = 16, 32
    region = _make_region(nlat, nlon, sub_lat, sub_lon)
    agg = PairedRegionalSpectrumAggregator(region, report_plot=report_plot)

    data = {"a": torch.randn(2, 3, nlat, nlon)}
    agg.record_batch(data, data, None, None)
    result = agg.get_logs("spectrum")

    if report_plot:
        assert isinstance(result["spectrum/a"], plt.Figure)
    else:
        assert "spectrum/a" not in result
    for prefix in [
        "smallest_scale_norm_bias",
        "positive_norm_bias",
        "negative_norm_bias",
        "mean_abs_norm_bias",
    ]:
        assert f"spectrum/{prefix}/a" in result
        assert isinstance(result[f"spectrum/{prefix}/a"], float)
