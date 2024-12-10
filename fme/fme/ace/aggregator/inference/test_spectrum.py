import matplotlib.pyplot as plt
import torch
import torch_harmonics

import fme
from fme.ace.aggregator.inference.spectrum import (
    PairedSphericalPowerSpectrumAggregator,
    SphericalPowerSpectrumAggregator,
)
from fme.core.metrics import spherical_power_spectrum


def test_spherical_power_spectrum_aggregator():
    nlat = 8
    nlon = 16
    grid = "legendre-gauss"
    agg = SphericalPowerSpectrumAggregator(nlat, nlon, grid=grid)
    data = {"a": torch.randn(2, 2, nlat, nlon, device=fme.get_device())}
    data2 = {"a": torch.randn(2, 3, nlat, nlon, device=fme.get_device())}
    agg.record_batch(data)
    agg.record_batch(data2)
    result = agg.get_mean()
    assert "a" in result
    assert result["a"].shape == (nlat,)

    sht = torch_harmonics.RealSHT(nlat, nlon, grid=grid)
    data_concat = torch.cat([data["a"], data2["a"]], dim=1)
    expected_value = torch.mean(spherical_power_spectrum(data_concat, sht), dim=(0, 1))
    torch.testing.assert_close(result["a"], expected_value)


def test_paired_spherical_power_spectrum_aggregator():
    nlat = 8
    nlon = 16
    agg = PairedSphericalPowerSpectrumAggregator(nlat, nlon)
    data = {"a": torch.randn(2, 3, nlat, nlon, device=fme.get_device())}
    agg.record_batch(data, data, None, None)
    result = agg.get_logs("spectrum")
    assert isinstance(result["spectrum/a"], plt.Figure)
