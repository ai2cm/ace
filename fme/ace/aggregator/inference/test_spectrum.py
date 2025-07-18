import matplotlib.pyplot as plt
import pytest
import torch
import torch_harmonics

import fme
from fme.ace.aggregator.inference.spectrum import (
    PairedSphericalPowerSpectrumAggregator,
    SphericalPowerSpectrumAggregator,
    get_positive_and_negative_power_bias,
    get_smallest_scale_power_bias,
)
from fme.core.gridded_ops import LatLonOperations
from fme.core.metrics import spherical_power_spectrum

DEVICE = fme.get_device()


def get_gridded_operations(nlat: int, nlon: int):
    return LatLonOperations(torch.ones(nlat, nlon))


def test_spherical_power_spectrum_aggregator():
    nlat = 8
    nlon = 16
    grid = "legendre-gauss"
    gridded_operations = get_gridded_operations(nlat, nlon)
    agg = SphericalPowerSpectrumAggregator(gridded_operations)
    data = {"a": torch.randn(2, 2, nlat, nlon, device=fme.get_device())}
    data2 = {"a": torch.randn(2, 3, nlat, nlon, device=fme.get_device())}
    agg.record_batch(data)
    agg.record_batch(data2)
    result = agg.get_mean()
    assert "a" in result
    assert result["a"].shape == (nlat,)

    sht = torch_harmonics.RealSHT(nlat, nlon, grid=grid).to(fme.get_device())
    data_concat = torch.cat([data["a"], data2["a"]], dim=1)
    expected_value = torch.mean(spherical_power_spectrum(data_concat, sht), dim=(0, 1))
    torch.testing.assert_close(result["a"], expected_value)


@pytest.mark.parametrize("report_plot", [True, False])
def test_paired_spherical_power_spectrum_aggregator(report_plot: bool):
    nlat = 8
    nlon = 16
    gridded_operations = get_gridded_operations(nlat, nlon)
    agg = PairedSphericalPowerSpectrumAggregator(
        gridded_operations, report_plot=report_plot
    )
    data = {"a": torch.randn(2, 3, nlat, nlon, device=fme.get_device())}
    agg.record_batch(data, data, None, None)
    result = agg.get_logs("spectrum")
    if report_plot:
        assert isinstance(result["spectrum/a"], plt.Figure)
    else:
        assert "spectrum/a" not in result


@pytest.mark.parametrize(
    "gen_spectrum, target_spectrum, expected_positive_bias, expected_negative_bias",
    [
        pytest.param(
            torch.tensor([1.0, 1.2, 1.4], device=DEVICE),
            torch.tensor([1, 1, 1], device=DEVICE),
            0.2,
            0.0,
            id="positive_bias",
        ),
        pytest.param(
            torch.tensor([1.0, 0.8, 0.6], device=DEVICE),
            torch.tensor([1, 1, 1], device=DEVICE),
            0.0,
            -0.2,
            id="negative_bias",
        ),
        pytest.param(
            torch.tensor([1.6, 0.8, 0.6], device=DEVICE),
            torch.tensor([1, 1, 1], device=DEVICE),
            0.2,
            -0.2,
            id="both_bias",
        ),
        pytest.param(
            torch.tensor([2.0, 2.4, 2.8], device=DEVICE),
            torch.tensor([2, 2, 2], device=DEVICE),
            0.2,
            0.0,
            id="positive_bias_with_scale",
        ),
    ],
)
def test_get_positive_and_negative_power_bias(
    gen_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    expected_positive_bias: float,
    expected_negative_bias: float,
):
    positive_bias, negative_bias = get_positive_and_negative_power_bias(
        gen_spectrum, target_spectrum
    )
    torch.testing.assert_close(positive_bias, expected_positive_bias)
    torch.testing.assert_close(negative_bias, expected_negative_bias)


@pytest.mark.parametrize(
    "gen_spectrum, target_spectrum, expected_bias",
    [
        pytest.param(
            torch.tensor([1.0, 1.2, 1.4], device=DEVICE),
            torch.tensor([1, 1, 1], device=DEVICE),
            0.4,
            id="positive_bias",
        ),
        pytest.param(
            torch.tensor([1.0, 0.8, 1.3], device=DEVICE),
            torch.tensor([1, 1, 1.3], device=DEVICE),
            0.0,
            id="no_bias",
        ),
    ],
)
def test_get_smallest_scale_power_bias(
    gen_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    expected_bias: float,
):
    bias = get_smallest_scale_power_bias(gen_spectrum, target_spectrum)
    torch.testing.assert_close(bias, expected_bias)
