import datetime

import numpy as np
import torch
import xarray as xr

import fme
from fme.ace.aggregator.inference.data import InferenceBatchData, make_dummy_time
from fme.ace.aggregator.inference.residual_spectrum import (
    ResidualSpectrumAggregator,
    ResidualSpectrumMetricConfig,
)
from fme.ace.aggregator.inference.spectrum import PairedSphericalPowerSpectrumAggregator
from fme.core.gridded_ops import LatLonOperations

DEVICE = fme.get_device()


def _make_ops(nlat: int = 8, nlon: int = 16):
    return LatLonOperations(torch.ones(nlat, nlon, device=DEVICE))


def test_aggregator_produces_spectrum_logs():
    nlat, nlon = 8, 16
    ops = _make_ops(nlat, nlon)
    inner = PairedSphericalPowerSpectrumAggregator(
        gridded_operations=ops,
        report_plot=False,
    )
    agg = ResidualSpectrumAggregator(inner)
    n_sample, n_time = 2, 4
    batch = InferenceBatchData(
        prediction={"a": torch.randn(n_sample, n_time, nlat, nlon, device=DEVICE)},
        target={"a": torch.randn(n_sample, n_time, nlat, nlon, device=DEVICE)},
        time=make_dummy_time(n_sample, n_time),
        i_time_start=0,
    )
    agg.record_batch(batch)
    logs = agg.get_logs("test")
    assert "test/smallest_scale_norm_bias/a" in logs
    assert "test/positive_norm_bias/a" in logs
    assert "test/negative_norm_bias/a" in logs
    assert "test/mean_abs_norm_bias/a" in logs


def test_aggregator_skips_without_target():
    nlat, nlon = 8, 16
    ops = _make_ops(nlat, nlon)
    inner = PairedSphericalPowerSpectrumAggregator(
        gridded_operations=ops,
        report_plot=False,
    )
    agg = ResidualSpectrumAggregator(inner)
    batch = InferenceBatchData(
        prediction={"a": torch.randn(2, 3, nlat, nlon, device=DEVICE)},
        time=make_dummy_time(2, 3),
        i_time_start=0,
    )
    agg.record_batch(batch)
    assert agg.get_logs("test") == {}


def test_aggregator_spectrum_on_diffs_not_raw():
    """The spectrum should be computed on temporal differences, not raw fields."""
    nlat, nlon = 8, 16
    ops = _make_ops(nlat, nlon)

    inner_residual = PairedSphericalPowerSpectrumAggregator(
        gridded_operations=ops, report_plot=False
    )
    inner_raw = PairedSphericalPowerSpectrumAggregator(
        gridded_operations=ops, report_plot=False
    )
    agg = ResidualSpectrumAggregator(inner_residual)

    torch.manual_seed(0)
    n_sample, n_time = 2, 5
    data = {"a": torch.randn(n_sample, n_time, nlat, nlon, device=DEVICE)}

    agg.record_batch(
        InferenceBatchData(
            prediction=data,
            target=data,
            time=make_dummy_time(n_sample, n_time),
            i_time_start=0,
        )
    )
    inner_raw.record_paired_data(prediction=data, target=data)

    residual_spectrum = inner_residual._gen_aggregator.get_mean()
    raw_spectrum = inner_raw._gen_aggregator.get_mean()

    assert not torch.allclose(residual_spectrum["a"], raw_spectrum["a"])


def test_metric_config_build():
    from fme.ace.aggregator.inference.build_context import MetricBuildContext
    from fme.core.coordinates import LatLonCoordinates

    config = ResidualSpectrumMetricConfig()
    assert config.get_name() == "residual_spectrum"

    ops = _make_ops(8, 16)
    coords = LatLonCoordinates(
        lat=xr.DataArray(np.linspace(-90, 90, 8), dims=["lat"]),
        lon=xr.DataArray(np.linspace(0, 360, 16, endpoint=False), dims=["lon"]),
    )
    ctx = MetricBuildContext(
        ops=ops,
        horizontal_coordinates=coords,
        n_timesteps=10,
        n_ic_steps=1,
        timestep=datetime.timedelta(hours=6),
        variable_metadata=None,
        channel_mean_names=None,
        monthly_reference_data=None,
        time_mean_reference_data=None,
        initial_time=xr.DataArray([0]),
    )
    result = config.build(ctx)
    assert result.aggregator is not None


def test_metric_config_with_variable_filter():
    from fme.ace.aggregator.inference.build_context import MetricBuildContext
    from fme.core.coordinates import LatLonCoordinates

    config = ResidualSpectrumMetricConfig(variables=["u", "v"])
    ops = _make_ops(8, 16)
    coords = LatLonCoordinates(
        lat=xr.DataArray(np.linspace(-90, 90, 8), dims=["lat"]),
        lon=xr.DataArray(np.linspace(0, 360, 16, endpoint=False), dims=["lon"]),
    )
    ctx = MetricBuildContext(
        ops=ops,
        horizontal_coordinates=coords,
        n_timesteps=10,
        n_ic_steps=1,
        timestep=datetime.timedelta(hours=6),
        variable_metadata=None,
        channel_mean_names=None,
        monthly_reference_data=None,
        time_mean_reference_data=None,
        initial_time=xr.DataArray([0]),
    )
    result = config.build(ctx)
    assert result.aggregator is not None
