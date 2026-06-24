from collections.abc import Sequence

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.testing import mock_distributed

from .build_context import maybe_filter
from .data import InferenceBatchData, make_dummy_time
from .main import InferenceEvaluatorAggregatorConfig
from .trend import TrendEvaluatorAggregator, TrendMetricConfig, _years_since_reference

N_LAT, N_LON = 3, 4


def _ops() -> LatLonOperations:
    return LatLonOperations(torch.ones(N_LAT, N_LON).to(get_device()))


def _linear_batch(
    slope_target: float,
    slope_gen: float,
    *,
    intercept: float = 5.0,
    i_time_start: int = 1,
    time: xr.DataArray | None = None,
    names: Sequence[str] = ("a",),
    n_sample: int = 2,
    n_time: int = 8,
) -> InferenceBatchData:
    """Build a batch whose value at each grid cell is exactly linear in time,
    ``intercept + slope * t`` (t in years), so the recovered trend equals
    ``slope`` exactly."""
    if time is None:
        time = make_dummy_time(n_sample, n_time)
    n_sample, n_time = time.shape
    tau = torch.tensor(_years_since_reference(time), dtype=torch.float32)

    def field(slope: float) -> torch.Tensor:
        y = intercept + slope * tau  # (sample, time)
        return (
            y[:, :, None, None]
            .expand(n_sample, n_time, N_LAT, N_LON)
            .contiguous()
            .to(get_device())
        )

    target = {name: field(slope_target) for name in names}
    gen = {name: field(slope_gen) for name in names}
    return InferenceBatchData(
        prediction=gen,
        prediction_norm=gen,
        target=target,
        target_norm=target,
        time=time,
        i_time_start=i_time_start,
    )


def test_trend_recovers_known_slope():
    agg = TrendEvaluatorAggregator(_ops(), horizontal_dims=["lat", "lon"])
    agg.record_batch(_linear_batch(slope_target=2.0, slope_gen=-1.5))
    ds = agg.get_dataset()
    target = ds["a"].sel(source="target").values
    gen = ds["a"].sel(source="prediction").values
    np.testing.assert_allclose(target, 2.0, rtol=1e-4)
    np.testing.assert_allclose(gen, -1.5, rtol=1e-4)


def test_trend_logs_contain_maps_and_rmse():
    agg = TrendEvaluatorAggregator(_ops(), horizontal_dims=["lat", "lon"])
    agg.record_batch(_linear_batch(slope_target=2.0, slope_gen=-1.5))
    logs = agg.get_logs(label="trend")
    assert "trend/trend_maps/a" in logs
    assert "trend/trend_difference_map/a" in logs
    # constant trends differing by 3.5 everywhere -> area-weighted RMSE is 3.5
    assert logs["trend/rmse/a"] == pytest.approx(3.5, rel=1e-3)


def test_trend_streaming_matches_single_batch():
    """Splitting the time series across batches gives the same trend as
    recording it all at once -- the streaming property we rely on."""
    time = make_dummy_time(n_sample=2, n_time=10)
    single = TrendEvaluatorAggregator(_ops(), horizontal_dims=["lat", "lon"])
    single.record_batch(
        _linear_batch(slope_target=2.0, slope_gen=-1.0, time=time, i_time_start=0)
    )

    streamed = TrendEvaluatorAggregator(_ops(), horizontal_dims=["lat", "lon"])
    first = time.isel(time=slice(0, 4))
    second = time.isel(time=slice(4, None))
    streamed.record_batch(
        _linear_batch(slope_target=2.0, slope_gen=-1.0, time=first, i_time_start=0)
    )
    streamed.record_batch(
        _linear_batch(slope_target=2.0, slope_gen=-1.0, time=second, i_time_start=4)
    )

    xr.testing.assert_allclose(single.get_dataset(), streamed.get_dataset())


def test_trend_metrics_call_distributed():
    """All trend statistics must be reduced across processes."""
    with mock_distributed(0.0) as mock:
        agg = TrendEvaluatorAggregator(_ops(), horizontal_dims=["lat", "lon"])
        agg.record_batch(_linear_batch(slope_target=2.0, slope_gen=-1.0))
        agg.get_dataset()
        assert mock.reduce_called


def test_trend_variable_filtering():
    agg = maybe_filter(
        TrendEvaluatorAggregator(_ops(), horizontal_dims=["lat", "lon"]),
        ["a"],
    )
    agg.record_batch(_linear_batch(slope_target=2.0, slope_gen=-1.0, names=("a", "b")))
    ds = agg.get_dataset()
    assert list(ds.data_vars) == ["a"]


def test_trend_metric_config_disabled_by_default():
    assert TrendMetricConfig().enabled is False


def test_trend_excluded_from_default_metrics():
    names = [m.get_name() for m in InferenceEvaluatorAggregatorConfig()._get_metrics()]
    assert "trend" not in names


def test_trend_included_when_enabled():
    config = InferenceEvaluatorAggregatorConfig(
        trend=TrendMetricConfig(enabled=True, variables=["a"])
    )
    names = [m.get_name() for m in config._get_metrics()]
    assert "trend" in names
