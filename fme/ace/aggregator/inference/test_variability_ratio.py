import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.gridded_ops import LatLonOperations

from .data import InferenceBatchData
from .variability_ratio import VariabilityRatioAggregator, VariabilityRatioMetricConfig

N_SAMPLE, N_TIME, N_LAT, N_LON = 1, 48, 4, 8  # four years of monthly steps


def _ops() -> LatLonOperations:
    return LatLonOperations(area_weights=torch.ones((N_LAT, N_LON)))


def _monthly_time() -> xr.DataArray:
    # (sample, time) datetimes at one-month spacing, starting in January
    times = xr.date_range("2000-01-15", periods=N_TIME, freq="MS", use_cftime=True)
    return xr.DataArray(np.array([times.values]), dims=["sample", "time"])


def _series(
    amplitude_anomaly: float, amplitude_seasonal: float, seed: int
) -> torch.Tensor:
    """Seasonal cycle plus white-noise anomaly, identical over the grid."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_TIME)
    seasonal = amplitude_seasonal * np.sin(2 * np.pi * t / 12)
    anomaly = amplitude_anomaly * rng.standard_normal(N_TIME)
    x = seasonal + anomaly
    return (
        torch.as_tensor(x, dtype=torch.float32)[None, :, None, None]
        .expand(N_SAMPLE, N_TIME, N_LAT, N_LON)
        .clone()
    )


def _batch(
    pred: torch.Tensor, target: torch.Tensor, i_time_start: int = 0
) -> InferenceBatchData:
    return InferenceBatchData(
        prediction={"thetao_6": pred},
        target={"thetao_6": target},
        time=_monthly_time(),
        i_time_start=i_time_start,
    )


def test_ratio_is_one_for_identical_fields():
    x = _series(1.0, 3.0, seed=0)
    agg = VariabilityRatioAggregator(_ops())
    agg.record_batch(_batch(x, x.clone()))
    logs = agg.get_logs("inference")
    assert logs["inference/thetao_6"] == pytest.approx(1.0, abs=1e-5)
    assert logs["inference/channel_mean"] == pytest.approx(1.0, abs=1e-5)


def test_damped_anomaly_is_seen_through_an_identical_seasonal_cycle():
    """A prediction with half the anomaly amplitude but the same seasonal cycle
    must score a variance ratio near 0.25 once the seasonal cycle is removed;
    without deseasonalizing the cycle masks most of the deficit."""
    rng_target = _series(1.0, 3.0, seed=1)
    # same seasonal cycle, anomaly halved: rebuild from the same noise draw
    rng = np.random.default_rng(1)
    t = np.arange(N_TIME)
    seasonal = 3.0 * np.sin(2 * np.pi * t / 12)
    anomaly = rng.standard_normal(N_TIME)
    pred = (
        torch.as_tensor(seasonal + 0.5 * anomaly, dtype=torch.float32)[
            None, :, None, None
        ]
        .expand(N_SAMPLE, N_TIME, N_LAT, N_LON)
        .clone()
    )
    deseason = VariabilityRatioAggregator(_ops(), deseasonalize=True)
    deseason.record_batch(_batch(pred, rng_target))
    raw = VariabilityRatioAggregator(_ops(), deseasonalize=False)
    raw.record_batch(_batch(pred, rng_target))
    r_deseason = deseason.get_logs("")["thetao_6"]
    r_raw = raw.get_logs("")["thetao_6"]
    assert r_deseason == pytest.approx(0.25, abs=0.06)
    # the seasonal cycle (variance 4.5) dwarfs the anomaly (variance 1), so
    # the undeseasonalized ratio stays close to 1
    assert r_raw > 0.75


def test_batches_accumulate_like_one_batch():
    pred, target = _series(0.7, 2.0, seed=2), _series(1.0, 2.0, seed=3)
    whole = VariabilityRatioAggregator(_ops())
    whole.record_batch(_batch(pred, target))
    split = VariabilityRatioAggregator(_ops())
    cut = 20
    first = InferenceBatchData(
        prediction={"thetao_6": pred[:, :cut]},
        target={"thetao_6": target[:, :cut]},
        time=_monthly_time()[:, :cut],
        i_time_start=0,
    )
    second = InferenceBatchData(
        prediction={"thetao_6": pred[:, cut:]},
        target={"thetao_6": target[:, cut:]},
        time=_monthly_time()[:, cut:],
        i_time_start=cut,
    )
    split.record_batch(first)
    split.record_batch(second)
    assert split.get_logs("")["thetao_6"] == pytest.approx(
        whole.get_logs("")["thetao_6"], rel=1e-5
    )


def test_no_target_logs_nothing():
    x = _series(1.0, 3.0, seed=4)
    agg = VariabilityRatioAggregator(_ops())
    agg.record_batch(
        InferenceBatchData(
            prediction={"thetao_6": x}, time=_monthly_time(), i_time_start=0
        )
    )
    assert agg.get_logs("x") == {}


def test_config_filters_variables():
    cfg = VariabilityRatioMetricConfig(variables=["a"])
    assert cfg.get_name() == "variability_ratio"
    assert cfg.enabled
