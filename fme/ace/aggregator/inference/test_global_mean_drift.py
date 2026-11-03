import datetime
from collections.abc import Sequence

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PairedData
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.testing import mock_distributed

from .data import InferenceBatchData
from .global_mean_drift import GlobalMeanDriftAggregator, GlobalMeanDriftConfig
from .main import (
    InferenceEvaluatorAggregator,
    InferenceEvaluatorAggregatorConfig,
    MetricConfig,
    build_inference_evaluator_aggregator,
)
from .trend import SECONDS_PER_YEAR

N_LAT, N_LON = 3, 4
TIMESTEP = datetime.timedelta(days=1)


def _daily_time(n_sample: int, n_time: int, start_day: int = 0) -> xr.DataArray:
    base = cftime.DatetimeNoLeap(2000, 1, 1)
    times = [base + datetime.timedelta(days=start_day + i) for i in range(n_time)]
    return xr.DataArray([times for _ in range(n_sample)], dims=["sample", "time"])


def _drift_fields(
    drift_per_year: Sequence[float],
    time: xr.DataArray,
    names: Sequence[str] = ("a",),
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Random target; prediction is target + drift * (years since 2000)."""
    n_sample, n_time = time.shape
    base = cftime.DatetimeNoLeap(2000, 1, 1)
    days = np.array([[(t - base).days for t in row] for row in time.values])
    t = torch.tensor(days * 86400.0 / SECONDS_PER_YEAR, dtype=torch.float32)
    drift = torch.tensor(drift_per_year, dtype=torch.float32)[:, None]
    offset = (drift * t)[:, :, None, None].to(get_device())
    target = {
        name: torch.randn(n_sample, n_time, N_LAT, N_LON, device=get_device())
        for name in names
    }
    gen = {name: target[name] + offset for name in names}
    return target, gen


def _batch(
    drift_per_year: Sequence[float],
    time: xr.DataArray,
    i_time_start: int = 1,
    names: Sequence[str] = ("a",),
) -> InferenceBatchData:
    target, gen = _drift_fields(drift_per_year, time, names)
    return InferenceBatchData(
        prediction=gen, target=target, time=time, i_time_start=i_time_start
    )


def _aggregator() -> GlobalMeanDriftAggregator:
    ops = LatLonOperations(torch.ones(N_LAT, N_LON).to(get_device()))
    return GlobalMeanDriftAggregator(ops, timestep=TIMESTEP)


def _record_drift(
    drift_per_year: Sequence[float], n_days: int = 3 * 365, n_batches: int = 3
) -> GlobalMeanDriftAggregator:
    agg = _aggregator()
    n_sample = len(drift_per_year)
    per_batch = n_days // n_batches
    for i in range(n_batches):
        time = _daily_time(n_sample, per_batch, start_day=i * per_batch)
        agg.record_batch(_batch(drift_per_year, time, i_time_start=1 + i * per_batch))
    return agg


def test_recovers_per_sample_slope():
    torch.manual_seed(0)
    drift = [0.5, -1.0, 2.0]
    agg = _record_drift(drift)
    slopes = agg.get_per_sample_slopes()["a"]
    assert slopes.shape == (3,)
    np.testing.assert_allclose(slopes.cpu().numpy(), drift, atol=1e-4)
    logs = agg.get_logs(label="drift")
    assert logs["drift/slope/a"] == pytest.approx(np.mean(drift), abs=1e-4)


def test_streaming_matches_single_batch():
    torch.manual_seed(0)
    time = _daily_time(2, 3 * 365)
    batch = _batch([1.0, 2.0], time)
    single = _aggregator()
    single.record_batch(batch)
    streamed = _aggregator()
    for start, stop in [(0, 400), (400, 3 * 365)]:
        streamed.record_batch(
            InferenceBatchData(
                prediction={k: v[:, start:stop] for k, v in batch.prediction.items()},
                target={k: v[:, start:stop] for k, v in batch.target.items()},
                time=time.isel(time=slice(start, stop)),
                i_time_start=1 + start,
            )
        )
    xr.testing.assert_allclose(single.get_dataset(), streamed.get_dataset())


def test_t_statistic_large_for_consistent_drift():
    torch.manual_seed(0)
    drift = [1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99, 1.03]
    logs = _record_drift(drift).get_logs(label="")
    assert logs["t/a"] > 50.0
    assert logs["max_abs_t"] == pytest.approx(logs["t/a"])


def test_t_statistic_small_for_random_sign_drift():
    torch.manual_seed(0)
    drift = [1.0, -1.2, 0.8, -0.9, 1.1, -0.7, 0.9, -1.0]
    logs = _record_drift(drift).get_logs(label="")
    assert abs(logs["t/a"]) < 1.0


def test_max_abs_t_over_variables():
    torch.manual_seed(0)
    agg = _aggregator()
    time = _daily_time(4, 3 * 365)
    target_a, gen_a = _drift_fields([1.0, 1.1, 0.9, 1.0], time, names=("a",))
    target_b, gen_b = _drift_fields([-1.0, -1.01, -0.99, -1.0], time, names=("b",))
    agg.record_batch(
        InferenceBatchData(
            prediction={**gen_a, **gen_b},
            target={**target_a, **target_b},
            time=time,
            i_time_start=1,
        )
    )
    logs = agg.get_logs(label="")
    assert logs["t/b"] < 0
    assert logs["max_abs_t"] == pytest.approx(max(abs(logs["t/a"]), abs(logs["t/b"])))


def test_short_rollout_gives_nan():
    torch.manual_seed(0)
    agg = _record_drift([1.0, 2.0], n_days=365, n_batches=1)
    assert torch.isnan(agg.get_per_sample_slopes()["a"]).all()
    logs = agg.get_logs(label="")
    assert np.isnan(logs["slope/a"])
    assert np.isnan(logs["t/a"])
    assert np.isnan(logs["max_abs_t"])


def test_single_sample_gives_nan_t_statistic():
    torch.manual_seed(0)
    logs = _record_drift([1.5]).get_logs(label="")
    assert logs["slope/a"] == pytest.approx(1.5, abs=1e-4)
    assert np.isnan(logs["t/a"])
    assert np.isnan(logs["max_abs_t"])


def test_dataset_has_sample_variable_slopes():
    torch.manual_seed(0)
    agg = _aggregator()
    time = _daily_time(3, 3 * 365)
    agg.record_batch(_batch([1.0, 2.0, 3.0], time, names=("b", "a")))
    ds = agg.get_dataset()
    assert ds["slope"].dims == ("sample", "variable")
    assert list(ds["variable"].values) == ["a", "b"]
    np.testing.assert_allclose(
        ds["slope"].sel(variable="a").values, [1.0, 2.0, 3.0], atol=1e-4
    )


def test_samples_gathered_across_data_parallel_ranks():
    torch.manual_seed(0)
    with mock_distributed(world_size=2):
        agg = _record_drift([1.0, 2.0])
        slopes = agg.get_per_sample_slopes()["a"].cpu().numpy()
    np.testing.assert_allclose(slopes, [1.0, 2.0, 1.0, 2.0], atol=1e-4)


def test_enabled_in_default_metrics():
    names = [m.get_name() for m in InferenceEvaluatorAggregatorConfig()._get_metrics()]
    assert "global_mean_drift" in names


def _build_top_level(
    metrics: list[MetricConfig], n_forward_steps: int = 3 * 365
) -> InferenceEvaluatorAggregator:
    n_sample = 4
    return build_inference_evaluator_aggregator(
        metrics=metrics,
        dataset_info=DatasetInfo(
            horizontal_coordinates=LatLonCoordinates(
                lat=torch.linspace(-60, 60, N_LAT), lon=torch.arange(N_LON) * 90.0
            ),
            timestep=TIMESTEP,
        ),
        n_ic_steps=1,
        n_forward_steps=n_forward_steps,
        initial_time=xr.DataArray(np.zeros(n_sample), dims=["sample"]),
        normalize=lambda x, apply_mean=True: dict(x),
        save_diagnostics=False,
        raise_on_unsupported=False,
    )


def test_top_level_aggregator_logs_and_slopes():
    torch.manual_seed(0)
    drift = [1.0, 1.1, 0.9, 1.0]
    agg = _build_top_level([GlobalMeanDriftConfig()])
    time = _daily_time(len(drift), 3 * 365, start_day=1)
    target, gen = _drift_fields(drift, time, names=("a", "b"))
    agg.record_batch(
        PairedData.from_batch_data(
            prediction=BatchData(data=gen, time=time),
            reference=BatchData(data=target, time=time),
        )
    )
    logs = agg.get_summary().logs
    for key in [
        "global_mean_drift/slope/a",
        "global_mean_drift/slope/b",
        "global_mean_drift/t/a",
        "global_mean_drift/t/b",
        "global_mean_drift/max_abs_t",
    ]:
        assert key in logs
    assert logs["global_mean_drift/slope/a"] == pytest.approx(1.0, abs=1e-3)
    slopes = agg.get_global_mean_drift_slopes()
    assert slopes is not None
    np.testing.assert_allclose(slopes["a"].cpu().numpy(), drift, atol=1e-3)


@pytest.mark.parametrize(
    "metrics, n_forward_steps",
    [
        pytest.param([], 3 * 365, id="not_configured"),
        pytest.param([GlobalMeanDriftConfig()], 365, id="short_rollout"),
    ],
)
def test_top_level_slopes_none_when_metric_not_built(
    metrics: list[MetricConfig], n_forward_steps: int
):
    agg = _build_top_level(metrics, n_forward_steps=n_forward_steps)
    assert agg.get_global_mean_drift_slopes() is None
