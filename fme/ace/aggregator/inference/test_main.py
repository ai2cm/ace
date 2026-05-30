import dataclasses
import datetime
from collections.abc import Sequence

import dacite
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.inference.main import (
    EnsembleMetricConfig,
    InferenceEvaluatorAggregatorConfig,
    InferenceEvaluatorAggregatorWithUncorrected,
    StepMeanMetricConfig,
    TimeMeanMetricConfig,
    build_inference_evaluator_aggregator,
)
from fme.ace.data_loading.batch_data import BatchData, PairedData
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.generics.aggregator import (
    InferenceAggregatorABC,
    InferenceLog,
    InferenceLogs,
)


def get_ds_info(nx: int, ny: int) -> DatasetInfo:
    coords = LatLonCoordinates(lon=torch.arange(nx), lat=torch.arange(ny))
    return DatasetInfo(
        horizontal_coordinates=coords, timestep=datetime.timedelta(hours=6)
    )


@pytest.mark.parametrize("n_ensemble", [1, 2])
@pytest.mark.parametrize("channel_mean_names", [None, ["a", "b"]])
def test_inference_evaluator_aggregator_channel_mean_names(
    n_ensemble: int,
    channel_mean_names: Sequence[str] | None,
):
    n_ic_steps = 1
    n_forward_steps = 2
    n_timesteps = n_ic_steps + n_forward_steps
    nx, ny = 4, 4
    batch_size = 2

    ds_info = get_ds_info(nx, ny)
    initial_time = xr.DataArray(np.zeros((batch_size * n_ensemble,)), dims=["sample"])

    agg = build_inference_evaluator_aggregator(
        metrics=[
            TimeMeanMetricConfig(target="norm"),
        ],
        dataset_info=ds_info,
        n_ic_steps=n_ic_steps,
        n_forward_steps=n_forward_steps,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        channel_mean_names=channel_mean_names,
        save_diagnostics=False,
        n_ensemble_per_ic=n_ensemble,
    )

    target_dict = {
        "a": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()),
        "b": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 3,
        "c": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 4,
    }
    gen_dict = {
        "a": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 2.0,
        "b": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 5,
        "c": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 6,
    }

    # Time must have shape (batch_size, n_timesteps) to match data; broadcast_ensemble
    # will expand both to (batch_size * n_ensemble, n_timesteps).
    time = xr.DataArray(np.zeros((batch_size, n_timesteps)), dims=["sample", "time"])

    target_batch = BatchData(data=target_dict, time=time)
    gen_batch = BatchData(data=gen_dict, time=time)

    target_batch = target_batch.broadcast_ensemble(n_ensemble=n_ensemble)
    gen_batch = gen_batch.broadcast_ensemble(n_ensemble=n_ensemble)

    paired_data = PairedData.from_batch_data(
        prediction=gen_batch, reference=target_batch
    )

    agg.record_batch(paired_data)

    summary_logs = agg.get_summary_logs()

    for varname in ["a", "b", "c"]:
        assert f"time_mean_norm/rmse/{varname}" in summary_logs

    assert "time_mean_norm/rmse/channel_mean" in summary_logs
    actual_channel_mean_rmse = summary_logs["time_mean_norm/rmse/channel_mean"]
    if channel_mean_names is None:
        expected_channel_mean_rmse = 5 / 3
    else:
        expected_channel_mean_rmse = 3 / 2
    np.testing.assert_allclose(
        actual_channel_mean_rmse,
        expected_channel_mean_rmse,
    )


def test_inference_evaluator_aggregator_ensemble():
    channel_mean_names = ["a", "b"]
    n_ic_steps = 1
    n_forward_steps = 39
    n_timesteps = n_ic_steps + n_forward_steps
    nx, ny = 4, 4
    batch_size = 2
    n_ensemble = 2

    ds_info = get_ds_info(nx, ny)
    initial_time = xr.DataArray(np.zeros((batch_size,)), dims=["sample"])

    agg = build_inference_evaluator_aggregator(
        metrics=[
            StepMeanMetricConfig(step=20, target="denorm"),
            StepMeanMetricConfig(step=20, target="norm"),
            TimeMeanMetricConfig(target="norm"),
            EnsembleMetricConfig(step=20, target="denorm"),
            EnsembleMetricConfig(step=20, target="norm"),
        ],
        dataset_info=ds_info,
        n_ic_steps=n_ic_steps,
        n_forward_steps=n_forward_steps,
        initial_time=initial_time,
        normalize=lambda x: {k: v * 0.5 for k, v in x.items()},
        channel_mean_names=channel_mean_names,
        save_diagnostics=False,
        n_ensemble_per_ic=n_ensemble,
    )

    target_data = BatchData.new_for_testing(
        names=["a", "b", "c"],
        n_samples=1,
        n_timesteps=n_timesteps,
        img_shape=(nx, ny),
    )

    gen_data = BatchData.new_for_testing(
        names=["a", "b", "c"],
        n_samples=1,
        n_timesteps=n_timesteps,
        img_shape=(nx, ny),
    )

    target_data = target_data.broadcast_ensemble(n_ensemble=n_ensemble)
    gen_data = gen_data.broadcast_ensemble(n_ensemble=n_ensemble)

    paired_data = PairedData.from_batch_data(prediction=gen_data, reference=target_data)
    agg.record_batch(paired_data)

    summary_logs = agg.get_summary_logs()
    for varname in ["a", "b", "c"]:
        assert f"ensemble_step_20/crps/{varname}" in summary_logs
        assert f"ensemble_step_20_norm/crps/{varname}" in summary_logs
    for varname in ["a", "b", "c"]:
        assert f"ensemble_step_20/ssr_bias/{varname}" in summary_logs
        assert f"ensemble_step_20_norm/ssr_bias/{varname}" in summary_logs
    # channel_mean is only emitted by the norm aggregator, and only for the
    # variables listed in channel_mean_names (["a", "b"], not "c").
    for metric in ("crps", "ensemble_mean_rmse"):
        assert f"ensemble_step_20_norm/{metric}/channel_mean" in summary_logs
        assert f"ensemble_step_20/{metric}/channel_mean" not in summary_logs
    # Differential: scaling by 0.5 changes crps/rmse, so denorm and norm
    # values must differ.
    for varname in ["a", "b", "c"]:
        for metric in ("crps", "ensemble_mean_rmse"):
            assert (
                summary_logs[f"ensemble_step_20/{metric}/{varname}"]
                != summary_logs[f"ensemble_step_20_norm/{metric}/{varname}"]
            )


def test_compute_uncorrected_metrics_defaults_on_and_round_trips():
    config = InferenceEvaluatorAggregatorConfig()
    assert config.compute_uncorrected_metrics is True
    disabled = InferenceEvaluatorAggregatorConfig(compute_uncorrected_metrics=False)
    restored = dacite.from_dict(
        InferenceEvaluatorAggregatorConfig,
        dataclasses.asdict(disabled),
        config=dacite.Config(strict=True),
    )
    assert restored.compute_uncorrected_metrics is False


def test_reduced_for_uncorrected_keeps_only_time_mean():
    reduced = InferenceEvaluatorAggregatorConfig().reduced_for_uncorrected()
    # Time-mean metrics remain enabled; the rest are disabled / cleared.
    assert reduced.time_mean_denorm.enabled
    assert reduced.time_mean_norm.enabled
    assert not reduced.mean_denorm.enabled
    assert not reduced.mean_norm.enabled
    assert not reduced.power_spectrum.enabled
    assert not reduced.zonal_mean.enabled
    assert not reduced.annual.enabled
    assert reduced.step_means == []
    assert reduced.ensembles == []
    # Avoid building a shadow-of-a-shadow.
    assert reduced.compute_uncorrected_metrics is False


class _StubAggregator(InferenceAggregatorABC):
    """Records calls and returns canned logs, for testing the composite."""

    def __init__(self, summary: InferenceLog, batch_logs: InferenceLogs):
        self._summary = summary
        self._batch_logs = batch_logs
        self.recorded_batches: list[PairedData] = []
        self.recorded_ic = 0
        self.flushed_subdirs: list[str | None] = []

    def record_batch(self, data: PairedData) -> InferenceLogs:
        self.recorded_batches.append(data)
        return [dict(log) for log in self._batch_logs]

    def record_initial_condition(self, initial_condition) -> InferenceLogs:
        self.recorded_ic += 1
        return [{}]

    def get_summary_logs(self) -> InferenceLog:
        return dict(self._summary)

    def flush_diagnostics(self, subdir: str | None = None) -> None:
        self.flushed_subdirs.append(subdir)


def _paired_with_uncorrected(uncorrected: dict | None) -> PairedData:
    time = xr.DataArray(np.zeros((1, 2)), dims=["sample", "time"])
    return PairedData(
        prediction={"a": torch.zeros(1, 2, 4, 4)},
        reference={"a": torch.zeros(1, 2, 4, 4)},
        time=time,
        uncorrected_prediction=uncorrected,
    )


def test_with_uncorrected_merges_and_prefixes_shadow_logs():
    main = _StubAggregator({"time_mean/rmse/a": 1.0}, [{"time_mean/rmse/a": 1.0}])
    shadow = _StubAggregator({"time_mean/rmse/a": 5.0}, [{"time_mean/rmse/a": 5.0}])
    agg = InferenceEvaluatorAggregatorWithUncorrected(main, shadow)

    data = _paired_with_uncorrected({"a": torch.ones(1, 2, 4, 4)})
    logs = agg.record_batch(data)
    assert logs[0]["time_mean/rmse/a"] == 1.0
    assert logs[0]["uncorrected/time_mean/rmse/a"] == 5.0
    # The shadow aggregator receives the uncorrected prediction tensors.
    assert len(shadow.recorded_batches) == 1
    torch.testing.assert_close(
        shadow.recorded_batches[0].prediction["a"], torch.ones(1, 2, 4, 4)
    )

    summary = agg.get_summary_logs()
    assert summary["time_mean/rmse/a"] == 1.0
    assert summary["uncorrected/time_mean/rmse/a"] == 5.0

    # The initial condition is only recorded into the main aggregator.
    agg.record_initial_condition(data)
    assert main.recorded_ic == 1
    assert shadow.recorded_ic == 0

    # Shadow diagnostics are routed to an "uncorrected" subdirectory.
    agg.flush_diagnostics(None)
    assert main.flushed_subdirs == [None]
    assert shadow.flushed_subdirs == ["uncorrected"]


def test_with_uncorrected_skips_shadow_when_absent():
    main = _StubAggregator({"time_mean/rmse/a": 1.0}, [{"time_mean/rmse/a": 1.0}])
    shadow = _StubAggregator({"time_mean/rmse/a": 5.0}, [{"time_mean/rmse/a": 5.0}])
    agg = InferenceEvaluatorAggregatorWithUncorrected(main, shadow)

    logs = agg.record_batch(_paired_with_uncorrected(None))
    assert logs[0]["time_mean/rmse/a"] == 1.0
    assert "uncorrected/time_mean/rmse/a" not in logs[0]
    assert len(shadow.recorded_batches) == 0

    # With nothing recorded, the shadow is neither summarized nor flushed, so a
    # shadow aggregator that would error on empty data is never touched.
    summary = agg.get_summary_logs()
    assert "uncorrected/time_mean/rmse/a" not in summary
    agg.flush_diagnostics(None)
    assert main.flushed_subdirs == [None]
    assert shadow.flushed_subdirs == []
