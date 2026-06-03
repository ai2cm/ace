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
    StepMeanMetricConfig,
    TimeMeanMetricConfig,
    build_inference_evaluator_aggregator,
)
from fme.ace.data_loading.batch_data import BatchData, PairedData
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device


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


def _build_aggregator_with_uncorrected(
    ds_info: DatasetInfo,
    output_dir: str | None = None,
    save_diagnostics: bool = False,
):
    n_ic_steps = 1
    n_forward_steps = 2
    initial_time = xr.DataArray(np.zeros((2,)), dims=["sample"])
    return build_inference_evaluator_aggregator(
        metrics=[TimeMeanMetricConfig(target="norm")],
        uncorrected_metrics=[TimeMeanMetricConfig(target="norm")],
        dataset_info=ds_info,
        n_ic_steps=n_ic_steps,
        n_forward_steps=n_forward_steps,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=save_diagnostics,
        output_dir=output_dir,
    )


def _paired(prediction, target, uncorrected, nx=4, ny=4):
    batch_size, n_timesteps = 2, 3

    def field(value):
        return (
            torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * value
        )

    time = xr.DataArray(np.zeros((batch_size, n_timesteps)), dims=["sample", "time"])
    return PairedData(
        prediction={"a": field(prediction)},
        reference={"a": field(target)},
        time=time,
        uncorrected_prediction=(
            {"a": field(uncorrected)} if uncorrected is not None else {}
        ),
    )


def test_uncorrected_metrics_logged_under_prefix():
    # The aggregator computes time-mean metrics on both the corrected prediction
    # and the uncorrected prediction, the latter under an "uncorrected/" prefix.
    agg = _build_aggregator_with_uncorrected(get_ds_info(4, 4))
    # corrected pred=2 vs target=1 -> rmse 1; uncorrected pred=4 -> rmse 3.
    agg.record_batch(_paired(prediction=2.0, target=1.0, uncorrected=4.0))

    summary = agg.get_summary_logs()
    assert "time_mean_norm/rmse/a" in summary
    assert "uncorrected/time_mean_norm/rmse/a" in summary
    np.testing.assert_allclose(summary["time_mean_norm/rmse/a"], 1.0)
    np.testing.assert_allclose(summary["uncorrected/time_mean_norm/rmse/a"], 3.0)


def test_uncorrected_metrics_skipped_when_empty(tmp_path):
    # A corrector-less stepper produces an empty uncorrected prediction. The
    # uncorrected aggregators must then be neither summarized nor flushed (they
    # hold no data and would otherwise raise "No data recorded.").
    agg = _build_aggregator_with_uncorrected(
        get_ds_info(4, 4), output_dir=str(tmp_path), save_diagnostics=True
    )
    agg.record_batch(_paired(prediction=2.0, target=1.0, uncorrected=None))

    summary = agg.get_summary_logs()
    assert "time_mean_norm/rmse/a" in summary
    assert "uncorrected/time_mean_norm/rmse/a" not in summary
    # Must not raise even though the uncorrected aggregators recorded nothing.
    agg.flush_diagnostics(subdir=None)


def test_config_build_wires_uncorrected_metrics():
    # The config.build path (used by both the standalone evaluator and the
    # training inline-inference loop) wires uncorrected time-mean aggregators iff
    # compute_uncorrected_metrics is set.
    ds_info = get_ds_info(4, 4)
    initial_time = xr.DataArray(np.zeros((2,)), dims=["sample"])

    def build(compute_uncorrected: bool):
        return InferenceEvaluatorAggregatorConfig(
            compute_uncorrected_metrics=compute_uncorrected
        ).build(
            dataset_info=ds_info,
            n_ic_steps=1,
            n_forward_steps=2,
            initial_time=initial_time,
            normalize=lambda x: dict(x),
            save_diagnostics=False,
        )

    paired = _paired(prediction=2.0, target=1.0, uncorrected=4.0)

    agg_on = build(compute_uncorrected=True)
    step_logs = agg_on.record_batch(paired)
    assert "uncorrected/time_mean_norm/rmse/a" in agg_on.get_summary_logs()

    # Per-step time-series also appear for uncorrected mean metrics.
    all_step_keys: set[str] = set()
    for log_dict in step_logs:
        all_step_keys.update(log_dict.keys())
    assert "uncorrected/mean/weighted_mean_gen/a" in all_step_keys
    assert "uncorrected/mean_norm/weighted_mean_gen/a" in all_step_keys
    assert "uncorrected/mean/weighted_mean_target/a" not in all_step_keys
    assert "uncorrected/mean_norm/weighted_mean_target/a" not in all_step_keys

    agg_off = build(compute_uncorrected=False)
    agg_off.record_batch(paired)
    assert not any(k.startswith("uncorrected/") for k in agg_off.get_summary_logs())
