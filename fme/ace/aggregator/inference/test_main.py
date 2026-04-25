import datetime
from collections.abc import Sequence

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.inference.main import (
    InferenceEvaluatorAggregator,
    StepMeanEntry,
)
from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
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

    agg = InferenceEvaluatorAggregator(
        dataset_info=ds_info,
        n_ic_steps=n_ic_steps,
        n_forward_steps=n_forward_steps,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        log_zonal_mean_images=False,
        log_step_means=[],
        log_video=False,
        log_seasonal_means=False,
        log_global_mean_time_series=False,
        log_global_mean_norm_time_series=False,
        log_histograms=False,
        channel_mean_names=channel_mean_names,
        log_nino34_index=False,
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
    n_ic_steps = 2
    n_forward_steps = 39
    nx, ny = 4, 4
    n_ensemble = 2

    ds_info = get_ds_info(nx, ny)
    initial_time = xr.DataArray(np.zeros((n_ensemble,)), dims=["sample"])

    agg = InferenceEvaluatorAggregator(
        dataset_info=ds_info,
        n_ic_steps=n_ic_steps,
        n_forward_steps=n_forward_steps,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        log_zonal_mean_images=False,
        log_video=False,
        log_seasonal_means=False,
        log_global_mean_time_series=False,
        log_global_mean_norm_time_series=False,
        log_histograms=False,
        channel_mean_names=channel_mean_names,
        log_nino34_index=False,
        save_diagnostics=False,
        n_ensemble_per_ic=n_ensemble,
        log_step_means=[StepMeanEntry(step=20)],
    )

    ic_data = BatchData.new_for_testing(
        names=["a", "b", "c"],
        n_samples=n_ensemble,
        n_timesteps=n_ic_steps,
        img_shape=(nx, ny),
    )
    agg.record_initial_condition(PrognosticState(ic_data))

    target_data = BatchData.new_for_testing(
        names=["a", "b", "c"],
        n_samples=n_ensemble,
        n_timesteps=n_forward_steps,
        img_shape=(nx, ny),
    )
    gen_data = BatchData.new_for_testing(
        names=["a", "b", "c"],
        n_samples=n_ensemble,
        n_timesteps=n_forward_steps,
        img_shape=(nx, ny),
    )
    gen_data.time = target_data.time

    paired_data = PairedData.from_batch_data(prediction=gen_data, reference=target_data)
    agg.record_batch(paired_data)

    summary_logs = agg.get_summary_logs()
    for varname in ["a", "b", "c"]:
        assert f"ensemble_mean_step_20/crps/{varname}" in summary_logs
        assert f"ensemble_mean_step_20/ssr_bias/{varname}" in summary_logs


def test_inference_evaluator_aggregator_ensemble_windowed():
    """Test ensemble metrics with windowed batches matching production flow,
    where data arrives in chunks across multiple record_batch calls."""
    n_ic_steps = 2
    n_forward_steps = 30
    nx, ny = 4, 4
    n_ensemble = 3
    window_size = 10

    ds_info = get_ds_info(nx, ny)
    initial_time = xr.DataArray(np.zeros((n_ensemble,)), dims=["sample"])

    agg = InferenceEvaluatorAggregator(
        dataset_info=ds_info,
        n_ic_steps=n_ic_steps,
        n_forward_steps=n_forward_steps,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        log_zonal_mean_images=False,
        log_video=False,
        log_seasonal_means=False,
        log_global_mean_time_series=False,
        log_global_mean_norm_time_series=False,
        log_histograms=False,
        channel_mean_names=["a", "b"],
        log_nino34_index=False,
        save_diagnostics=False,
        n_ensemble_per_ic=n_ensemble,
        log_step_means=[StepMeanEntry(step=20)],
    )

    ic_data = BatchData.new_for_testing(
        names=["a", "b", "c"],
        n_samples=n_ensemble,
        n_timesteps=n_ic_steps,
        img_shape=(nx, ny),
    )
    agg.record_initial_condition(PrognosticState(ic_data))

    for i_start in range(0, n_forward_steps, window_size):
        i_end = min(i_start + window_size, n_forward_steps)
        chunk_size = i_end - i_start
        target = BatchData.new_for_testing(
            names=["a", "b", "c"],
            n_samples=n_ensemble,
            n_timesteps=chunk_size,
            img_shape=(nx, ny),
        )
        gen = BatchData.new_for_testing(
            names=["a", "b", "c"],
            n_samples=n_ensemble,
            n_timesteps=chunk_size,
            img_shape=(nx, ny),
        )
        gen.time = target.time
        paired = PairedData.from_batch_data(prediction=gen, reference=target)
        agg.record_batch(paired)

    summary_logs = agg.get_summary_logs()
    for varname in ["a", "b", "c"]:
        key_crps = f"ensemble_mean_step_20/crps/{varname}"
        key_ssr = f"ensemble_mean_step_20/ssr_bias/{varname}"
        assert key_crps in summary_logs, f"Missing {key_crps}"
        assert key_ssr in summary_logs, f"Missing {key_ssr}"
        assert np.isfinite(
            summary_logs[key_ssr]
        ), f"{key_ssr} is not finite: {summary_logs[key_ssr]}"
