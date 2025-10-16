import datetime
from collections.abc import Sequence

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.inference.main import InferenceEvaluatorAggregator
from fme.ace.data_loading.batch_data import PairedData
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device


def get_ds_info(nx: int, ny: int) -> DatasetInfo:
    coords = LatLonCoordinates(lon=torch.arange(nx), lat=torch.arange(ny))
    return DatasetInfo(
        horizontal_coordinates=coords, timestep=datetime.timedelta(hours=6)
    )


@pytest.mark.parametrize("channel_mean_names", [None, ["a", "b"]])
def test_inference_evaluator_aggregator_channel_mean_names(
    channel_mean_names: Sequence[str] | None,
):
    n_timesteps = 3
    nx, ny = 4, 4
    batch_size = 2

    ds_info = get_ds_info(nx, ny)
    initial_time = xr.DataArray(np.zeros((batch_size,)), dims=["sample"])

    agg = InferenceEvaluatorAggregator(
        dataset_info=ds_info,
        n_timesteps=n_timesteps,
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
    )

    target_data = {
        "a": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()),
        "b": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 3,
        "c": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 4,
    }
    gen_data = {
        "a": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 2.0,
        "b": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 5,
        "c": torch.ones([batch_size, n_timesteps, nx, ny], device=get_device()) * 6,
    }

    time = xr.DataArray(np.zeros((batch_size, n_timesteps)), dims=["sample", "time"])

    paired_data = PairedData.new_on_device(
        prediction=gen_data,
        reference=target_data,
        labels=[set() for _ in range(batch_size)],
        time=time,
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
