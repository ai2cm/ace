"""
Score IFS

Wall time is around 30 minutes on a single selene node.
"""
import datetime
import json
import logging
from typing import Any, Mapping

import xarray
import h5py
import joblib
import numpy as np
import torch
import typer

import config
from datasets.era5 import METADATA
from fcn_mip import (
    inference_medium_range,
    initial_conditions,
    metrics,
    schema,
    filesystem,
)
from fcn_mip.report import scorecard


def loop_ifs(start_time: datetime.datetime, channel: str):
    print(start_time, channel)

    first_time_in_file = datetime.datetime(2018, 1, 1)
    initial_time_timestep = datetime.timedelta(hours=12)
    lead_time_interval = datetime.timedelta(hours=6)

    time_index = int((start_time - first_time_in_file) / initial_time_timestep)

    assert start_time.year == 2018, start_time
    ifs_channel_name = {"u10m": "u10", "v10m": "v10"}.get(channel, channel)
    path = f"{config.IFS_ROOT}/{ifs_channel_name}_2018.h5"
    logging.info(f"Opening {path}")

    with h5py.File(path) as f:
        first_array = list(f)[0]
        arr = f[first_array]
        for i in range(arr.shape[1]):
            lead_time = i * lead_time_interval
            valid_time = lead_time + start_time
            factor = 9.81 if channel == "z500" else 1.0
            yield valid_time, lead_time, arr[time_index, i] * factor


def generate(time, channel, metrics):
    for valid_time, lead_time, data in loop_ifs(time, channel):
        truth = initial_conditions.get(0, valid_time, schema.ChannelSet.var34)
        truth_channel = truth.sel(channel=channel).isel(lat=slice(0, -1))
        t = torch.from_numpy(truth_channel.values)
        p = torch.from_numpy(data)[None]
        output = [m.call(t, p) for m in metrics]
        yield lead_time, output
    print(f"Done with {time} {channel}")


class flat_map_parallel:
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    def __call__(self, func, seq, *args):
        for items in joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(lambda x, *args: list(func(x, *args)))(x, *args) for x in seq
        ):
            for x in items:
                yield x


def main(output: str, test: bool = False, n_jobs: int = 1):

    logging.basicConfig(level=logging.INFO)

    metadata = json.loads(METADATA.read_text())
    lat = metadata["coords"]["lat"][:-1]
    nlat = len(lat)
    cos_lat = np.cos(np.deg2rad(lat))[:, None]
    cos_lat_tensor = torch.from_numpy(cos_lat)

    time_mean_path = filesystem.download_cached(config.TIME_MEAN)
    mean = np.load(time_mean_path)[0, :, :nlat]
    mean = torch.from_numpy(mean)

    def get_acc(channel: str):
        c_idx = metadata["coords"]["channel"].index(channel)
        return metrics.ACC(mean=mean[c_idx], weight=cos_lat_tensor)

    metrics_d = {"rmse": lambda ch: metrics.RMSE(cos_lat_tensor), "acc": get_acc}

    times = inference_medium_range.get_times()
    flat_map = flat_map_parallel(n_jobs)
    channels = ["z500", "t2m", "t850", "v10m", "u10m"]

    if test:
        # a fast test setup
        channels = ["z500"]
        # non parallel
        flat_map = inference_medium_range.flat_map
        times = times[-1:]

    def get_scores(channel: str, metrics: Mapping[str, Any]):
        metric_values = [{} for _ in metrics]
        metric_for_this_channel = [m(channel) for m in metrics.values()]

        for lead_time, values in flat_map(
            generate, times, channel, metric_for_this_channel
        ):
            assert len(values) == len(metrics)
            for k, value in enumerate(values):
                ans = metric_values[k]
                array = ans.setdefault(lead_time, [])
                array.append(value)

        output = xarray.Dataset()
        for k, (name, metric) in enumerate(metrics.items()):
            ans = metric_values[k]
            out = {key: metric(channel).gather(values) for key, values in ans.items()}
            darray = xarray.DataArray(
                list(out.values()),
                dims=["lead_time"],
                coords={"lead_time": list(out.keys())},
            )
            output[name] = darray
        return output

    channel_dim = xarray.Variable(["channel"], channels)
    rmse = xarray.concat([get_scores(c, metrics_d) for c in channels], dim=channel_dim)
    rmse["initial_times"] = xarray.DataArray(times, dims=["initial_time"])
    rmse.attrs["model"] = "IFS"
    rmse = rmse.transpose(..., "lead_time", "channel")
    rmse.to_netcdf(output)

    # TODO refactor to test
    # If this fails there is some thing wrong with score-ifs.py
    # ensure that data is written in correct format
    assert scorecard.read([output], scorecard.CHANNELS, scorecard.LEAD_TIMES)["IFS"]


typer.run(main)
