import logging

import os
import xarray as xr
import torch
import networks
import argparse
import numpy as np
import datetime
import sys
import config
from fcn_mip import initial_conditions, filesystem, schema
from modulus.distributed.manager import DistributedManager


def get_times():
    # the IFS data Jaideep downloaded only has 668 steps (up to end of november 2018)
    nsteps = 668
    times = [
        datetime.datetime(2018, 1, 1) + k * datetime.timedelta(hours=12)
        for k in range(nsteps)
    ]
    return times


class RMSE:
    def __init__(self, weight=None):
        self._xy = {}
        self.weight = weight

    def _mean(self, x):
        if self.weight is not None:
            x = self.weight * x
            denom = self.weight.mean(-1).mean(-1)
        else:
            denom = 1

        num = x.mean(0).mean(-1).mean(-1)
        return num / denom

    def call(self, truth, pred):
        xy = self._mean((truth - pred) ** 2)
        return xy.cpu()

    def gather(self, seq):
        return torch.sqrt(sum(seq) / len(seq))


class ACC:
    def __init__(self, mean, weight=None):
        self.mean = mean
        self._xy = {}
        self._xx = {}
        self._yy = {}
        self.weight = weight

    def _mean(self, x):
        if self.weight is not None:
            x = self.weight * x
            denom = self.weight.mean(-1).mean(-1)
        else:
            denom = 1

        num = x.mean(0).mean(-1).mean(-1)
        return num / denom

    def call(self, truth, pred):
        xx = self._mean((truth - self.mean) ** 2).cpu()
        yy = self._mean((pred - self.mean) ** 2).cpu()
        xy = self._mean((pred - self.mean) * (truth - self.mean)).cpu()
        return xx, yy, xy

    def gather(self, seq):
        """seq is an iterable of (xx, yy, xy) tuples"""
        # transpose seq
        xx, yy, xy = zip(*seq)

        xx = sum(xx)
        xy = sum(xy)
        yy = sum(yy)
        return xy / torch.sqrt(xx) / torch.sqrt(yy)


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("inference")


def flat_map(func, seq, *args):
    for x in seq:
        yield from func(x, *args)


def main(model_name, n, initial_times, device):
    model = networks.get_model(model_name)
    model.to(device)

    nlat = {schema.Grid.grid_720x1440: 720, schema.Grid.grid_721x1440: 721}[model.grid]
    time_mean_path = {
        schema.ChannelSet.var34: config.TIME_MEAN,
        schema.ChannelSet.var73: config.TIME_MEAN_73,
    }[model.channel_set]

    time_mean_path = filesystem.download_cached(time_mean_path)

    mean = np.load(time_mean_path)[0, model.channels, :nlat]
    mean = torch.from_numpy(mean).to(device)

    dt = datetime.timedelta(hours=6)

    ds = initial_conditions.get(
        n_history=0, time=initial_times[0], channel_set=model.channel_set
    )

    lat = np.deg2rad(ds.lat).values
    assert lat.ndim == 1
    weight = np.cos(lat)[:, np.newaxis]
    weight_torch = torch.from_numpy(weight).to(device)

    if model.grid == schema.Grid.grid_720x1440:
        weight_torch = weight_torch[:720, :]

    acc = ACC(mean, weight=weight_torch)
    metrics = {"acc": acc, "rmse": RMSE(weight=weight_torch)}
    model.to(device)

    def process(initial_time):
        logger.info(f"Running {initial_time}")
        initial_condition = initial_conditions.get(
            n_history=model.n_history, time=initial_time, channel_set=model.channel_set
        ).values

        if model.grid == schema.Grid.grid_720x1440:
            x = torch.from_numpy(initial_condition[None, :, :, :720]).to(device)
        elif model.grid == schema.Grid.grid_721x1440:
            x = torch.from_numpy(initial_condition[None]).to(device)
        else:
            raise NotImplementedError(f"Grid {model.grid} not supported")

        logger.debug("Initial Condition Loaded.")

        lead_time = datetime.timedelta(days=0)
        for data in model.run_steps(x, n):
            lead_time = lead_time + dt
            valid_time = initial_time + lead_time
            logger.debug(f"{valid_time}")
            v = initial_conditions.get(
                n_history=0, time=valid_time, channel_set=model.channel_set
            )
            verification = v.values[:, model.channels, :nlat, :]
            verification_torch = torch.from_numpy(verification).to(device)

            output = {}
            for name, metric in metrics.items():
                output[name] = metric.call(verification_torch, data)
            yield (initial_time, lead_time), output

    # collect outputs for lead_times
    my_channels = np.array(ds.channel)[model.channels]
    return metrics, my_channels, list(flat_map(process, initial_times))


def gather(seq, metrics, model_name, channels, output):
    outputs_by_lead_time = {}
    initial_times = set()
    for (initial_time, lead_time), metric_values in seq:
        forecasts_at_lead_time = outputs_by_lead_time.setdefault(lead_time, [])
        forecasts_at_lead_time.append(metric_values)
        initial_times.add(initial_time)

    def to_dataset(metric, name):
        outputs = {
            k: [v[name] for v in snapshots]
            for k, snapshots in outputs_by_lead_time.items()
        }
        times, accs = zip(*outputs.items())
        times = list(times)
        acc_arr = [metric.gather(acc) for acc in accs]
        stacked = torch.stack(acc_arr, 0)
        stacked = stacked.cpu().numpy()
        return xr.DataArray(
            stacked,
            dims=["lead_time", "channel"],
            coords={"lead_time": times, "channel": channels},
        ).to_dataset(name=name)

    ds = xr.merge(to_dataset(metric, name) for name, metric in metrics.items())
    ds = ds.assign(
        initial_times=xr.DataArray(list(initial_times), dims=["initial_time"])
    )
    ds.attrs["model"] = model_name
    ds.attrs["history"] = " ".join(sys.argv)
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    ds.to_netcdf(output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    model = args.model
    DistributedManager.initialize()
    dist = DistributedManager()
    # torch.distributed.init_process_group("nccl")

    rank = dist.rank

    global_initial_times = get_times()

    local_initial_times = global_initial_times[dist.rank :: dist.world_size]
    if args.test:
        local_initial_times = local_initial_times[-dist.world_size :]

    device = dist.device

    metrics, channels, seq = main(
        model, n=args.n, device=device, initial_times=local_initial_times
    )

    output_list = [None] * dist.world_size
    torch.distributed.all_gather_object(output_list, seq)

    if dist.rank == 0:

        seq = []
        for item in output_list:
            seq.extend(item)
        gather(
            seq,
            metrics=metrics,
            model_name=model,
            channels=channels,
            output=args.output,
        )
