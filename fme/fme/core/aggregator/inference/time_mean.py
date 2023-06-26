from typing import Mapping, Optional, Dict

import torch
import xarray as xr

from fme.core import metrics
from fme.core.device import get_device
from fme.core.distributed import Distributed


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class TimeMeanAggregator:
    """Statistics on the time-mean state.

    This aggregator keeps track of the time-mean state, then computes
    statistics on that time-mean state when logs are retrieved.
    """

    def __init__(self):
        self._target_data = None
        self._gen_data = None
        self._target_data_norm = None
        self._gen_data_norm = None
        self._n_batches = 0

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ):
        time_dim = 1

        def add_or_initialize_time_mean(
            maybe_dict: Optional[Dict[str, torch.Tensor]],
            new_data: Mapping[str, torch.Tensor],
        ) -> Mapping[str, torch.Tensor]:
            if maybe_dict is None:
                d: Dict[str, torch.Tensor] = {
                    name: tensor.mean(dim=time_dim) for name, tensor in new_data.items()
                }
            else:
                d = maybe_dict
                for name, tensor in new_data.items():
                    d[name] += tensor.mean(dim=time_dim)
            return d

        self._target_data = add_or_initialize_time_mean(self._target_data, target_data)
        self._gen_data = add_or_initialize_time_mean(self._gen_data, gen_data)

        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        if self._n_batches == 0:
            raise ValueError("No data recorded.")
        gen_shape = get_gen_shape(self._gen_data)
        area_weights = metrics.spherical_area_weights(
            num_lat=gen_shape[-2],
            num_lon=gen_shape[-1],
            device=get_device(),
        )
        logs = {}
        dist = Distributed.get_instance()
        for name in self._gen_data.keys():
            gen = dist.reduce_mean(self._gen_data[name] / self._n_batches)
            target = dist.reduce_mean(self._target_data[name] / self._n_batches)
            logs[f"rmse/{name}"] = float(
                metrics.root_mean_squared_error(
                    predicted=gen, truth=target, weights=area_weights
                )
                .cpu()
                .numpy()
            )
            logs[f"bias/{name}"] = float(
                metrics.time_and_global_mean_bias(
                    predicted=gen, truth=target, weights=area_weights
                )
                .cpu()
                .numpy()
            )
        return {f"{label}/{key}": logs[key] for key in logs}

    @torch.no_grad()
    def get_dataset(self, label: str) -> xr.Dataset:
        logs = self.get_logs(label=label)
        logs = {key.replace("/", "-"): logs[key] for key in logs}
        data_vars = {}
        for key, value in logs.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)
