from typing import Dict, Mapping, MutableMapping, Optional

import torch
import xarray as xr

from fme.core import metrics
from fme.core.distributed import Distributed
from fme.core.wandb import WandB

wandb = WandB.get_instance()


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class TimeMeanAggregator:
    """Statistics and images on the time-mean state.

    This aggregator keeps track of the time-mean state, then computes
    statistics and images on that time-mean state when logs are retrieved.
    """

    _image_captions = {
        "bias_map": "{name} time-mean bias (generated - target)",
        "gen_map": "{name} time-mean generated",
    }

    def __init__(self, area_weights: torch.Tensor, dist: Optional[Distributed] = None):
        self._area_weights = area_weights
        self._target_data: Optional[Dict[str, torch.Tensor]] = None
        self._gen_data: Optional[Dict[str, torch.Tensor]] = None
        self._target_data_norm = None
        self._gen_data_norm = None
        self._n_batches = 0
        if dist is None:
            self._dist = Distributed.get_instance()
        else:
            self._dist = dist

    @staticmethod
    def _add_or_initialize_time_mean(
        maybe_dict: Optional[MutableMapping[str, torch.Tensor]],
        new_data: Mapping[str, torch.Tensor],
        ignore_initial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        time_dim = 1
        if ignore_initial:
            time_slice = slice(1, None)
        else:
            time_slice = slice(0, None)
        if maybe_dict is None:
            d: Dict[str, torch.Tensor] = {
                name: tensor[:, time_slice].mean(dim=time_dim)
                for name, tensor in new_data.items()
            }
        else:
            d = dict(maybe_dict)
            for name, tensor in new_data.items():
                d[name] += tensor[:, time_slice].mean(dim=time_dim)
        return d

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        i_time_start: int = 0,
    ):
        ignore_initial = i_time_start == 0
        self._target_data = self._add_or_initialize_time_mean(
            self._target_data, target_data, ignore_initial
        )
        self._gen_data = self._add_or_initialize_time_mean(
            self._gen_data, gen_data, ignore_initial
        )

        # we can ignore time slicing and just treat segments as though they're
        # different batches, because we can assume all time segments have the
        # same length
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        if self._n_batches == 0 or self._gen_data is None or self._target_data is None:
            raise ValueError("No data recorded.")
        sample_dim = 0
        lat_dim = -2
        logs = {}
        for name in self._gen_data.keys():
            gen = self._dist.reduce_mean(self._gen_data[name] / self._n_batches)
            target = self._dist.reduce_mean(self._target_data[name] / self._n_batches)
            images = {
                "bias_map": (gen - target).mean(dim=sample_dim).cpu(),
                "gen_map": (gen).mean(dim=sample_dim).cpu(),
            }
            for key, data in images.items():
                caption = self._image_captions[key].format(name=name)
                caption += f" vmin={data.min():.4g}, vmax={data.max():.4g}."
                data = data.flip(dims=[lat_dim])
                wandb_image = wandb.Image(data, caption=caption)
                logs[f"{key}/{name}"] = wandb_image
            logs[f"rmse/{name}"] = float(
                metrics.root_mean_squared_error(
                    predicted=gen, truth=target, weights=self._area_weights
                )
                .cpu()
                .numpy()
            )
            logs[f"bias/{name}"] = float(
                metrics.time_and_global_mean_bias(
                    predicted=gen, truth=target, weights=self._area_weights
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
