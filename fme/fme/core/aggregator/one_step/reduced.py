from typing import Dict, Mapping, Optional, Union

import torch
import xarray as xr

from fme.core import metrics
from fme.core.device import get_device
from fme.core.distributed import Distributed

from .reduced_metrics import AreaWeightedReducedMetric, ReducedMetric


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class MeanAggregator:
    """
    Aggregator for mean-reduced metrics.

    These are metrics such as means which reduce to a single float for each batch,
    and then can be averaged across batches to get a single float for the
    entire dataset. This is important because the aggregator uses the mean to combine
    metrics across batches and processors.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        target_time: int = 1,
    ):
        self._area_weights = area_weights
        self._shape_x = None
        self._shape_y = None
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())
        self._variable_metrics: Optional[Dict[str, Dict[str, ReducedMetric]]] = None
        self._target_time = target_time
        self._dist = Distributed.get_instance()

    def _get_variable_metrics(self, gen_data: Mapping[str, torch.Tensor]):
        if self._variable_metrics is None:
            self._variable_metrics = {
                "weighted_rmse": {},
                "weighted_bias": {},
                "weighted_grad_mag_percent_diff": {},
            }
            device = get_device()
            for key in gen_data:
                self._variable_metrics["weighted_rmse"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=metrics.root_mean_squared_error,
                )
                self._variable_metrics["weighted_bias"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=metrics.weighted_mean_bias,
                )
                self._variable_metrics["weighted_grad_mag_percent_diff"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=metrics.gradient_magnitude_percent_diff,
                )
        return self._variable_metrics

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
        self._loss += loss
        variable_metrics = self._get_variable_metrics(gen_data)
        time_dim = 1
        time_len = gen_data[list(gen_data.keys())[0]].shape[time_dim]
        target_time = self._target_time - i_time_start
        if target_time >= 0 and time_len > target_time:
            for name in gen_data.keys():
                target = target_data[name].select(dim=time_dim, index=target_time)
                gen = gen_data[name].select(dim=time_dim, index=target_time)
                for metric in variable_metrics:
                    variable_metrics[metric][name].record(
                        target=target,
                        gen=gen,
                    )
            # only increment n_batches if we actually recorded a batch
            self._n_batches += 1

    def _get_data(self):
        if self._variable_metrics is None or self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        data: Dict[str, Union[float, torch.Tensor]] = {
            "loss": self._loss / self._n_batches
        }
        for metric in self._variable_metrics:
            for key in self._variable_metrics[metric]:
                data[f"{metric}/{key}"] = (
                    self._variable_metrics[metric][key].get() / self._n_batches
                )
        for key in sorted(data.keys()):
            data[key] = float(self._dist.reduce_mean(data[key].detach()).cpu().numpy())
        return data

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return {
            f"{label}/{key}": data for key, data in sorted(self._get_data().items())
        }

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        data = self._get_data()
        data = {key.replace("/", "-"): data[key] for key in data}
        data_vars = {}
        for key, value in data.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)
