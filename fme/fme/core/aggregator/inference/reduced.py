from typing import Mapping, Optional, Dict, Literal, Protocol

import torch
import numpy as np

from fme.core import metrics
from fme.core.device import get_device
from fme.core.distributed import Distributed
from ..reduced_metrics import AreaWeightedReducedMetric, compute_metric_on
from fme.core.wandb import WandB
import xarray as xr

wandb = WandB.get_instance()


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class MeanMetric(Protocol):
    def record(self, target: torch.Tensor, gen: torch.Tensor):
        """
        Update metric for a batch of data.
        """
        ...

    def get(self) -> torch.Tensor:
        """
        Get the total metric value, not divided by number of recorded batches.
        """
        ...


class MeanAggregator:
    def __init__(self, area_weights: torch.Tensor, target: Literal["norm", "denorm"]):
        self._area_weights = area_weights
        self._n_batches = 0
        self._variable_metrics: Optional[Dict[str, Dict[str, MeanMetric]]] = None
        self._shape_x = None
        self._shape_y = None
        self._target = target

    def _get_variable_metrics(self, gen_data: Mapping[str, torch.Tensor]):
        if self._variable_metrics is None:
            self._variable_metrics = {
                "weighted_rmse": {},
                "weighted_grad_mag_percent_diff": {},
                "weighted_mean_gen": {},
                "weighted_bias": {},
            }
            device = get_device()
            area_weights = self._area_weights
            for key in gen_data:
                self._variable_metrics["weighted_rmse"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=metrics.root_mean_squared_error,
                )
                self._variable_metrics["weighted_grad_mag_percent_diff"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=metrics.gradient_magnitude_percent_diff,
                )
                self._variable_metrics["weighted_mean_gen"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=area_weights,
                    device=device,
                    compute_metric=compute_metric_on(
                        source="gen", metric=metrics.weighted_mean
                    ),
                )
                self._variable_metrics["weighted_bias"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=area_weights,
                    device=device,
                    compute_metric=metrics.weighted_mean_bias,
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
    ):
        if self._target == "norm":
            target_data = target_data_norm
            gen_data = gen_data_norm
        variable_metrics = self._get_variable_metrics(gen_data)
        for name in gen_data.keys():
            for metric in variable_metrics:
                variable_metrics[metric][name].record(
                    target=target_data[name],
                    gen=gen_data[name],
                )
        self._n_batches += 1

    def _get_series_data(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary of series data, where each key is a metric name.

        Used as a helper, as we present this series data in different ways.
        """
        if self._variable_metrics is None or self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        series_tensors: Dict[str, torch.Tensor] = {}
        for metric in self._variable_metrics:
            for key in self._variable_metrics[metric]:
                series_tensors[f"{metric}/{key}"] = (
                    self._variable_metrics[metric][key].get() / self._n_batches
                )
        dist = Distributed.get_instance()
        series_arrays: Dict[str, np.ndarray] = {}
        for key in sorted(series_tensors.keys()):
            series_arrays[key] = (
                dist.reduce_mean(series_tensors[key].detach()).cpu().numpy()
            )
        return series_arrays

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {}
        series_data = self._get_series_data()
        table = data_to_table(series_data)
        logs[f"{label}/series"] = table
        return logs

    @torch.no_grad()
    def get_dataset(self, label: str) -> xr.Dataset:
        """
        Returns a dataset representation of the logs.
        """
        if len(label) > 0:
            label = f"{label}_"
        series_data = self._get_series_data()
        series_data = {
            key.replace("/", "-"): value for key, value in series_data.items()
        }
        data_vars = {}
        for key in series_data:
            data_vars[f"{label}{key}"] = (["forecast_step"], series_data[key])
        coords = {"forecast_step": np.arange(len(series_data[key]))}
        return xr.Dataset(data_vars=data_vars, coords=coords)


def data_to_table(data: Dict[str, np.ndarray]):
    """
    Convert a dictionary of 1-dimensional timeseries data to a wandb Table.
    """
    keys = sorted(list(data.keys()))
    table = wandb.Table(columns=["forecast_step"] + keys)
    for i in range(len(data[keys[0]])):
        row = [i]
        for key in keys:
            row.append(data[key][i])
        table.add_data(*row)
    return table
