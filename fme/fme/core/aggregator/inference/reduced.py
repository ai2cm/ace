import dataclasses
from typing import Dict, List, Literal, Mapping, Optional, Protocol

import numpy as np
import torch
import xarray as xr

from fme.core import metrics
from fme.core.data_loading.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.metrics import Dimension
from fme.core.typing_ import TensorMapping
from fme.core.wandb import Table, WandB


@dataclasses.dataclass
class _SeriesData:
    metric_name: str
    var_name: str
    data: np.ndarray

    def get_wandb_key(self) -> str:
        return f"{self.metric_name}/{self.var_name}"

    def get_xarray_key(self) -> str:
        return f"{self.metric_name}-{self.var_name}"


def get_gen_shape(gen_data: TensorMapping):
    for name in gen_data:
        return gen_data[name].shape


class MeanMetric(Protocol):
    def record(self, target: torch.Tensor, gen: torch.Tensor, i_time_start: int):
        """
        Update metric for a batch of data.
        """
        ...

    def get(self) -> torch.Tensor:
        """
        Get the total metric value, not divided by number of recorded batches.
        """
        ...


class SingleTargetMeanMetric(Protocol):
    def record(self, tensor: torch.Tensor, i_time_start: int):
        """
        Update metric for a batch of data.
        """
        ...

    def get(self) -> torch.Tensor:
        """
        Get the total metric value, not divided by number of recorded batches.
        """
        ...


class AreaWeightedFunction(Protocol):
    """
    A function that computes a metric on the true and predicted values,
    weighted by area.
    """

    def __call__(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: Dimension = (),
    ) -> torch.Tensor:
        ...


class AreaWeightedSingleTargetFunction(Protocol):
    """
    A function that computes a metric on a single value, weighted by area.
    """

    def __call__(
        self,
        tensor: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: Dimension = (),
    ) -> torch.Tensor:
        ...


def compute_metric_on(
    source: Literal["gen", "target"], metric: AreaWeightedSingleTargetFunction
) -> AreaWeightedFunction:
    """Turns a single-target metric function
    (computed on only the generated or target data) into a function that takes in
    both the generated and target data as arguments, as required for the APIs
    which call generic metric functions.
    """

    def metric_wrapper(
        truth: torch.Tensor,
        predicted: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: Dimension = (),
    ) -> torch.Tensor:
        if source == "gen":
            return metric(predicted, weights=weights, dim=dim)
        elif source == "target":
            return metric(truth, weights=weights, dim=dim)

    return metric_wrapper


class AreaWeightedReducedMetric:
    """
    A wrapper around an area-weighted metric function.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        device: torch.device,
        compute_metric: AreaWeightedFunction,
        n_timesteps: int,
    ):
        self._area_weights = area_weights
        self._compute_metric = compute_metric
        self._total: Optional[torch.Tensor] = None
        self._n_batches = torch.zeros(
            n_timesteps, dtype=torch.int32, device=get_device()
        )
        self._device = device
        self._n_timesteps = n_timesteps

    def record(self, target: torch.Tensor, gen: torch.Tensor, i_time_start: int):
        """Add a batch of data to the metric.

        Args:
            target: Target data. Should have shape [batch, time, height, width].
            gen: Generated data. Should have shape [batch, time, height, width].
            i_time_start: The index of the first timestep in the batch.
        """
        time_dim = 1
        if target.shape[time_dim] >= gen.shape[time_dim]:
            new_value = self._compute_metric(
                target, gen, weights=self._area_weights, dim=(-2, -1)
            ).mean(dim=0)
            if self._total is None:
                self._total = torch.zeros(
                    [self._n_timesteps], dtype=new_value.dtype, device=self._device
                )
            time_slice = slice(i_time_start, i_time_start + gen.shape[1])
            self._total[time_slice] += new_value
            self._n_batches[time_slice] += 1

    def get(self) -> torch.Tensor:
        """Returns the mean metric across recorded batches."""
        if self._total is None:
            return torch.tensor(torch.nan)
        return self._total / self._n_batches


class MeanAggregator:
    def __init__(
        self,
        area_weights: torch.Tensor,
        target: Literal["norm", "denorm"],
        n_timesteps: int,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        self._area_weights = area_weights
        self._variable_metrics: Optional[Dict[str, Dict[str, MeanMetric]]] = None
        self._shape_x = None
        self._shape_y = None
        self._target = target
        self._n_timesteps = n_timesteps

        self._dist = Distributed.get_instance()
        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata

    def _get_variable_metrics(self, gen_data: TensorMapping):
        if self._variable_metrics is None:
            self._variable_metrics = {
                "weighted_rmse": {},
                "weighted_mean_gen": {},
                "weighted_mean_target": {},
                "weighted_bias": {},
                "weighted_std_gen": {},
            }

            if self._target == "denorm":
                # redundant for the "norm" case
                self._variable_metrics["weighted_grad_mag_percent_diff"] = {}

            device = get_device()
            for key in gen_data:
                self._variable_metrics["weighted_rmse"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=metrics.root_mean_squared_error,
                    n_timesteps=self._n_timesteps,
                )
                if self._target == "denorm":
                    self._variable_metrics["weighted_grad_mag_percent_diff"][
                        key
                    ] = AreaWeightedReducedMetric(
                        area_weights=self._area_weights,
                        device=device,
                        compute_metric=metrics.gradient_magnitude_percent_diff,
                        n_timesteps=self._n_timesteps,
                    )
                self._variable_metrics["weighted_mean_gen"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=compute_metric_on(
                        source="gen", metric=metrics.weighted_mean
                    ),
                    n_timesteps=self._n_timesteps,
                )
                self._variable_metrics["weighted_mean_target"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=compute_metric_on(
                        source="target", metric=metrics.weighted_mean
                    ),
                    n_timesteps=self._n_timesteps,
                )
                self._variable_metrics["weighted_bias"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=metrics.weighted_mean_bias,
                    n_timesteps=self._n_timesteps,
                )
                self._variable_metrics["weighted_std_gen"][
                    key
                ] = AreaWeightedReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=compute_metric_on(
                        source="gen", metric=metrics.weighted_std
                    ),
                    n_timesteps=self._n_timesteps,
                )

        return self._variable_metrics

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        i_time_start: int = 0,
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
                    i_time_start=i_time_start,
                )

    def _get_series_data(self) -> List[_SeriesData]:
        """Converts internally stored variable_metrics to a list."""
        if self._variable_metrics is None:
            raise ValueError("No batches have been recorded.")
        data: List[_SeriesData] = []
        for metric in self._variable_metrics:
            sorted_keys = sorted(list(self._variable_metrics[metric].keys()))
            for key in sorted_keys:
                arr = self._variable_metrics[metric][key].get().detach()
                datum = _SeriesData(
                    metric_name=metric,
                    var_name=key,
                    data=self._dist.reduce_mean(arr).cpu().numpy(),
                )
                data.append(datum)
        return data

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {}
        series_data: Dict[str, np.ndarray] = {
            datum.get_wandb_key(): datum.data for datum in self._get_series_data()
        }
        table = data_to_table(series_data)
        logs[f"{label}/series"] = table
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        """
        Returns a dataset representation of the logs.
        """
        data_vars = {}
        for datum in self._get_series_data():
            metadata = self._metadata.get(
                datum.var_name, VariableMetadata("unknown_units", datum.var_name)
            )
            data_vars[datum.get_xarray_key()] = xr.DataArray(
                datum.data, dims=["forecast_step"], attrs=metadata._asdict()
            )

        n_forecast_steps = len(next(iter(data_vars.values())))
        coords = {"forecast_step": np.arange(n_forecast_steps)}
        return xr.Dataset(data_vars=data_vars, coords=coords)


def data_to_table(data: Dict[str, np.ndarray]) -> Table:
    """
    Convert a dictionary of 1-dimensional timeseries data to a wandb Table.
    """
    keys = sorted(list(data.keys()))
    wandb = WandB.get_instance()
    table = wandb.Table(columns=["forecast_step"] + keys)
    for i in range(len(data[keys[0]])):
        row = [i]
        for key in keys:
            row.append(data[key][i])
        table.add_data(*row)
    return table


class AreaWeightedSingleTargetReducedMetric:
    """
    A wrapper around an area-weighted metric function on a single data source.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        device: torch.device,
        compute_metric: AreaWeightedSingleTargetFunction,
        n_timesteps: int,
    ):
        self._area_weights = area_weights
        self._compute_metric = compute_metric
        self._total: Optional[torch.Tensor] = None
        self._n_batches = torch.zeros(
            n_timesteps, dtype=torch.int32, device=get_device()
        )
        self._device = device
        self._n_timesteps = n_timesteps

    def record(self, tensor: torch.Tensor, i_time_start: int):
        """Add a batch of data to the metric.

        Args:
            tensor: batch data. Should have shape [batch, time, height, width].
            i_time_start: The index of the first timestep in the batch.
        """
        new_value = self._compute_metric(
            tensor, weights=self._area_weights, dim=(-2, -1)
        ).mean(dim=0)
        if self._total is None:
            self._total = torch.zeros(
                [self._n_timesteps], dtype=new_value.dtype, device=self._device
            )
        time_slice = slice(i_time_start, i_time_start + tensor.shape[1])
        self._total[time_slice] += new_value
        self._n_batches[time_slice] += 1

    def get(self) -> torch.Tensor:
        """Returns the mean metric across recorded batches."""
        if self._total is None:
            return torch.tensor(torch.nan)
        return self._total / self._n_batches


class SingleTargetMeanAggregator:
    def __init__(
        self,
        area_weights: torch.Tensor,
        n_timesteps: int,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        self._area_weights = area_weights
        self._variable_metrics: Optional[
            Dict[str, Dict[str, SingleTargetMeanMetric]]
        ] = None
        self._shape_x = None
        self._shape_y = None
        self._n_timesteps = n_timesteps

        self._dist = Distributed.get_instance()
        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata

    def _get_variable_metrics(self, gen_data: TensorMapping):
        if self._variable_metrics is None:
            self._variable_metrics = {
                "weighted_mean_gen": {},
                "weighted_std_gen": {},
            }

            device = get_device()
            for key in gen_data:
                self._variable_metrics["weighted_mean_gen"][
                    key
                ] = AreaWeightedSingleTargetReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=metrics.weighted_mean,
                    n_timesteps=self._n_timesteps,
                )
                self._variable_metrics["weighted_std_gen"][
                    key
                ] = AreaWeightedSingleTargetReducedMetric(
                    area_weights=self._area_weights,
                    device=device,
                    compute_metric=metrics.weighted_std,
                    n_timesteps=self._n_timesteps,
                )

        return self._variable_metrics

    @torch.no_grad()
    def record_batch(
        self,
        data: TensorMapping,
        i_time_start: int = 0,
    ):
        variable_metrics = self._get_variable_metrics(data)
        for name in data.keys():
            for metric in variable_metrics:
                variable_metrics[metric][name].record(
                    tensor=data[name],
                    i_time_start=i_time_start,
                )

    def _get_series_data(self) -> List[_SeriesData]:
        """Converts internally stored variable_metrics to a list."""
        if self._variable_metrics is None:
            raise ValueError("No batches have been recorded.")
        data: List[_SeriesData] = []
        for metric in self._variable_metrics:
            sorted_keys = sorted(list(self._variable_metrics[metric].keys()))
            for key in sorted_keys:
                arr = self._variable_metrics[metric][key].get().detach()
                datum = _SeriesData(
                    metric_name=metric,
                    var_name=key,
                    data=self._dist.reduce_mean(arr).cpu().numpy(),
                )
                data.append(datum)
        return data

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {}
        series_data: Dict[str, np.ndarray] = {
            datum.get_wandb_key(): datum.data for datum in self._get_series_data()
        }
        table = data_to_table(series_data)
        logs[f"{label}/series"] = table
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        """
        Returns a dataset representation of the logs.
        """
        data_vars = {}
        for datum in self._get_series_data():
            metadata = self._metadata.get(
                datum.var_name, VariableMetadata("unknown_units", datum.var_name)
            )
            data_vars[datum.get_xarray_key()] = xr.DataArray(
                datum.data, dims=["forecast_step"], attrs=metadata._asdict()
            )

        n_forecast_steps = len(next(iter(data_vars.values())))
        coords = {"forecast_step": np.arange(n_forecast_steps)}
        return xr.Dataset(data_vars=data_vars, coords=coords)
