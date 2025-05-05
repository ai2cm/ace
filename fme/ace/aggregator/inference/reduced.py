import dataclasses
from collections import defaultdict
from collections.abc import Mapping
from typing import Literal, Protocol

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping
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
    def record(self, target: TensorMapping, gen: TensorMapping, i_time_start: int):
        """
        Update metric for a batch of data.
        """
        ...

    def get(self) -> TensorDict:
        """
        Get the total metric value per variable,
        not divided by number of recorded batches.
        """
        ...


class SingleTargetMeanMetric(Protocol):
    def record(self, tensors: TensorMapping, i_time_start: int):
        """
        Update metric for a batch of data.
        """
        ...

    def get(self) -> TensorDict:
        """
        Get the total metric value per variable,
        not divided by number of recorded batches.
        """
        ...


class AreaWeightedFunction(Protocol):
    """
    A function that computes a metric on the true and predicted values,
    weighted by area.
    """

    def __call__(
        self,
        truth: TensorMapping,
        predicted: TensorMapping,
    ) -> TensorDict: ...


class AreaWeightedSingleTargetFunction(Protocol):
    """
    A function that computes a metric on a single value, weighted by area.
    """

    def __call__(
        self,
        tensors: TensorMapping,
    ) -> TensorDict: ...


def compute_metric_on(
    source: Literal["gen", "target"], metric: AreaWeightedSingleTargetFunction
) -> AreaWeightedFunction:
    """Turns a single-target metric function
    (computed on only the generated or target data) into a function that takes in
    both the generated and target data as arguments, as required for the APIs
    which call generic metric functions.
    """

    def metric_wrapper(
        truth: TensorMapping,
        predicted: TensorMapping,
    ) -> TensorDict:
        if source == "gen":
            return metric(predicted)
        elif source == "target":
            return metric(truth)

    return metric_wrapper


class AreaWeightedReducedMetric:
    """
    A wrapper around an area-weighted metric function.
    """

    def __init__(
        self,
        device: torch.device,
        compute_metric: AreaWeightedFunction,
        n_timesteps: int,
    ):
        self._compute_metric = compute_metric
        self._total: TensorDict = {}
        self._n_batches = torch.zeros(
            n_timesteps, dtype=torch.int32, device=get_device()
        )
        self._device = device
        self._n_timesteps = n_timesteps

    def record(self, target: TensorMapping, gen: TensorMapping, i_time_start: int):
        """Add a batch of data to the metric.

        Args:
            target: Target data. Dictionary mapping variable names to tensors of shape
                [batch, time, height, width].
            gen: Generated data. Dictionary mapping variable names to tensors of shape
                [batch, time, height, width].
            i_time_start: The index of the first timestep in the batch.
        """
        time_dim = 1
        for name in target:
            if target[name].shape != gen[name].shape:
                raise RuntimeError(
                    "Tensors in target and gen must have the same shape, but got "
                    f"{target[name].shape} and {gen[name].shape} "
                    f"for the tensor '{name}'."
                )

        time_dim_len = next(iter(gen.values())).shape[time_dim]
        time_slice = slice(i_time_start, i_time_start + time_dim_len)

        # Update totals for each variable
        new_values = self._compute_metric(truth=target, predicted=gen)
        for name, tensor in new_values.items():
            if name not in self._total:
                self._total[name] = torch.zeros(
                    [self._n_timesteps], dtype=tensor.dtype, device=self._device
                )
            new_value = tensor.mean(dim=0)
            self._total[name][time_slice] += new_value

        self._n_batches[time_slice] += 1

    def get(self) -> TensorDict:
        """Returns the mean metric across recorded batches for each variable."""
        if not self._total:
            # no batches recorded yet
            return defaultdict(lambda: torch.tensor(torch.nan))
        return {name: tensor / self._n_batches for name, tensor in self._total.items()}


class MeanAggregator:
    def __init__(
        self,
        gridded_operations: GriddedOperations,
        target: Literal["norm", "denorm"],
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._gridded_operations = gridded_operations
        # Store one metric object per metric type (e.g., rmse, bias)
        self._target = target
        self._n_timesteps = n_timesteps

        self._dist = Distributed.get_instance()
        if variable_metadata is None:
            self._variable_metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._variable_metadata = variable_metadata

        self._variable_metrics: dict[str, MeanMetric] = {}
        device = get_device()

        self._variable_metrics["weighted_rmse"] = AreaWeightedReducedMetric(
            device=device,
            compute_metric=self._gridded_operations.area_weighted_rmse_dict,
            n_timesteps=self._n_timesteps,
        )
        if self._target == "denorm":
            self._variable_metrics["weighted_grad_mag_percent_diff"] = (
                AreaWeightedReducedMetric(
                    device=device,
                    compute_metric=self._gridded_operations.area_weighted_gradient_magnitude_percent_diff_dict,  # noqa: E501
                    n_timesteps=self._n_timesteps,
                )
            )
        self._variable_metrics["weighted_mean_gen"] = AreaWeightedReducedMetric(
            device=device,
            compute_metric=compute_metric_on(
                source="gen",
                metric=(
                    lambda tensors: self._gridded_operations.area_weighted_mean_dict(
                        tensors
                    )
                ),
            ),
            n_timesteps=self._n_timesteps,
        )
        self._variable_metrics["weighted_mean_target"] = AreaWeightedReducedMetric(
            device=device,
            compute_metric=compute_metric_on(
                source="target",
                metric=(
                    lambda tensors: self._gridded_operations.area_weighted_mean_dict(
                        tensors
                    )
                ),
            ),
            n_timesteps=self._n_timesteps,
        )
        self._variable_metrics["weighted_bias"] = AreaWeightedReducedMetric(
            device=device,
            compute_metric=self._gridded_operations.area_weighted_mean_bias_dict,
            n_timesteps=self._n_timesteps,
        )
        self._variable_metrics["weighted_std_gen"] = AreaWeightedReducedMetric(
            device=device,
            compute_metric=compute_metric_on(
                source="gen",
                metric=(
                    lambda tensors: self._gridded_operations.area_weighted_std_dict(
                        tensors
                    )
                ),
            ),
            n_timesteps=self._n_timesteps,
        )
        self._n_batches = 0

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        i_time_start: int = 0,
    ):
        if self._target == "norm":
            target_data = target_data_norm
            gen_data = gen_data_norm
        for metric in self._variable_metrics.values():
            metric.record(
                target=target_data,
                gen=gen_data,
                i_time_start=i_time_start,
            )
        self._n_batches += 1

    def _get_series_data(self, step_slice: slice | None = None) -> list[_SeriesData]:
        """Converts internally stored variable_metrics to a list."""
        if self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        data: list[_SeriesData] = []
        for name, metric in self._variable_metrics.items():
            metric_results = metric.get()  # TensorDict: {var_name: metric_series}
            sorted_keys = sorted(list(metric_results.keys()))
            for key in sorted_keys:
                arr = metric_results[key].detach()
                if step_slice is not None:
                    arr = arr[step_slice]
                datum = _SeriesData(
                    metric_name=name,
                    var_name=key,
                    data=self._dist.reduce_mean(arr).cpu().numpy(),
                )
                data.append(datum)
        return data

    @torch.no_grad()
    def get_logs(self, label: str, step_slice: slice | None = None):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
            step_slice: Slice of forecast steps to log.
        """
        logs = {}
        series_data: dict[str, np.ndarray] = {
            datum.get_wandb_key(): datum.data
            for datum in self._get_series_data(step_slice)
        }
        init_step = 0 if step_slice is None else step_slice.start
        table = data_to_table(series_data, init_step)
        logs[f"{label}/series"] = table
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        """
        Returns a dataset representation of the logs.
        """
        data_vars = {}
        for datum in self._get_series_data():
            metadata = self._variable_metadata.get(
                datum.var_name, VariableMetadata("unknown_units", datum.var_name)
            )
            data_vars[datum.get_xarray_key()] = xr.DataArray(
                datum.data, dims=["forecast_step"], attrs=metadata._asdict()
            )

        if len(data_vars.values()) > 0:
            n_forecast_steps = len(next(iter(data_vars.values())))
            coords = {"forecast_step": np.arange(n_forecast_steps)}
        else:
            coords = {"forecast_step": np.arange(0)}

        return xr.Dataset(data_vars=data_vars, coords=coords)


def data_to_table(data: dict[str, np.ndarray], init_step: int = 0) -> Table:
    """
    Convert a dictionary of 1-dimensional timeseries data to a wandb Table.

    Args:
        data: dictionary of timeseries data.
        init_step: initial step corresponding to the first row's "forecast_step"
    """
    keys = sorted(list(data.keys()))
    wandb = WandB.get_instance()
    table = wandb.Table(columns=["forecast_step"] + keys)
    if len(keys) > 0:
        for i in range(len(data[keys[0]])):
            row = [init_step + i]
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
        device: torch.device,
        compute_metric: AreaWeightedSingleTargetFunction,
        n_timesteps: int,
    ):
        self._compute_metric = compute_metric
        self._total: TensorDict = {}
        self._n_batches = torch.zeros(
            n_timesteps, dtype=torch.int32, device=get_device()
        )
        self._device = device
        self._n_timesteps = n_timesteps

    def record(self, tensors: TensorMapping, i_time_start: int):
        """Add a batch of data to the metric.

        Args:
            tensors: Dictionary mapping variable names to tensors of shape
                [batch, time, height, width].
            i_time_start: The index of the first timestep in the batch.
        """
        time_dim = 1

        time_dim_len = next(iter(tensors.values())).shape[time_dim]
        time_slice = slice(i_time_start, i_time_start + time_dim_len)

        # Update totals for each variable
        new_values = self._compute_metric(tensors)
        for name, tensor in new_values.items():
            if name not in self._total:
                self._total[name] = torch.zeros(
                    [self._n_timesteps], dtype=tensor.dtype, device=self._device
                )
            new_value = tensor.mean(dim=0)
            self._total[name][time_slice] += new_value

        self._n_batches[time_slice] += 1

    def get(self) -> TensorDict:
        """Returns the mean metric across recorded batches for each variable."""
        if not self._total:
            return defaultdict(lambda: torch.tensor(torch.nan))
        return {name: tensor / self._n_batches for name, tensor in self._total.items()}


class SingleTargetMeanAggregator:
    def __init__(
        self,
        gridded_operations: GriddedOperations,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._ops = gridded_operations
        self._n_timesteps = n_timesteps

        self._dist = Distributed.get_instance()
        if variable_metadata is None:
            self._variable_metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._variable_metadata = variable_metadata

        self._variable_metrics: dict[str, SingleTargetMeanMetric] = {}
        device = get_device()

        self._variable_metrics["weighted_mean_gen"] = (
            AreaWeightedSingleTargetReducedMetric(
                device=device,
                compute_metric=(
                    lambda tensors: self._ops.area_weighted_mean_dict(tensors)
                ),
                n_timesteps=self._n_timesteps,
            )
        )
        self._variable_metrics["weighted_std_gen"] = (
            AreaWeightedSingleTargetReducedMetric(
                device=device,
                compute_metric=(
                    lambda tensors: self._ops.area_weighted_std_dict(tensors)
                ),
                n_timesteps=self._n_timesteps,
            )
        )
        self._n_batches = 0

    @torch.no_grad()
    def record_batch(
        self,
        data: TensorMapping,
        i_time_start: int = 0,
    ):
        for metric in self._variable_metrics.values():
            metric.record(
                tensors=data,
                i_time_start=i_time_start,
            )
        self._n_batches += 1

    def _get_series_data(self, step_slice: slice | None = None) -> list[_SeriesData]:
        """Converts internally stored variable_metrics to a list."""
        if self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        data: list[_SeriesData] = []
        for name, metric in self._variable_metrics.items():
            metric_results = metric.get()  # TensorDict: {var_name: metric_series}
            sorted_keys = sorted(list(metric_results.keys()))
            for key in sorted_keys:
                arr = metric_results[key].detach()
                if step_slice is not None:
                    arr = arr[step_slice]
                datum = _SeriesData(
                    metric_name=name,
                    var_name=key,
                    data=self._dist.reduce_mean(arr).cpu().numpy(),
                )
                data.append(datum)
        return data

    @torch.no_grad()
    def get_logs(self, label: str, step_slice: slice | None = None):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
            step_slice: Slice of forecast steps to log.
        """
        logs = {}
        series_data: dict[str, np.ndarray] = {
            datum.get_wandb_key(): datum.data
            for datum in self._get_series_data(step_slice)
        }
        init_step = 0 if step_slice is None else step_slice.start
        table = data_to_table(series_data, init_step)
        logs[f"{label}/series"] = table
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        """
        Returns a dataset representation of the logs.
        """
        data_vars = {}
        for datum in self._get_series_data():
            metadata = self._variable_metadata.get(
                datum.var_name, VariableMetadata("unknown_units", datum.var_name)
            )
            data_vars[datum.get_xarray_key()] = xr.DataArray(
                datum.data, dims=["forecast_step"], attrs=metadata._asdict()
            )

        n_forecast_steps = len(next(iter(data_vars.values())))
        coords = {"forecast_step": np.arange(n_forecast_steps)}
        return xr.Dataset(data_vars=data_vars, coords=coords)
