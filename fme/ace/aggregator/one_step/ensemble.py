import abc
from collections.abc import Mapping

import torch
import xarray as xr

from fme.ace.aggregator.plotting import plot_paneled_data
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.ensemble import get_crps
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import EnsembleTensorDict, TensorMapping


def get_gen_shape(gen_data: TensorMapping):
    for name in gen_data:
        return gen_data[name].shape


def get_one_step_ensemble_aggregator(
    gridded_operations: GriddedOperations,
    target_time: int = 1,
    log_mean_maps: bool = True,
    metadata: Mapping[str, VariableMetadata] | None = None,
) -> "SelectStepEnsembleAggregator":
    return SelectStepEnsembleAggregator(
        aggregator=_EnsembleAggregator(
            gridded_operations=gridded_operations,
            log_mean_maps=log_mean_maps,
            metadata=metadata,
        ),
        i_target_time=target_time,
    )


class ReducedMetric(abc.ABC):
    """
    Used to record a metric value on batches of data (potentially out-of-memory)
    and then get the final metric at the end.
    """

    @abc.abstractmethod
    def record(self, target: torch.Tensor, gen: torch.Tensor):
        """
        Update metric for a batch of data.
        """
        ...

    @abc.abstractmethod
    def get(self) -> torch.Tensor:
        """
        Get the final metric value.
        """
        ...


class CRPSMetric(ReducedMetric):
    def __init__(self):
        self._total = None
        self._n_batches = 0

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        crps = get_crps(gen=gen, target=target, alpha=0.95).mean(dim=(0, 1))
        if self._total is None:
            self._total = crps
        else:
            self._total += crps
        self._n_batches += 1

    def get(self) -> torch.Tensor:
        if self._total is None:
            raise ValueError("No batches have been recorded.")
        return self._total / self._n_batches


class SSRBiasMetric(ReducedMetric):
    """
    Computes the spread-skill ratio bias (equal to (stdev / rmse) - 1).
    """

    def __init__(self):
        self._total_mse = None
        self._total_variance = None
        self._n_batches = 0

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        mse = ((gen - target) ** 2).mean(dim=(0, 1, 2))  # batch, ensemble, time
        variance = gen.var(dim=1, unbiased=True).mean(dim=(0, 1))
        self._add_mse(mse)
        self._add_variance(variance)
        self._n_batches += 1

    def _add_mse(self, mse: torch.Tensor):
        if self._total_mse is None:
            self._total_mse = torch.zeros_like(mse)
        self._total_mse += mse

    def _add_variance(self, variance: torch.Tensor):
        if self._total_variance is None:
            self._total_variance = torch.zeros_like(variance)
        self._total_variance += variance

    def get(self) -> torch.Tensor:
        if self._total_mse is None or self._total_variance is None:
            raise ValueError("No batches have been recorded.")
        spread = self._total_variance.sqrt()
        # must remove the component of the MSE that is due to the
        # variance of the generated values
        skill = (self._total_mse - self._total_variance).sqrt()
        return spread / skill - 1


class _EnsembleAggregator:
    """
    Aggregator for ensemble-based metrics.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        log_mean_maps: bool = True,
        metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        """
        Args:
            gridded_operations: Gridded operations to use.
            log_mean_maps: Whether to log mean maps.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
        """
        self._gridded_operations = gridded_operations
        self._n_batches = 0
        self._variable_metrics: dict[str, dict[str, ReducedMetric]] | None = None
        self._dist = Distributed.get_instance()
        self._log_mean_maps = log_mean_maps
        self._metadata = metadata
        self._diverging_metrics = {"ssr_bias"}

    def _get_variable_metrics(self, gen_data: TensorMapping):
        if self._variable_metrics is None:
            self._variable_metrics = {
                "crps": {},
                "ssr_bias": {},
            }
            for key in gen_data:
                self._variable_metrics["crps"][key] = CRPSMetric()
                self._variable_metrics["ssr_bias"][key] = SSRBiasMetric()
        return self._variable_metrics

    @torch.no_grad()
    def record_batch(
        self,
        target_data: EnsembleTensorDict,
        gen_data: EnsembleTensorDict,
    ):
        """
        Record a batch of data.

        Args:
            target_data: Target data, of shape [batch, ensemble, time, ...].
            gen_data: Generated data, of shape [batch, ensemble, time, ...].
        """
        variable_metrics = self._get_variable_metrics(gen_data)
        for metric in variable_metrics:
            for name in gen_data:
                variable_metrics[metric][name].record(
                    target=target_data[name],
                    gen=gen_data[name],
                )
        self._n_batches += 1

    def _get_caption(self, name: str) -> str:
        if self._metadata is not None and name in self._metadata:
            caption_name = self._metadata[name].long_name
            units = self._metadata[name].units
        else:
            caption_name, units = name, "unknown_units"
        caption = f"{caption_name} ({units})"
        return caption

    def _get_data(self):
        if self._variable_metrics is None or self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        data: dict[str, torch.Tensor] = {}
        for metric in sorted(self._variable_metrics):
            for key in sorted(self._variable_metrics[metric]):
                metric_value = self._dist.reduce_mean(
                    self._variable_metrics[metric][key].get()
                )
                data[f"{metric}/{key}"] = (
                    self._gridded_operations.area_weighted_mean(metric_value, name=key)
                    .cpu()
                    .numpy()
                )
                if self._log_mean_maps:
                    data[f"{metric}/mean_map/{key}"] = plot_paneled_data(
                        [[metric_value.cpu().numpy()]],
                        diverging=key in self._diverging_metrics,
                        caption=self._get_caption(key),
                    )
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


class SelectStepEnsembleAggregator:
    """
    Wraps an aggregator that takes a time dimension, and records
    metrics for a specific time step.
    """

    def __init__(
        self,
        aggregator: _EnsembleAggregator,
        i_target_time: int,
    ):
        """
        Args:
            aggregator: Aggregator to wrap.
            i_target_time: Global time index of the target time step.
        """
        self._aggregator = aggregator
        self._i_target_time = i_target_time

    def record_batch(
        self,
        target_data: EnsembleTensorDict,
        gen_data: EnsembleTensorDict,
        i_time_start: int = 0,
    ):
        """
        Record a specific global timestep of data.

        Call does nothing if the target time is not in this batch.

        Args:
            target_data: Target data, of shape [batch, ensemble, time, ...].
            gen_data: Generated data, of shape [batch, ensemble, time, ...].
            i_time_start: Global time index of the first time step in the batch.
        """
        n_timesteps = next(iter(target_data.values())).shape[2]
        if (
            self._i_target_time >= i_time_start
            and self._i_target_time < i_time_start + n_timesteps
        ):
            batch_i_target_time = self._i_target_time - i_time_start
            target_data = EnsembleTensorDict(
                {
                    key: value[:, :, batch_i_target_time : batch_i_target_time + 1, ...]
                    for key, value in target_data.items()
                }
            )
            gen_data = EnsembleTensorDict(
                {
                    key: value[:, :, batch_i_target_time : batch_i_target_time + 1, ...]
                    for key, value in gen_data.items()
                }
            )
            self._aggregator.record_batch(
                target_data,
                gen_data,
            )

    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return self._aggregator.get_logs(label)

    def get_dataset(self) -> xr.Dataset:
        return self._aggregator.get_dataset()
