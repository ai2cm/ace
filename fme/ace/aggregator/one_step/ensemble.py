import abc
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import torch
import xarray as xr

from fme.ace.aggregator.plotting import plot_paneled_data
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.ensemble import get_crps
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import EnsembleTensorDict, TensorMapping

from ..inference.build_context import MetricBuildContext, MetricNotSupportedError
from ..inference.data import MetricBuildResult
from .build_context import OneStepBuildContext, OneStepMetricBuildResult


def get_gen_shape(gen_data: TensorMapping):
    for name in gen_data:
        return gen_data[name].shape


def get_one_step_ensemble_aggregator(
    gridded_operations: GriddedOperations,
    target_time: int = 1,
    log_mean_maps: bool = True,
    metadata: Mapping[str, VariableMetadata] | None = None,
    target: Literal["norm", "denorm"] = "denorm",
    channel_mean_names: Sequence[str] | None = None,
) -> "SelectStepEnsembleAggregator":
    return SelectStepEnsembleAggregator(
        aggregator=_EnsembleAggregator(
            gridded_operations=gridded_operations,
            log_mean_maps=log_mean_maps,
            metadata=metadata,
            target=target,
            channel_mean_names=channel_mean_names,
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


class EnsembleMeanRMSEMetric(ReducedMetric):
    """
    Computes the ensemble mean RMSE.
    """

    def __init__(self):
        self._total_rmse = None
        self._n_batches = 0

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        ensemble_mean = gen.mean(dim=1, keepdim=True)  # mean over ensemble dimension
        rmse = ((ensemble_mean - target) ** 2).mean(dim=(0, 1, 2)).sqrt()
        if self._total_rmse is None:
            self._total_rmse = rmse
        else:
            self._total_rmse += rmse
        self._n_batches += 1

    def get(self) -> torch.Tensor:
        if self._total_rmse is None:
            raise ValueError("No batches have been recorded.")
        return self._total_rmse / self._n_batches


class SSRBiasMetric(ReducedMetric):
    """
    Computes the spread-skill ratio bias (equal to (stdev / rmse) - 1).
    """

    def __init__(self):
        self._total_unbiased_mse = None
        self._total_variance = None
        self._n_batches = 0

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        num_ensemble = gen.shape[1]
        ensemble_mean = gen.mean(dim=1, keepdim=True)  # batch, 1, time
        mse = ((ensemble_mean - target) ** 2).mean(dim=(0, 1, 2))  # batch, 1, time
        variance = gen.var(dim=1, unbiased=True).mean(dim=(0, 1))
        self._add_unbiased_mse(mse, variance, num_ensemble)
        self._add_variance(variance)
        self._n_batches += 1

    def _add_unbiased_mse(
        self, mse: torch.Tensor, variance: torch.Tensor, num_ensemble: int
    ):
        if self._total_unbiased_mse is None:
            self._total_unbiased_mse = torch.zeros_like(mse)
        # must remove the component of the MSE that is due to the
        # variance of the generated values
        self._total_unbiased_mse += mse - variance / num_ensemble

    def _add_variance(self, variance: torch.Tensor):
        if self._total_variance is None:
            self._total_variance = torch.zeros_like(variance)
        self._total_variance += variance

    def get(self) -> torch.Tensor:
        if self._total_unbiased_mse is None or self._total_variance is None:
            raise ValueError("No batches have been recorded.")
        spread = self._total_variance.sqrt()
        # Clamp to avoid NaN from sqrt of negative values. The unbiased MSE
        # correction (mse - variance/n_ensemble) can overshoot with small
        # ensembles or few batches, producing negative values at some grid
        # cells that do not indicate spread truly exceeding skill.
        skill = torch.clamp(self._total_unbiased_mse, min=0.0).sqrt()
        # When skill is zero (clamped or genuinely perfect), SSR is undefined.
        # Use -1 as the convention (equivalent to zero spread).
        return torch.where(skill > 0, spread / skill - 1, torch.tensor(-1.0))


class _EnsembleAggregator:
    """
    Aggregator for ensemble-based metrics.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        log_mean_maps: bool = True,
        metadata: Mapping[str, VariableMetadata] | None = None,
        target: Literal["norm", "denorm"] = "denorm",
        channel_mean_names: Sequence[str] | None = None,
    ):
        """
        Args:
            gridded_operations: Gridded operations to use.
            log_mean_maps: Whether to log mean maps.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            target: Whether to compute metrics on normalized ("norm") or
                denormalized ("denorm") data. Channel-mean metrics are only
                logged when target is "norm", since averaging metrics across
                variables with different physical units is not meaningful.
            channel_mean_names: Names of variables to include in channel-mean
                metrics. If None and target is "norm", the channel mean is
                computed over all variables present in the data. Names that
                are not present in the data raise KeyError. Ignored when
                target is "denorm".
        """
        self._gridded_operations = gridded_operations
        self._n_batches = 0
        self._variable_metrics: dict[str, dict[str, ReducedMetric]] | None = None
        self._dist = Distributed.get_instance()
        self._log_mean_maps = log_mean_maps
        self._metadata = metadata
        self._diverging_metrics = {"ssr_bias"}
        self._target = target
        self._channel_mean_names = channel_mean_names

    def _get_variable_metrics(self, gen_data: TensorMapping):
        if self._variable_metrics is None:
            self._variable_metrics = {
                "crps": {},
                "ssr_bias": {},
                "ensemble_mean_rmse": {},
            }
            for key in gen_data:
                self._variable_metrics["crps"][key] = CRPSMetric()
                self._variable_metrics["ssr_bias"][key] = SSRBiasMetric()
                self._variable_metrics["ensemble_mean_rmse"][key] = (
                    EnsembleMeanRMSEMetric()
                )
        return self._variable_metrics

    @torch.no_grad()
    def record_batch(
        self,
        target_data: EnsembleTensorDict,
        gen_data: EnsembleTensorDict,
        target_data_norm: EnsembleTensorDict | None = None,
        gen_data_norm: EnsembleTensorDict | None = None,
    ):
        """
        Record a batch of data.

        Args:
            target_data: Target data, of shape [batch, ensemble, time, ...].
            gen_data: Generated data, of shape [batch, ensemble, time, ...].
            target_data_norm: Normalized target data. Required when target is
                "norm".
            gen_data_norm: Normalized generated data. Required when target is
                "norm".
        """
        if self._target == "norm":
            if target_data_norm is None or gen_data_norm is None:
                raise ValueError(
                    "target_data_norm and gen_data_norm must be provided "
                    "when target is 'norm'."
                )
            target_data = target_data_norm
            gen_data = gen_data_norm
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
                        diverging=metric in self._diverging_metrics,
                        caption=self._get_caption(key),
                    )
            if self._target == "norm":
                all_keys = list(self._variable_metrics[metric].keys())
                if self._channel_mean_names is None:
                    names = all_keys
                else:
                    missing = [n for n in self._channel_mean_names if n not in all_keys]
                    if missing:
                        raise KeyError(
                            f"channel_mean_names contains entries not present "
                            f"in the recorded data: {missing}. Available: "
                            f"{sorted(all_keys)}."
                        )
                    names = list(self._channel_mean_names)
                if names:
                    scalars = [data[f"{metric}/{key}"] for key in names]
                    data[f"{metric}/channel_mean"] = sum(scalars) / len(scalars)
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
        target_data_norm: EnsembleTensorDict | None = None,
        gen_data_norm: EnsembleTensorDict | None = None,
        i_time_start: int = 0,
    ):
        """
        Record a specific global timestep of data.

        Call does nothing if the target time is not in this batch.

        Args:
            target_data: Target data, of shape [batch, ensemble, time, ...].
            gen_data: Generated data, of shape [batch, ensemble, time, ...].
            target_data_norm: Normalized target data, same shape as target_data.
            gen_data_norm: Normalized generated data, same shape as gen_data.
            i_time_start: Global time index of the first time step in the batch.
        """
        n_timesteps = next(iter(target_data.values())).shape[2]
        if (
            self._i_target_time >= i_time_start
            and self._i_target_time < i_time_start + n_timesteps
        ):
            batch_i_target_time = self._i_target_time - i_time_start

            def _select(data: EnsembleTensorDict) -> EnsembleTensorDict:
                return EnsembleTensorDict(
                    {
                        key: value[
                            :, :, batch_i_target_time : batch_i_target_time + 1, ...
                        ]
                        for key, value in data.items()
                    }
                )

            self._aggregator.record_batch(
                target_data=_select(target_data),
                gen_data=_select(gen_data),
                target_data_norm=(
                    _select(target_data_norm) if target_data_norm is not None else None
                ),
                gen_data_norm=(
                    _select(gen_data_norm) if gen_data_norm is not None else None
                ),
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


@dataclasses.dataclass
class EnsembleMetricConfig:
    """
    Configuration for an ensemble metric (CRPS, SSR bias, ensemble-mean RMSE)
    at a specific forward step.

    Attributes:
        step: Forward step at which to compute the metric.
        name: Name to use for the logged metric. Defaults to
            ``ensemble_step_{step}`` for ``target="denorm"`` and
            ``ensemble_step_{step}_norm`` for ``target="norm"``.
        log_mean_maps: Whether to log per-variable mean maps.
        enabled: Whether the metric is enabled.
        strict: Whether to raise if the metric cannot be built.
        target: Whether to compute metrics on normalized ("norm") or
            denormalized ("denorm") data. ``channel_mean`` is only logged
            when ``target="norm"``, since averaging metrics across variables
            with different physical units is not meaningful.
        channel_mean_names: Names of variables to include in the channel-mean
            metric. If None, falls back to the aggregator-level value passed
            via the build context, and finally to all variables present in
            the data when that is also None. Names not present in the data
            raise KeyError. Ignored when ``target="denorm"``.
    """

    step: int = 20
    name: str | None = None
    log_mean_maps: bool = False
    enabled: bool = True
    strict: bool = False
    target: Literal["norm", "denorm"] = "denorm"
    channel_mean_names: list[str] | None = None

    def __post_init__(self):
        if self.name is None:
            base = f"ensemble_step_{self.step}"
            self.name = f"{base}_norm" if self.target == "norm" else base

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        if self.step > ctx.n_forward_steps:
            raise MetricNotSupportedError(
                f"ensemble step {self.step} exceeds "
                f"n_forward_steps={ctx.n_forward_steps}"
            )
        is_norm = self.target == "norm"
        return MetricBuildResult(
            ensemble=get_one_step_ensemble_aggregator(
                gridded_operations=ctx.ops,
                target_time=self.step,
                log_mean_maps=self.log_mean_maps,
                metadata=ctx.variable_metadata,
                target=self.target,
                channel_mean_names=(
                    (self.channel_mean_names or ctx.channel_mean_names)
                    if is_norm
                    else None
                ),
            )
        )


@dataclasses.dataclass
class OneStepEnsembleMetricConfig:
    """
    Configuration for the one-step ensemble metric (CRPS, SSR bias,
    ensemble-mean RMSE) at the first forward step.

    Attributes:
        name: Name to use for the logged metric. Defaults to ``ensemble``
            for ``target="denorm"`` and ``ensemble_norm`` for
            ``target="norm"``.
        log_mean_maps: Whether to log per-variable mean maps.
        enabled: Whether the metric is enabled.
        strict: Whether to raise if the metric cannot be built.
        target: Whether to compute metrics on normalized ("norm") or
            denormalized ("denorm") data. ``channel_mean`` is only logged
            when ``target="norm"``, since averaging metrics across variables
            with different physical units is not meaningful.
        channel_mean_names: Names of variables to include in the channel-mean
            metric. If None, falls back to the aggregator-level value passed
            via the build context, and finally to all variables present in
            the data when that is also None. Names not present in the data
            raise KeyError. Ignored when ``target="denorm"``.
    """

    name: str | None = None
    log_mean_maps: bool = True
    enabled: bool = True
    strict: bool = False
    target: Literal["norm", "denorm"] = "denorm"
    channel_mean_names: list[str] | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = "ensemble_norm" if self.target == "norm" else "ensemble"

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: OneStepBuildContext) -> OneStepMetricBuildResult:
        is_norm = self.target == "norm"
        return OneStepMetricBuildResult(
            ensemble=get_one_step_ensemble_aggregator(
                gridded_operations=ctx.ops,
                log_mean_maps=self.log_mean_maps,
                target_time=1,
                metadata=ctx.variable_metadata,
                target=self.target,
                channel_mean_names=(
                    (self.channel_mean_names or ctx.channel_mean_names)
                    if is_norm
                    else None
                ),
            )
        )
