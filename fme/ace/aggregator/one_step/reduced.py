import dataclasses
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping

from ..inference.build_context import MetricBuildContext, maybe_filter
from ..inference.data import InferenceBatchData, SubAggregator
from .reduced_metrics import AreaWeightedReducedMetric, ReducedMetric


def get_gen_shape(gen_data: TensorMapping):
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
        gridded_operations: GriddedOperations,
        target_time: int = 1,
        include_bias: bool = True,
        include_grad_mag_percent_diff: bool = True,
        target: Literal["norm", "denorm"] = "denorm",
        channel_mean_names: Sequence[str] | None = None,
        log_loss: bool = True,
    ):
        """
        Args:
            gridded_operations: GriddedOperations object for computing metrics.
            target_time: Time index to compute metrics at, where 0 corresponds to the
                first timestep of the initial condition. For example, target_time=1 will
                compute metrics at the first timestep of the forward trajectory if there
                is 1 initial condition step.
            include_bias: Whether to include bias metrics.
            include_grad_mag_percent_diff: Whether to include gradient magnitude percent
                difference metrics.
            target: Whether to compute metrics on normalized ("norm") or denormalized
                ("denorm") data.
            channel_mean_names: Names to include in channel-mean metrics. If None,
                channel means will not be logged.
            log_loss: Whether to log the mean loss across batches.
        """
        self._gridded_operations = gridded_operations
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())
        self._target_time = target_time
        self._target = target
        self._log_loss = log_loss
        self._dist = Distributed.get_instance()

        device = get_device()
        self._variable_metrics: dict[str, ReducedMetric] = {}
        self._variable_metrics["weighted_rmse"] = AreaWeightedReducedMetric(
            device=device,
            compute_metric=self._gridded_operations.area_weighted_rmse_dict,
            channel_mean_names=channel_mean_names,
        )
        if include_bias:
            self._variable_metrics["weighted_bias"] = AreaWeightedReducedMetric(
                device=device,
                compute_metric=self._gridded_operations.area_weighted_mean_bias_dict,
            )
        if include_grad_mag_percent_diff:
            self._variable_metrics["weighted_grad_mag_percent_diff"] = (
                AreaWeightedReducedMetric(
                    device=device,
                    compute_metric=self._gridded_operations.area_weighted_gradient_magnitude_percent_diff_dict,  # noqa: E501
                )
            )

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping | None = None,
        gen_data_norm: TensorMapping | None = None,
        loss: torch.Tensor = torch.tensor(np.nan),
        i_time_start: int = 0,
    ):
        self._loss += loss
        time_dim = 1
        time_len = gen_data[list(gen_data.keys())[0]].shape[time_dim]
        target_time = self._target_time - i_time_start
        if self._target == "norm":
            if target_data_norm is None or gen_data_norm is None:
                raise ValueError(
                    "target_data_norm and gen_data_norm must be provided "
                    "if target is 'norm'."
                )
            target_data = target_data_norm
            gen_data = gen_data_norm
        if target_time >= 0 and time_len > target_time:
            target_snapshot = {}
            gen_snapshot = {}
            for name in gen_data.keys():
                target_snapshot[name] = target_data[name].select(
                    dim=time_dim, index=target_time
                )
                gen_snapshot[name] = gen_data[name].select(
                    dim=time_dim, index=target_time
                )
            for metric in self._variable_metrics.values():
                metric.record(
                    target=target_snapshot,
                    gen=gen_snapshot,
                )
            # only increment n_batches if we actually recorded a batch
            self._n_batches += 1

    def _get_data(self):
        if self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        data: dict[str, torch.Tensor] = {}
        if self._log_loss:
            data["loss"] = self._loss / self._n_batches
        for name, metric in self._variable_metrics.items():
            metric_results = metric.get()
            for key in metric_results:
                data[f"{name}/{key}"] = metric_results[key] / self._n_batches
            if self._target == "norm":
                data[f"{name}/channel_mean"] = (
                    metric.get_channel_mean() / self._n_batches
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


class OneStepMeanAdapter:
    """Adapts OneStepMeanAggregator to accept InferenceBatchData."""

    def __init__(self, inner: MeanAggregator):
        self._inner = inner

    def record_batch(self, data: InferenceBatchData) -> None:
        self._inner.record_batch(
            target_data=data.target,
            gen_data=data.prediction,
            target_data_norm=data.target_norm,
            gen_data_norm=data.prediction_norm,
            i_time_start=data.i_time_start,
        )

    def get_logs(self, label: str) -> dict[str, Any]:
        return self._inner.get_logs(label)

    def get_dataset(self) -> xr.Dataset:
        return self._inner.get_dataset()


@dataclasses.dataclass
class StepMeanMetricConfig:
    step: int
    type: Literal["step_mean"] = "step_mean"
    variables: list[str] | None = None
    name: str | None = None
    target: Literal["denorm", "norm"] = "denorm"
    channel_mean_names: list[str] | None = None

    def __post_init__(self):
        if self.name is None:
            base = f"mean_step_{self.step}"
            self.name = f"{base}_norm" if self.target == "norm" else base

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: MetricBuildContext) -> SubAggregator:
        if self.step > ctx.n_forward_steps:
            raise ValueError(
                f"step_mean step {self.step} exceeds "
                f"n_forward_steps={ctx.n_forward_steps}"
            )
        target_time = self.step + ctx.n_ic_steps - 1
        is_norm = self.target == "norm"
        agg: SubAggregator = OneStepMeanAdapter(
            MeanAggregator(
                ctx.ops,
                target_time=target_time,
                target=self.target,
                log_loss=False,
                include_bias=not is_norm,
                include_grad_mag_percent_diff=not is_norm,
                channel_mean_names=(
                    (self.channel_mean_names or ctx.channel_mean_names)
                    if is_norm
                    else None
                ),
            )
        )
        return maybe_filter(agg, self.variables)
