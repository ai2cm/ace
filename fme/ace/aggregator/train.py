import dataclasses
from typing import Protocol

import torch

from fme.ace.aggregator.one_step.reduced import MeanAggregator
from fme.ace.aggregator.one_step.spectrum import PairedSphericalPowerSpectrumAggregator
from fme.ace.stepper import TrainOutput
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.aggregator import AggregatorABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.tensors import fold_ensemble_dim, fold_sized_ensemble_dim
from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class TrainAggregatorConfig:
    """
    Configuration for the train aggregator.

    Attributes:
        spherical_power_spectrum: Whether to compute the spherical power spectrum.
        weighted_rmse: Whether to compute the weighted RMSE.
        metric_sample_probability: The probability of sampling a batch for computing
            metrics. The first batch is always sampled for metrics, so that metrics
            are always logged. All batches are used for computing the loss.
    """

    spherical_power_spectrum: bool = True
    weighted_rmse: bool = True
    metric_sample_probability: float = 0.01


class Aggregator(Protocol):
    def record_batch(self, target_data: TensorMapping, gen_data: TensorMapping):
        pass

    def get_logs(self, label: str) -> dict[str, torch.Tensor]:
        pass


class TrainAggregator(AggregatorABC[TrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(self, config: TrainAggregatorConfig, operations: GriddedOperations):
        self._n_loss_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())
        self._paired_aggregators: dict[str, Aggregator] = {}
        if config.spherical_power_spectrum:
            self._paired_aggregators["power_spectrum"] = (
                PairedSphericalPowerSpectrumAggregator(
                    gridded_operations=operations,
                    report_plot=False,
                )
            )
        if config.weighted_rmse:
            self._paired_aggregators["mean"] = MeanAggregator(
                gridded_operations=operations,
                include_bias=False,
                include_grad_mag_percent_diff=False,
            )
        self._metric_sample_probability = config.metric_sample_probability

    @torch.no_grad()
    def record_batch(self, batch: TrainOutput):
        self._loss += batch.metrics["loss"]
        self._n_loss_batches += 1

        # always sample metrics for the first batch, so that metrics are always logged
        if self._n_loss_batches == 1 or (
            torch.rand(1) < self._metric_sample_probability
        ):
            folded_gen_data, n_ensemble = fold_ensemble_dim(batch.gen_data)
            folded_target_data = fold_sized_ensemble_dim(batch.target_data, n_ensemble)
            for aggregator in self._paired_aggregators.values():
                aggregator.record_batch(
                    target_data=folded_target_data,
                    gen_data=folded_gen_data,
                )

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, torch.Tensor]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {}
        if self._n_loss_batches > 0:
            logs[f"{label}/mean/loss"] = self._loss / self._n_loss_batches
            dist = Distributed.get_instance()
            for key in sorted(logs.keys()):
                logs[key] = float(dist.reduce_mean(logs[key].detach()).cpu().numpy())
            for name, aggregator in self._paired_aggregators.items():
                logs.update(
                    {f"{label}/{k}": v for k, v in aggregator.get_logs(name).items()}
                )
        return logs

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None) -> None:
        pass
