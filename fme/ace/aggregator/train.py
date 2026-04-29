import dataclasses
import logging
from typing import Protocol

import torch

from fme.ace.aggregator.inference.data import InferenceBatchData
from fme.ace.aggregator.inference.spectrum import PairedSphericalPowerSpectrumAggregator
from fme.ace.aggregator.one_step.reduced import MeanAggregator
from fme.ace.stepper import TrainOutput
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.fill import SmoothFloodFill
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
        per_channel_loss: Whether to accumulate and report per-variable (per-channel)
            loss in get_logs (e.g. train/mean/loss/<var_name>).
    """

    spherical_power_spectrum: bool = True
    weighted_rmse: bool = True
    per_channel_loss: bool = True


class Aggregator(Protocol):
    def record_batch(self, target_data: TensorMapping, gen_data: TensorMapping):
        pass

    def get_logs(self, label: str) -> dict[str, torch.Tensor]:
        pass


class _TrainSpectrumAdapter:
    """Adapts PairedSphericalPowerSpectrumAggregator for the training context."""

    def __init__(self, inner: PairedSphericalPowerSpectrumAggregator):
        self._inner = inner

    def record_batch(self, target_data: TensorMapping, gen_data: TensorMapping):
        self._inner.record_batch(
            InferenceBatchData(
                prediction=gen_data,
                prediction_norm=gen_data,
                target=target_data,
                target_norm=target_data,
                time=None,
                i_time_start=0,
            )
        )

    def get_logs(self, label: str) -> dict[str, torch.Tensor]:
        return self._inner.get_logs(label)


class TrainAggregator(AggregatorABC[TrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(self, config: TrainAggregatorConfig, operations: GriddedOperations):
        self._n_loss_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())
        self._per_channel_loss: dict[str, torch.Tensor] = {}
        self._per_channel_loss_enabled = config.per_channel_loss
        self._paired_aggregators: dict[str, Aggregator] = {}
        if config.spherical_power_spectrum:
            try:
                flood_fill = SmoothFloodFill(num_steps=4)
                self._paired_aggregators["power_spectrum"] = _TrainSpectrumAdapter(
                    PairedSphericalPowerSpectrumAggregator(
                        gridded_operations=operations,
                        report_plot=False,
                        nan_fill_fn=flood_fill,
                    )
                )
            except NotImplementedError:
                logging.warning(
                    "Power spectrum aggregator not implemented "
                    "for this grid type, omitting."
                )
        if config.weighted_rmse:
            self._paired_aggregators["mean"] = MeanAggregator(
                gridded_operations=operations,
                include_bias=False,
                include_grad_mag_percent_diff=False,
            )

    @torch.no_grad()
    def record_batch(self, batch: TrainOutput):
        self._loss += batch.metrics["loss"]
        self._n_loss_batches += 1
        if self._per_channel_loss_enabled and batch.per_channel_losses is not None:
            for var_name, value in batch.per_channel_losses.items():
                acc = self._per_channel_loss.get(
                    var_name,
                    torch.tensor(0.0, device=get_device(), dtype=value.dtype),
                )
                self._per_channel_loss[var_name] = acc + value

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
            for name, aggregator in self._paired_aggregators.items():
                logs.update(
                    {f"{label}/{k}": v for k, v in aggregator.get_logs(name).items()}
                )
        dist = Distributed.get_instance()
        logs[f"{label}/mean/loss"] = float(
            dist.reduce_mean(self._loss / self._n_loss_batches).cpu().numpy()
        )
        if self._n_loss_batches > 0 and self._per_channel_loss_enabled:
            for var_name, acc in self._per_channel_loss.items():
                logs[f"{label}/mean/loss/{var_name}"] = float(
                    dist.reduce_mean(acc / self._n_loss_batches).cpu().numpy()
                )
        return logs

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None) -> None:
        pass
