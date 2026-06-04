import dataclasses
import logging
from collections.abc import Callable, Mapping

import numpy as np
import torch

from fme.core.device import get_device
from fme.core.diagnostics import get_reduced_diagnostics, write_reduced_diagnostics
from fme.core.distributed import Distributed
from fme.core.generics.aggregator import AggregatorABC
from fme.core.typing_ import TensorDict, TensorMapping

from .build_context import Aggregator


@dataclasses.dataclass
class DeterministicTrainOutput:
    metrics: TensorDict
    gen_data: TensorDict
    target_data: TensorDict
    normalize: Callable[[TensorDict], TensorDict]


class OneStepDeterministicAggregator(AggregatorABC[DeterministicTrainOutput]):
    """
    Aggregates statistics for the first timestep.

    Operates on batches without an ensemble dimension (i.e. shape [batch, ...]).

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        aggregators: Mapping[str, Aggregator],
        coords: Mapping[str, np.ndarray],
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        loss_scaling: TensorMapping | None = None,
    ):
        """
        Args:
            aggregators: Named sub-aggregators keyed by metric name. Must
                include a ``"mean_norm"`` entry.
            coords: Coordinate arrays for writing diagnostics.
            save_diagnostics: Whether to save diagnostics.
            output_dir: Directory to write diagnostics to.
            loss_scaling: Dictionary of variables and their scaling factors
                used in loss computation.
        """
        if save_diagnostics and output_dir is None:
            raise ValueError("Output directory must be set to save diagnostics.")
        if "mean_norm" not in aggregators:
            raise ValueError(
                "An aggregator named 'mean_norm' is required. "
                "Include a OneStepMeanMetricConfig with target='norm' "
                "in your metrics list."
            )
        self._output_dir = output_dir
        self._save_diagnostics = save_diagnostics
        self._coords = coords
        self._aggregators = aggregators
        self._loss_scaling = loss_scaling or {}
        self._loss = torch.tensor(0.0, device=get_device())
        self._n_loss_batches = 0

    @torch.no_grad()
    def record_batch(
        self,
        batch: DeterministicTrainOutput,
    ):
        if len(batch.target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        loss = batch.metrics.get("loss", np.nan)
        self._loss = self._loss + loss
        self._n_loss_batches += 1
        target_data_norm = batch.normalize(batch.target_data)
        gen_data_norm = batch.normalize(batch.gen_data)
        for agg in self._aggregators.values():
            agg.record_batch(
                loss=loss,
                target_data=batch.target_data,
                gen_data=batch.gen_data,
                target_data_norm=target_data_norm,
                gen_data_norm=gen_data_norm,
            )

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {}
        for agg_label in self._aggregators:
            logging.info(f"Getting logs for {agg_label} aggregator")
            for k, v in self._aggregators[agg_label].get_logs(label=agg_label).items():
                logs[f"{label}/{k}"] = v
        logs.pop(f"{label}/mean_norm/loss", None)
        logging.info(f"Inserting loss-scaled MSE componenets into logs")
        logs.update(
            self._get_loss_scaled_mse_components(
                validation_metrics=logs,
                label=label,
            )
        )
        return logs

    def _get_loss_scaled_mse_components(
        self,
        validation_metrics: Mapping[str, float],
        label: str,
    ):
        scaled_squared_errors = {}

        for var in self._loss_scaling:
            rmse_key = f"{label}/mean/weighted_rmse/{var}"
            if rmse_key in validation_metrics:
                scaled_squared_errors[var] = (
                    validation_metrics[rmse_key] / self._loss_scaling[var].item()
                ) ** 2
        scaled_squared_errors_sum = sum(scaled_squared_errors.values())
        fractional_contribs = {
            f"{label}/mean/mse_fractional_components/{k}": v / scaled_squared_errors_sum
            for k, v in scaled_squared_errors.items()
        }
        return fractional_contribs

    @torch.no_grad()
    def get_loss(self) -> float | None:
        if self._n_loss_batches == 0:
            return None
        dist = Distributed.get_instance()
        return float(dist.reduce_mean(self._loss / self._n_loss_batches).cpu().numpy())

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None = None):
        if self._save_diagnostics:
            reduced_diagnostics = get_reduced_diagnostics(
                sub_aggregators=self._aggregators,
                coords=self._coords,
            )
            if self._output_dir is not None:
                write_reduced_diagnostics(
                    reduced_diagnostics,
                    self._output_dir,
                    subdir=subdir,
                )
            else:
                raise ValueError("Output directory is not set.")
