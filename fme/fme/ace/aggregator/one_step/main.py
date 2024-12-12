from typing import Dict, Mapping, Optional, Protocol

import numpy as np
import torch

from fme.ace.stepper import TrainOutput
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.generics.aggregator import AggregatorABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping

from .map import MapAggregator
from .reduced import MeanAggregator
from .snapshot import SnapshotAggregator


class _Aggregator(Protocol):
    def get_logs(self, label: str) -> TensorMapping: ...

    def record_batch(
        self,
        loss: float,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
    ) -> None: ...


class OneStepAggregator(AggregatorABC[TrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        loss_scaling: Optional[TensorMapping] = None,
    ):
        """
        Args:
            gridded_operations: Operations for computing metrics on gridded data.
            variable_metadata: Metadata for each variable.
            loss_scaling: Dictionary of variables and their scaling factors
                used in loss computation.
        """
        aggregators: Dict[str, _Aggregator] = {
            "mean": MeanAggregator(gridded_operations)
        }
        aggregators["snapshot"] = SnapshotAggregator(variable_metadata)
        aggregators["mean_map"] = MapAggregator(variable_metadata)
        self._aggregators = aggregators
        self._loss_scaling = loss_scaling or {}

    @torch.no_grad()
    def record_batch(
        self,
        batch: TrainOutput,
    ):
        if len(batch.target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        gen_data_norm = batch.normalize(batch.gen_data)
        target_data_norm = batch.normalize(batch.target_data)
        for agg in self._aggregators.values():
            agg.record_batch(
                loss=batch.metrics.get("loss", np.nan),
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
            for k, v in self._aggregators[agg_label].get_logs(label=agg_label).items():
                logs[f"{label}/{k}"] = v
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
