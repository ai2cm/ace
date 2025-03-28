import dataclasses
from typing import Dict, Mapping, Optional, Protocol

import numpy as np
import torch
import xarray as xr

from fme.ace.stepper import TrainOutput
from fme.core.coordinates import HorizontalCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.diagnostics import get_reduced_diagnostics, write_reduced_diagnostics
from fme.core.generics.aggregator import AggregatorABC
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

    def get_dataset(self) -> xr.Dataset: ...


@dataclasses.dataclass
class OneStepAggregatorConfig:
    """
    Configuration for the validation OneStepAggregator.

    Arguments:
        log_snapshots: Whether to log snapshot images during.
        log_mean_maps: Whether to log mean map images during.
    """

    log_snapshots: bool = True
    log_mean_maps: bool = True

    def build(
        self,
        horizontal_coordinates: HorizontalCoordinates,
        save_diagnostics: bool = True,
        output_dir: Optional[str] = None,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        loss_scaling: Optional[TensorMapping] = None,
    ):
        return OneStepAggregator(
            horizontal_coordinates=horizontal_coordinates,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
            variable_metadata=variable_metadata,
            loss_scaling=loss_scaling,
            log_snapshots=self.log_snapshots,
            log_mean_maps=self.log_mean_maps,
        )


class OneStepAggregator(AggregatorABC[TrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        horizontal_coordinates: HorizontalCoordinates,
        save_diagnostics: bool = True,
        output_dir: Optional[str] = None,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        loss_scaling: Optional[TensorMapping] = None,
        log_snapshots: bool = True,
        log_mean_maps: bool = True,
    ):
        """
        Args:
            horizontal_coordinates: Horizontal coordinates of the data.
            save_diagnostics: Whether to save diagnostics.
            output_dir: Directory to write diagnostics to.
            variable_metadata: Metadata for each variable.
            loss_scaling: Dictionary of variables and their scaling factors
                used in loss computation.
            log_snapshots: Whether to include snapshots in diagnostics.
            log_mean_maps: Whether to include mean maps in diagnostics.
        """
        if save_diagnostics and output_dir is None:
            raise ValueError("Output directory must be set to save diagnostics.")
        self._output_dir = output_dir
        self._save_diagnostics = save_diagnostics
        self._coords = horizontal_coordinates.coords
        aggregators: Dict[str, _Aggregator] = {
            "mean": MeanAggregator(horizontal_coordinates.gridded_operations)
        }
        if log_snapshots:
            aggregators["snapshot"] = SnapshotAggregator(
                horizontal_coordinates.dims, variable_metadata
            )
        if log_mean_maps:
            aggregators["mean_map"] = MapAggregator(
                horizontal_coordinates.dims, variable_metadata
            )
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

    @torch.no_grad()
    def flush_diagnostics(self, subdir: Optional[str] = None):
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
