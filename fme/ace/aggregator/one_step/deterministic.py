import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np
import torch
import xarray as xr

from fme.core.dataset_info import DatasetInfo
from fme.core.diagnostics import get_reduced_diagnostics, write_reduced_diagnostics
from fme.core.generics.aggregator import AggregatorABC
from fme.core.typing_ import TensorDict, TensorMapping

from .map import MapAggregator
from .reduced import MeanAggregator
from .snapshot import SnapshotAggregator
from .spectrum import SpectrumAggregator


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
        dataset_info: DatasetInfo,
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        loss_scaling: TensorMapping | None = None,
        log_snapshots: bool = True,
        log_mean_maps: bool = True,
    ):
        """
        Args:
            dataset_info: Dataset coordinates and metadata.
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
        horizontal_coordinates = dataset_info.horizontal_coordinates
        self._coords = horizontal_coordinates.coords
        self._aggregators: dict[str, _Aggregator] = {
            "mean": MeanAggregator(dataset_info.gridded_operations),
        }
        try:
            self._aggregators["power_spectrum"] = SpectrumAggregator(
                dataset_info.gridded_operations,
            )
        except NotImplementedError:
            logging.warning(
                "Spectrum aggregator not implemented for this grid type, omitting."
            )
        if log_snapshots:
            self._aggregators["snapshot"] = SnapshotAggregator(
                horizontal_coordinates.dims, dataset_info.variable_metadata
            )
        if log_mean_maps:
            self._aggregators["mean_map"] = MapAggregator(
                horizontal_coordinates.dims, dataset_info.variable_metadata
            )

        self._loss_scaling = loss_scaling or {}

    @torch.no_grad()
    def record_batch(
        self,
        batch: DeterministicTrainOutput,
    ):
        if len(batch.target_data) == 0:
            raise ValueError("No data in target_data")
        if len(batch.gen_data) == 0:
            raise ValueError("No data in gen_data")

        target_data_norm = batch.normalize(batch.target_data)
        gen_data_norm = batch.normalize(batch.gen_data)
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
            logging.info(f"Getting logs for {agg_label} aggregator")
            for k, v in self._aggregators[agg_label].get_logs(label=agg_label).items():
                logs[f"{label}/{k}"] = v
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
