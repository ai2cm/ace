import dataclasses
import logging
from collections.abc import Sequence

import torch

from fme.ace.aggregator.one_step.deterministic import (
    DeterministicTrainOutput,
    OneStepDeterministicAggregator,
    _Aggregator,
)
from fme.ace.aggregator.one_step.ensemble import (
    SelectStepEnsembleAggregator,
    get_one_step_ensemble_aggregator,
)
from fme.ace.stepper import TrainOutput
from fme.core.dataset_info import DatasetInfo
from fme.core.fill import SmoothFloodFill
from fme.core.generics.aggregator import AggregatorABC
from fme.core.tensors import fold_ensemble_dim, fold_sized_ensemble_dim
from fme.core.typing_ import TensorMapping

from .map import MapAggregator
from .reduced import MeanAggregator
from .snapshot import SnapshotAggregator
from .spectrum import SpectrumAggregator


class OneStepAggregator(AggregatorABC[TrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        deterministic_aggregator: OneStepDeterministicAggregator,
        ensemble_aggregator: SelectStepEnsembleAggregator,
    ):
        self._deterministic_aggregator = deterministic_aggregator
        self._ensemble_aggregator = ensemble_aggregator
        self._ensemble_recorded = False

    @torch.no_grad()
    def record_batch(
        self,
        batch: TrainOutput,
    ):
        folded_gen_data, n_ensemble = fold_ensemble_dim(batch.gen_data)
        folded_target_data = fold_sized_ensemble_dim(batch.target_data, n_ensemble)
        self._deterministic_aggregator.record_batch(
            DeterministicTrainOutput(
                metrics=batch.metrics,
                gen_data=folded_gen_data,
                target_data=folded_target_data,
                normalize=batch.normalize,
            )
        )
        if n_ensemble > 1:
            self._ensemble_aggregator.record_batch(
                target_data=batch.target_data,
                gen_data=batch.gen_data,
                i_time_start=0,
            )
            self._ensemble_recorded = True

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        deterministic_logs = self._deterministic_aggregator.get_logs(label)
        if self._ensemble_recorded:
            stochastic_logs = self._ensemble_aggregator.get_logs(label)
            if len(set(deterministic_logs.keys()) & set(stochastic_logs.keys())) > 0:
                raise ValueError(
                    "Stochastic and deterministic logs have overlapping keys, "
                    f"stochastic logs: {stochastic_logs}, "
                    f"deterministic logs: {deterministic_logs}"
                )
            return {**deterministic_logs, **stochastic_logs}
        else:
            return deterministic_logs

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None = None):
        self._deterministic_aggregator.flush_diagnostics(subdir)


def _build_default_deterministic_aggregators(
    dataset_info: DatasetInfo,
    log_snapshots: bool,
    log_mean_maps: bool,
    channel_mean_names: Sequence[str] | None,
) -> dict[str, _Aggregator]:
    aggregators: dict[str, _Aggregator] = {
        "mean": MeanAggregator(dataset_info.gridded_operations),
        "mean_norm": MeanAggregator(
            dataset_info.gridded_operations,
            target="norm",
            channel_mean_names=channel_mean_names,
            include_bias=False,
            include_grad_mag_percent_diff=False,
        ),
    }
    try:
        flood_fill = SmoothFloodFill(num_steps=4)
        aggregators["power_spectrum"] = SpectrumAggregator(
            dataset_info.gridded_operations,
            nan_fill_fn=flood_fill,
        )
    except NotImplementedError:
        logging.warning(
            "Spectrum aggregator not implemented for this grid type, omitting."
        )
    horizontal_coordinates = dataset_info.horizontal_coordinates
    if log_snapshots:
        aggregators["snapshot"] = SnapshotAggregator(
            horizontal_coordinates.dims, dataset_info.variable_metadata
        )
    if log_mean_maps:
        aggregators["mean_map"] = MapAggregator(
            horizontal_coordinates.dims, dataset_info.variable_metadata
        )
    return aggregators


@dataclasses.dataclass
class OneStepAggregatorConfig:
    """
    Configuration for the validation OneStepAggregator.

    Arguments:
        log_snapshots: Whether to log snapshot images.
        log_mean_maps: Whether to log mean map images.
    """

    log_snapshots: bool = True
    log_mean_maps: bool = True

    def build(
        self,
        dataset_info: DatasetInfo,
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        loss_scaling: TensorMapping | None = None,
        channel_mean_names: Sequence[str] | None = None,
    ):
        aggregators = _build_default_deterministic_aggregators(
            dataset_info=dataset_info,
            log_snapshots=self.log_snapshots,
            log_mean_maps=self.log_mean_maps,
            channel_mean_names=channel_mean_names,
        )
        deterministic_aggregator = OneStepDeterministicAggregator(
            aggregators=aggregators,
            coords=dataset_info.horizontal_coordinates.coords,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
            loss_scaling=loss_scaling,
        )
        ensemble_aggregator = get_one_step_ensemble_aggregator(
            gridded_operations=dataset_info.gridded_operations,
            log_mean_maps=self.log_mean_maps,
            target_time=1,
            metadata=dataset_info.variable_metadata,
        )
        return OneStepAggregator(
            deterministic_aggregator=deterministic_aggregator,
            ensemble_aggregator=ensemble_aggregator,
        )
