import dataclasses

import torch

from fme.ace.aggregator.one_step.deterministic import (
    DeterministicTrainOutput,
    OneStepDeterministicAggregator,
)
from fme.ace.aggregator.one_step.ensemble import get_one_step_ensemble_aggregator
from fme.ace.stepper import TrainOutput
from fme.core.dataset_info import DatasetInfo
from fme.core.generics.aggregator import AggregatorABC
from fme.core.tensors import fold_ensemble_dim, fold_sized_ensemble_dim
from fme.core.typing_ import TensorMapping


class OneStepAggregator(AggregatorABC[TrainOutput]):
    """
    Aggregates statistics for the first timestep.

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
            loss_scaling: Dictionary of variables and their scaling factors
                used in loss computation.
            log_snapshots: Whether to include snapshots in diagnostics.
            log_mean_maps: Whether to include mean maps in diagnostics.
        """
        self._deterministic_aggregator = OneStepDeterministicAggregator(
            dataset_info=dataset_info,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
            loss_scaling=loss_scaling,
            log_snapshots=log_snapshots,
            log_mean_maps=log_mean_maps,
        )
        self._ensemble_aggregator = get_one_step_ensemble_aggregator(
            gridded_operations=dataset_info.gridded_operations,
            log_mean_maps=log_mean_maps,
            target_time=1,
            metadata=dataset_info.variable_metadata,
        )
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
    ):
        return OneStepAggregator(
            dataset_info=dataset_info,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
            loss_scaling=loss_scaling,
            log_snapshots=self.log_snapshots,
            log_mean_maps=self.log_mean_maps,
        )
