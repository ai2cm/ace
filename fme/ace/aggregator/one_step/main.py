import dataclasses
import logging
import warnings
from collections.abc import Sequence

import torch

from fme.ace.aggregator.loss_metrics import PerStepLossAggregator
from fme.ace.aggregator.one_step.deterministic import (
    DeterministicTrainOutput,
    OneStepDeterministicAggregator,
)
from fme.ace.aggregator.one_step.ensemble import (
    OneStepEnsembleMetricConfig,
    get_one_step_ensemble_aggregator,
)
from fme.ace.stepper import TrainOutput
from fme.core.dataset_info import DatasetInfo
from fme.core.generics.aggregator import AggregatorABC
from fme.core.tensors import fold_ensemble_dim, fold_sized_ensemble_dim
from fme.core.typing_ import TensorMapping

from .build_context import (
    MetricNotSupportedError,
    OneStepBuildContext,
    OneStepMetricBuildResult,
    _Aggregator,
    _EnsembleAggregator,
)
from .map import OneStepMapMetricConfig
from .reduced import OneStepMeanMetricConfig
from .snapshot import OneStepSnapshotMetricConfig
from .spectrum import OneStepSpectrumMetricConfig

OneStepMetricConfig = (
    OneStepMeanMetricConfig
    | OneStepSnapshotMetricConfig
    | OneStepMapMetricConfig
    | OneStepSpectrumMetricConfig
    | OneStepEnsembleMetricConfig
)


class OneStepAggregator(AggregatorABC[TrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        deterministic_aggregator: OneStepDeterministicAggregator,
        ensemble_aggregator: _EnsembleAggregator,
    ):
        self._deterministic_aggregator = deterministic_aggregator
        self._ensemble_aggregator = ensemble_aggregator
        self._ensemble_recorded = False
        self._per_step_losses = PerStepLossAggregator()

    @torch.no_grad()
    def record_batch(
        self,
        batch: TrainOutput,
    ):
        step_metrics = {
            k: v
            for k, v in batch.metrics.items()
            if k.startswith("loss_step_") or k.startswith("loss/")
        }
        self._per_step_losses.record(step_metrics)
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
        deterministic_logs.update(self._per_step_losses.get_logs(label))
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


def _validate_no_duplicate_names(metrics: list[OneStepMetricConfig]) -> None:
    names = [m.get_name() for m in metrics]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for n in names:
        if n in seen:
            duplicates.add(n)
        seen.add(n)
    if duplicates:
        raise ValueError(
            f"Duplicate metric names: {sorted(duplicates)}. "
            "Use the 'name' field to disambiguate."
        )


def build_one_step_aggregator(
    metrics: list[OneStepMetricConfig],
    dataset_info: DatasetInfo,
    save_diagnostics: bool = True,
    output_dir: str | None = None,
    loss_scaling: TensorMapping | None = None,
    channel_mean_names: Sequence[str] | None = None,
    raise_on_unsupported: bool = True,
) -> OneStepAggregator:
    _validate_no_duplicate_names(metrics)
    ctx = OneStepBuildContext(
        ops=dataset_info.gridded_operations,
        horizontal_coordinates=dataset_info.horizontal_coordinates,
        variable_metadata=dataset_info.variable_metadata,
        channel_mean_names=channel_mean_names,
    )

    deterministic_aggregators: dict[str, _Aggregator] = {}
    ensemble_aggregator: _EnsembleAggregator | None = None

    for metric in metrics:
        name = metric.get_name()
        try:
            result: OneStepMetricBuildResult = metric.build(ctx)
        except MetricNotSupportedError as e:
            if raise_on_unsupported or metric.strict:
                raise
            logging.warning(
                f"{name} metric not supported for this configuration, omitting: {e}"
            )
            continue

        if result.deterministic is not None:
            deterministic_aggregators[name] = result.deterministic
        if result.ensemble is not None:
            if ensemble_aggregator is not None:
                raise ValueError("Multiple ensemble metrics are not supported.")
            ensemble_aggregator = result.ensemble

    if ensemble_aggregator is None:
        ensemble_aggregator = get_one_step_ensemble_aggregator(
            gridded_operations=ctx.ops,
            target_time=1,
            metadata=ctx.variable_metadata,
        )

    deterministic = OneStepDeterministicAggregator(
        aggregators=deterministic_aggregators,
        coords=dataset_info.horizontal_coordinates.coords,
        save_diagnostics=save_diagnostics,
        output_dir=output_dir,
        loss_scaling=loss_scaling,
    )
    return OneStepAggregator(
        deterministic_aggregator=deterministic,
        ensemble_aggregator=ensemble_aggregator,
    )


@dataclasses.dataclass
class OneStepAggregatorConfig:
    """
    Configuration for the validation OneStepAggregator.

    Each metric is a named field with its own typed configuration and an
    ``enabled`` flag.  Defaults match the standard metric set.

    Metrics whose runtime requirements are not met (e.g. ``power_spectrum``
    on a non-spherical grid) are skipped with a warning when ``strict``
    is ``False`` (the default for built-in metrics).

    Parameters:
        mean_denorm: Mean metrics on denormalized data.
        mean_norm: Mean metrics on normalized data.
        power_spectrum: Spherical power spectrum metrics.
        snapshot: Snapshot image metrics.
        mean_map: Mean map image metrics.
        ensemble: Ensemble spread metrics.
    """

    mean_denorm: OneStepMeanMetricConfig = dataclasses.field(
        default_factory=lambda: OneStepMeanMetricConfig(target="denorm")
    )
    mean_norm: OneStepMeanMetricConfig = dataclasses.field(
        default_factory=lambda: OneStepMeanMetricConfig(
            target="norm",
            include_bias=False,
            include_grad_mag_percent_diff=False,
        )
    )
    power_spectrum: OneStepSpectrumMetricConfig = dataclasses.field(
        default_factory=OneStepSpectrumMetricConfig
    )
    snapshot: OneStepSnapshotMetricConfig = dataclasses.field(
        default_factory=OneStepSnapshotMetricConfig
    )
    mean_map: OneStepMapMetricConfig = dataclasses.field(
        default_factory=OneStepMapMetricConfig
    )
    ensemble: OneStepEnsembleMetricConfig = dataclasses.field(
        default_factory=OneStepEnsembleMetricConfig
    )

    def __post_init__(self):
        if self.mean_denorm.target != "denorm":
            raise ValueError(
                f"mean_denorm.target must be 'denorm', "
                f"got '{self.mean_denorm.target}'"
            )
        if self.mean_norm.target != "norm":
            raise ValueError(
                f"mean_norm.target must be 'norm', got '{self.mean_norm.target}'"
            )

    def _get_metrics(self) -> list[OneStepMetricConfig]:
        all_metrics: list[OneStepMetricConfig] = [
            self.mean_denorm,
            self.mean_norm,
            self.power_spectrum,
            self.snapshot,
            self.mean_map,
            self.ensemble,
        ]
        return [m for m in all_metrics if m.enabled]

    def build(
        self,
        dataset_info: DatasetInfo,
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        loss_scaling: TensorMapping | None = None,
        channel_mean_names: Sequence[str] | None = None,
    ) -> OneStepAggregator:
        return build_one_step_aggregator(
            metrics=self._get_metrics(),
            dataset_info=dataset_info,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
            loss_scaling=loss_scaling,
            channel_mean_names=channel_mean_names,
            raise_on_unsupported=False,
        )


@dataclasses.dataclass
class LegacyFlagOneStepAggregatorConfig:
    """
    Legacy configuration for the validation OneStepAggregator using boolean flags.

    Deprecated: Use OneStepAggregatorConfig with typed metrics instead.

    Arguments:
        log_snapshots: Whether to log snapshot images.
        log_mean_maps: Whether to log mean map images.
    """

    log_snapshots: bool = True
    log_mean_maps: bool = True

    def __post_init__(self):
        warnings.warn(
            "LegacyFlagOneStepAggregatorConfig is deprecated. "
            "Use OneStepAggregatorConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def _get_metrics(self) -> list[OneStepMetricConfig]:
        metrics: list[OneStepMetricConfig] = [
            OneStepMeanMetricConfig(target="denorm"),
            OneStepMeanMetricConfig(
                target="norm",
                include_bias=False,
                include_grad_mag_percent_diff=False,
            ),
            OneStepSpectrumMetricConfig(),
        ]
        if self.log_snapshots:
            metrics.append(OneStepSnapshotMetricConfig())
        if self.log_mean_maps:
            metrics.append(OneStepMapMetricConfig())
        metrics.append(OneStepEnsembleMetricConfig(log_mean_maps=self.log_mean_maps))
        return metrics

    def build(
        self,
        dataset_info: DatasetInfo,
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        loss_scaling: TensorMapping | None = None,
        channel_mean_names: Sequence[str] | None = None,
    ) -> OneStepAggregator:
        return build_one_step_aggregator(
            metrics=self._get_metrics(),
            dataset_info=dataset_info,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
            loss_scaling=loss_scaling,
            channel_mean_names=channel_mean_names,
        )
