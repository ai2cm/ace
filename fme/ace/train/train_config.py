import dataclasses
import functools
import os
from collections.abc import Mapping
from typing import Any

import torch

from fme.ace.aggregator import (
    InferenceEvaluatorAggregatorConfig,
    LegacyFlagInferenceEvaluatorAggregatorConfig,
    LegacyFlagOneStepAggregatorConfig,
    OneStepAggregatorConfig,
)
from fme.ace.aggregator.train import TrainAggregatorConfig
from fme.ace.data_loading.batch_data import PrognosticState
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.getters import get_gridded_data, get_inference_data
from fme.ace.data_loading.gridded_data import GriddedData, InferenceGriddedData
from fme.ace.data_loading.inference import InferenceDataLoaderConfig
from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements
from fme.ace.stepper import TrainStepper
from fme.ace.stepper.single_module import (
    CheckpointStepperConfig,
    StepperConfig,
    TrainStepperConfig,
)
from fme.core.cli import ResumeResultsConfig
from fme.core.cloud import is_local
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.lr_tuning import LRTuningConfig
from fme.core.generics.trainer import EndOfBatchCallback
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.rand import set_seed
from fme.core.typing_ import Slice
from fme.core.weight_ops import CopyWeightsConfig


@dataclasses.dataclass
class InlineInferenceConfig:
    """
    Parameters:
        loader: configuration for the data loader used during inference
        n_forward_steps: number of forward steps to take
        forward_steps_in_memory: number of forward steps to take before
            re-reading data from disk
        n_ensemble_per_ic: number of initial condition based ensembles
        epochs: epochs on which to run inference. By default runs inference every epoch.
        aggregator: configuration of inline inference aggregator.
        name: name used as wandb log prefix and output subdirectory. If None,
            defaults to "inference" when there is a single inference config
            and "inference_{i}" when there are multiple. Note: adding a second
            unnamed config will rename the first from "inference" to
            "inference_0", changing its wandb keys and output directory.
        weight: weight for this inference's error in the combined checkpoint
            selection metric. Must be non-negative.
    """

    loader: InferenceDataLoaderConfig
    n_forward_steps: int
    forward_steps_in_memory: int
    n_ensemble_per_ic: int = 1
    epochs: Slice = dataclasses.field(default_factory=lambda: Slice())
    aggregator: (
        InferenceEvaluatorAggregatorConfig
        | LegacyFlagInferenceEvaluatorAggregatorConfig
    ) = dataclasses.field(default_factory=lambda: InferenceEvaluatorAggregatorConfig())
    name: str | None = None
    weight: float = 1.0

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(
                f"InlineInferenceConfig weight must be non-negative, got {self.weight}"
            )
        dist = Distributed.get_instance()
        if self.loader.start_indices.n_initial_conditions % dist.world_size != 0:
            raise ValueError(
                "Number of inference initial conditions must be divisible by the "
                "number of parallel workers, got "
                f"{self.loader.start_indices.n_initial_conditions} and "
                f"{dist.world_size}."
            )

    @property
    def using_labels(self) -> bool:
        return self.loader.using_labels

    def get_inference_data(
        self,
        window_requirements: DataRequirements,
        initial_condition: PrognosticStateDataRequirements,
    ) -> InferenceGriddedData:
        data = get_inference_data(
            config=self.loader,
            total_forward_steps=self.n_forward_steps,
            window_requirements=window_requirements,
            initial_condition=initial_condition,
        )
        if self.n_ensemble_per_ic > 1:
            ic = data.initial_condition.as_batch_data()
            data._initial_condition = PrognosticState(
                ic.broadcast_ensemble(self.n_ensemble_per_ic)
            )
        return data


@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for training a model.

    Arguments:
        train_loader: Configuration for the training data loader.
        validation_loader: Configuration for the validation data loader.
        stepper: Configuration for the stepper.
        optimization: Configuration for the optimization.
        logging: Configuration for logging.
        max_epochs: Total number of epochs to train for.
        save_checkpoint: Whether to save checkpoints. If false, no checkpoints
            are saved regardless of other checkpoint configuration settings. If
            true, checkpoints are saved at the end of the training loop, after
            evaluation, and on catching a termination signal.
        experiment_dir: Directory where checkpoints and logs are saved. For the
            time being, this must be a local directory.
        inference: Configuration(s) for inline inference runs. Accepts a single
            InlineInferenceConfig or a list of them. The weighted sum of each
            run's error is used for checkpoint selection. Each entry can specify
            a name (used as wandb log prefix) and weight.
        stepper_training: Training-specific configuration including loss, ensemble
            settings, parameter initialization, and forward step scheduling.
        train_aggregator: Configuration for the train aggregator.
        seed: Random seed for reproducibility. If set, is used for all types of
            randomization, including data shuffling and model initialization.
            If unset, weight initialization is not reproducible but data shuffling is.
        copy_weights_after_batch: Configuration for copying weights from the
            base model to the training model after each batch.
        ema: Configuration for exponential moving average of model weights.
        validate_using_ema: Whether to validate and perform inference using
            the EMA model.
        checkpoint_save_epochs: How often to save epoch-based checkpoints,
            if save_checkpoint is True. If None, checkpoints are only saved
            for the most recent epoch
            (and the best epochs if validate_using_ema == False).
        ema_checkpoint_save_epochs: How often to save epoch-based EMA checkpoints,
            if save_checkpoint is True. If None, EMA checkpoints are only saved
            for the most recent epoch
            (and the best epochs if validate_using_ema == True).
        log_train_every_n_batches: How often to log batch_loss during training.
        train_evaluation_samples: Number of samples to evaluate on after training
            on each epoch. The remainder samples after dividing by the batch size
            are discarded.
        checkpoint_every_n_batches: How often to save latest checkpoint during training.
            If 0 is given, checkpoints will not be saved based on batch progress,
            only other factors like pre-emption or being at the end of an epoch.
        segment_epochs: Exit after training for at most this many epochs
            in current job, without exceeding `max_epochs`. Use this if training
            must be run in segments, e.g. due to wall clock limit.
        save_per_epoch_diagnostics: Whether to save per-epoch diagnostics from
            training, validation and inline inference aggregators.
        validation_aggregator: Configuration for the validation aggregator.
        evaluate_before_training: Whether to run validation and inline inference before
            any training is done.
        save_best_inference_epoch_checkpoints: Whether to save a separate checkpoint
            for each epoch where best_inference_error achieves a new minimum.
            Checkpoints are saved as best_inference_ckpt_XXXX.tar.
        resume_results:  Configuration for resuming a previously stopped or finished
            training job. When provided and experiment_dir has no training_checkpoints
            subdirectory, then it is assumed that this is a new run to resume a
            previously completed run and resume_results.existing_dir is recursively
            copied to experiment_dir.
    """

    train_loader: DataLoaderConfig
    validation_loader: DataLoaderConfig
    stepper: StepperConfig | CheckpointStepperConfig
    optimization: OptimizationConfig
    logging: LoggingConfig
    max_epochs: int
    save_checkpoint: bool
    experiment_dir: str
    inference: InlineInferenceConfig | list[InlineInferenceConfig] = dataclasses.field(
        default_factory=list
    )
    stepper_training: TrainStepperConfig = dataclasses.field(
        default_factory=lambda: TrainStepperConfig()
    )
    train_aggregator: TrainAggregatorConfig = dataclasses.field(
        default_factory=lambda: TrainAggregatorConfig()
    )
    seed: int | None = None
    copy_weights_after_batch: list[CopyWeightsConfig] = dataclasses.field(
        default_factory=list
    )
    ema: EMAConfig = dataclasses.field(default_factory=lambda: EMAConfig())
    validate_using_ema: bool = False
    checkpoint_save_epochs: Slice | None = None
    ema_checkpoint_save_epochs: Slice | None = None
    log_train_every_n_batches: int = 100
    train_evaluation_samples: int = 1000
    checkpoint_every_n_batches: int = 1000
    segment_epochs: int | None = None
    save_per_epoch_diagnostics: bool = False
    validation_aggregator: (
        OneStepAggregatorConfig | LegacyFlagOneStepAggregatorConfig
    ) = dataclasses.field(default_factory=lambda: OneStepAggregatorConfig())
    evaluate_before_training: bool = False
    save_best_inference_epoch_checkpoints: bool = False
    lr_tuning: LRTuningConfig | None = None
    resume_results: ResumeResultsConfig | None = None

    @functools.cached_property
    def stepper_config(self) -> StepperConfig:
        if isinstance(self.stepper, CheckpointStepperConfig):
            return self.stepper.to_stepper_config()
        return self.stepper

    _RESERVED_NAMES = {"train", "val"}

    def __post_init__(self):
        if self.train_loader.using_labels != self.validation_loader.using_labels:
            raise ValueError(
                "train_loader and validation_loader must both use labels or both not "
                "use labels"
            )
        resolved_names = self.inference_names
        if len(resolved_names) != len(set(resolved_names)):
            raise ValueError(f"Duplicate inference names: {resolved_names}")
        reserved_overlap = set(resolved_names) & self._RESERVED_NAMES
        if reserved_overlap:
            raise ValueError(
                f"Inference names {sorted(reserved_overlap)} collide with "
                f"reserved names {sorted(self._RESERVED_NAMES)}"
            )
        for i, entry in enumerate(self.inference_list):
            if self.train_loader.using_labels != entry.using_labels:
                name = resolved_names[i]
                raise ValueError(
                    f"train_loader and inference {name!r} loader "
                    "must both use labels or both not use labels"
                )
        if self.lr_tuning is not None and self.optimization.has_lr_schedule:
            raise ValueError(
                "lr_tuning and optimization.scheduler cannot both be specified; "
                "lr_tuning is an alternative form of learning rate scheduling"
            )
        if not is_local(self.experiment_dir):
            raise ValueError(
                f"During training, experiment_dir must currently be a local "
                f"directory, got {self.experiment_dir!r}."
            )
        self._validate_weighted_inference_epochs()
        if self.stepper_training.n_forward_steps is None:
            raise ValueError(
                "n_forward_steps must be specified in stepper_training "
                "to determine data loading requirements."
            )
        if self.stepper_training.n_forward_steps_schedule is None:
            raise RuntimeError(
                "expected n_forward_steps_schedule to be defined when "
                "n_forward_steps is not None, is there a bug?"
            )

    def _validate_weighted_inference_epochs(self):
        epoch_sets = self.get_inference_epoch_sets()
        weighted_epoch_set: set[int] | None = None
        for entry, epoch_set in zip(self.inference_list, epoch_sets):
            if entry.weight > 0:
                if weighted_epoch_set is None:
                    weighted_epoch_set = epoch_set
                elif epoch_set != weighted_epoch_set:
                    raise ValueError(
                        "All inference entries with weight > 0 must share the same "
                        "epoch schedule, so that the weighted checkpoint selection "
                        "metric is comparable across epochs. Use weight=0 for "
                        "supplementary entries that run on different epochs."
                    )

    def set_random_seed(self):
        if self.seed is not None:
            set_seed(self.seed)

    @property
    def train_evaluation_batches(self) -> int:
        return self.train_evaluation_samples // self.train_loader.batch_size

    @property
    def inference_list(self) -> list[InlineInferenceConfig]:
        if isinstance(self.inference, InlineInferenceConfig):
            return [self.inference]
        return self.inference

    @property
    def inference_names(self) -> list[str]:
        inference = self.inference_list
        names = []
        for i, entry in enumerate(inference):
            if entry.name is not None:
                names.append(entry.name)
            elif len(inference) == 1:
                names.append("inference")
            else:
                names.append(f"inference_{i}")
        return names

    @property
    def checkpoint_dir(self) -> str:
        """
        The directory where checkpoints are saved.
        """
        return os.path.join(self.experiment_dir, "training_checkpoints")

    @property
    def output_dir(self) -> str:
        """
        The directory where output files are saved.
        """
        return os.path.join(self.experiment_dir, "output")

    def get_inference_epoch_sets(self) -> list[set[int]]:
        inference = self.inference_list
        if not inference:
            return []
        start_epoch = 0 if self.evaluate_before_training else 1
        all_epochs = list(range(start_epoch, self.max_epochs + 1))
        return [set(all_epochs[entry.epochs.slice]) for entry in inference]

    def get_inference_epochs(self) -> list[int]:
        epoch_sets = self.get_inference_epoch_sets()
        if not epoch_sets:
            return []
        return sorted(set().union(*epoch_sets))


class TrainBuilders:
    def __init__(self, config: TrainConfig):
        self.config = config

    def _get_n_forward_steps(self) -> int | IntSchedule:
        """Get n_forward_steps for data loading requirements."""
        if self.config.stepper_training.n_forward_steps_schedule is None:
            raise ValueError(
                "n_forward_steps must be specified in stepper_training "
                "to determine data loading requirements."
            )
        schedule = self.config.stepper_training.n_forward_steps_schedule
        return schedule.max_n_forward_steps

    def _get_train_window_data_requirements(self) -> DataRequirements:
        n_forward_steps = self._get_n_forward_steps()
        return self.config.stepper_config.get_evaluation_window_data_requirements(
            n_forward_steps
        )

    def get_train_data(self) -> GriddedData:
        data_requirements = self._get_train_window_data_requirements()
        return get_gridded_data(
            self.config.train_loader,
            requirements=data_requirements,
            train=True,
        )

    def get_validation_data(self) -> GriddedData:
        data_requirements = self._get_train_window_data_requirements()
        return get_gridded_data(
            self.config.validation_loader,
            requirements=data_requirements,
            train=False,
        )

    def get_inference_data(
        self,
        variable_metadata: Mapping[str, VariableMetadata],
    ) -> list[tuple[InlineInferenceConfig, InferenceGriddedData, DatasetInfo, str]]:
        names = self.config.inference_names
        entries: list[
            tuple[InlineInferenceConfig, InferenceGriddedData, DatasetInfo, str]
        ] = []
        for entry, name in zip(self.config.inference_list, names):
            window_requirements = (
                self.config.stepper_config.get_evaluation_window_data_requirements(
                    entry.forward_steps_in_memory
                )
            )
            data = entry.get_inference_data(
                window_requirements=window_requirements,
                initial_condition=self.config.stepper_config.get_prognostic_state_data_requirements(),
            )
            dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)
            entries.append((entry, data, dataset_info, name))
        return entries

    def get_optimization(self, modules: torch.nn.ModuleList) -> Optimization:
        return self.config.optimization.build(modules, self.config.max_epochs)

    def get_stepper(
        self,
        dataset_info: DatasetInfo,
    ) -> TrainStepper:
        """
        Get the training stepper.

        Creates a Stepper for inference and wraps it in a TrainStepper with
        training-specific configuration including the loss and parameter
        initialization.

        """
        return self.config.stepper_training.get_train_stepper(
            stepper_config=self.config.stepper_config,
            dataset_info=dataset_info,
        )

    def get_ema(self, modules) -> EMATracker:
        return self.config.ema.build(modules)

    def get_end_of_batch_ops(
        self,
        modules: list[torch.nn.Module],
        base_weights: list[Mapping[str, Any]] | None,
    ) -> EndOfBatchCallback:
        if base_weights is not None:

            def copy_after_batch():
                for module, copy_config in zip(
                    modules, self.config.copy_weights_after_batch
                ):
                    copy_config.apply(weights=base_weights, modules=[module])
                return

            return copy_after_batch
        return lambda: None
