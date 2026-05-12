import dataclasses
import datetime
import os

import torch

from fme.core.cli import ResumeResultsConfig
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.lr_tuning import LRTuningConfig
from fme.core.generics.trainer import EndOfBatchCallback
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.rand import set_seed
from fme.core.typing_ import Slice
from fme.core.weight_ops import CopyWeightsConfig
from fme.coupled.aggregator import InferenceEvaluatorAggregatorConfig
from fme.coupled.data_loading.config import CoupledDataLoaderConfig
from fme.coupled.data_loading.getters import (
    get_gridded_data,
    get_gridded_train_data,
    get_inference_data,
)
from fme.coupled.data_loading.gridded_data import GriddedData, InferenceGriddedData
from fme.coupled.data_loading.inference import InferenceDataLoaderConfig
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.requirements import (
    CoupledDataRequirements,
    CoupledTrainDataRequirements,
)
from fme.coupled.stepper import (
    CoupledStepperConfig,
    CoupledTrainStepper,
    CoupledTrainStepperConfig,
)
from fme.coupled.typing_ import CoupledOptionalInt


def _validate_loss_n_steps(
    n_coupled_steps: int,
    n_inner_steps: int,
    component_n_steps_max: CoupledOptionalInt,
) -> None:
    """Ensure each component's ``LossContributionsConfig.n_steps`` upper bound
    fits within the rollout horizon implied by ``n_coupled_steps`` and the
    atmosphere/ocean step ratio.

    Raises:
        ValueError: If either component's ``n_steps_max`` exceeds its limit.
            The error message lists every misconfigured component.
    """
    atmos_limit = n_coupled_steps * n_inner_steps
    errors: list[str] = []
    if (
        component_n_steps_max.ocean is not None
        and component_n_steps_max.ocean > n_coupled_steps
    ):
        errors.append(
            f"ocean loss_contributions.n_steps max "
            f"({component_n_steps_max.ocean}) exceeds n_coupled_steps "
            f"({n_coupled_steps})."
        )
    if (
        component_n_steps_max.atmosphere is not None
        and component_n_steps_max.atmosphere > atmos_limit
    ):
        errors.append(
            f"atmosphere loss_contributions.n_steps max "
            f"({component_n_steps_max.atmosphere}) exceeds n_coupled_steps * "
            f"n_inner_steps ({n_coupled_steps} * {n_inner_steps} = "
            f"{atmos_limit})."
        )
    if errors:
        raise ValueError(
            "Incompatible LossContributionsConfig n_steps: " + " ".join(errors)
        )


@dataclasses.dataclass
class InlineInferenceConfig:
    """
    Parameters:
        loader: configuration for the data loader used during inference
        n_coupled_steps: number of coupled forward steps to take
        coupled_steps_in_memory: number of coupled forward steps to take before
            re-reading data from disk
        epochs: epochs on which to run inference. By default runs inference every epoch.
        aggregator: configuration of inline coupled inference aggregator.
        name: name used as wandb log prefix and output subdirectory. If None,
            defaults to "inference" when there is a single inference config
            and "inference_{i}" when there are multiple. Note: adding a second
            unnamed config will rename the first from "inference" to
            "inference_0", changing its wandb keys and output directory.
        weight: weight for this inference's error in the combined checkpoint
            selection metric. Must be non-negative.
    """

    loader: InferenceDataLoaderConfig
    n_coupled_steps: int = 2
    coupled_steps_in_memory: int = 2
    epochs: Slice = dataclasses.field(default_factory=lambda: Slice())
    aggregator: InferenceEvaluatorAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceEvaluatorAggregatorConfig(
            log_global_mean_time_series=False, log_global_mean_norm_time_series=False
        )
    )
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
        if (
            self.aggregator.log_global_mean_time_series
            or self.aggregator.log_global_mean_norm_time_series
        ):
            # Both of log_global_mean_time_series and
            # log_global_mean_norm_time_series must be False for inline inference.
            self.aggregator.log_global_mean_time_series = False
            self.aggregator.log_global_mean_norm_time_series = False


@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for training a coupled model.

    Attributes:
        train_loader: Configuration for the coupled training data loader.
        validation_loader: Configuration for the coupled validation data loader.
        stepper: Configuration for the coupled stepper.
        optimization: Configuration for the optimization.
        logging: Configuration for logging.
        max_epochs: Total number of epochs to train for.
        save_checkpoint: Whether to save checkpoints. If false, no checkpoints
            are saved regardless of other checkpoint configuration settings. If
            true, checkpoints are saved at the end of the training loop, after
            evaluation, and on catching a termination signal.
        experiment_dir: Directory where checkpoints and logs are saved.
        inference: Configuration(s) for inline inference runs. Accepts a single
            InlineInferenceConfig or a list of them. The weighted sum of each
            run's error is used for checkpoint selection. Each entry can specify
            a name (used as wandb log prefix) and weight.
        n_coupled_steps: Number of coupled forward steps to take gradient over.
            This is equal to the number of forward steps of the ocean model.
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
        checkpoint_every_n_batches: How often to save checkpoints.
        segment_epochs: Exit after training for at most this many epochs
            in current job, without exceeding `max_epochs`. Use this if training
            must be run in segments, e.g. due to wall clock limit.
        save_per_epoch_diagnostics: Whether to save per-epoch diagnostics from
            training, validation and inline inference aggregators.
        evaluate_before_training: Whether to run validation and inline inference before
            any training is done.
        save_best_inference_epoch_checkpoints: Whether to save a separate checkpoint
            for each epoch where best_inference_error achieves a new minimum.
            Checkpoints are saved as best_inference_ckpt_XXXX.tar.
        resume_results: Configuration for resuming a previously stopped or finished
            training job. When provided and experiment_dir has no training_checkpoints
            subdirectory, then it is assumed that this is a new run to resume a
            previously completed run and resume_results.existing_dir is recursively
            copied to experiment_dir.
    """

    train_loader: CoupledDataLoaderConfig
    validation_loader: CoupledDataLoaderConfig
    stepper: CoupledStepperConfig
    stepper_training: CoupledTrainStepperConfig
    optimization: OptimizationConfig
    logging: LoggingConfig
    max_epochs: int
    save_checkpoint: bool
    experiment_dir: str
    inference: InlineInferenceConfig | list[InlineInferenceConfig] = dataclasses.field(
        default_factory=list
    )
    seed: int | None = None
    copy_weights_after_batch: CopyWeightsConfig = dataclasses.field(
        default_factory=lambda: CopyWeightsConfig(exclude=["*"])
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
    evaluate_before_training: bool = True
    save_best_inference_epoch_checkpoints: bool = False
    lr_tuning: LRTuningConfig | None = None
    resume_results: ResumeResultsConfig | None = None

    _RESERVED_NAMES = {"train", "val"}

    def __post_init__(self):
        resolved_names = self.inference_names
        if len(resolved_names) != len(set(resolved_names)):
            raise ValueError(f"Duplicate inference names: {resolved_names}")
        reserved_overlap = set(resolved_names) & self._RESERVED_NAMES
        if reserved_overlap:
            raise ValueError(
                f"Inference names {sorted(reserved_overlap)} collide with "
                f"reserved names {sorted(self._RESERVED_NAMES)}"
            )
        if self.lr_tuning is not None and self.optimization.has_lr_schedule:
            raise ValueError(
                "lr_tuning and optimization.scheduler cannot both be specified; "
                "lr_tuning is an alternative form of learning rate scheduling"
            )
        _validate_loss_n_steps(
            n_coupled_steps=self.stepper_training.n_coupled_steps,
            n_inner_steps=self.stepper.n_inner_steps,
            component_n_steps_max=self.stepper_training.component_n_steps_max,
        )

    @property
    def n_coupled_steps(self) -> int:
        return self.stepper_training.n_coupled_steps

    @property
    def n_forward_steps(self) -> int:
        return self.n_coupled_steps

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.experiment_dir, "training_checkpoints")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.experiment_dir, "output")

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

    def _get_train_window_data_requirements(self) -> CoupledTrainDataRequirements:
        return self.config.stepper_training.get_train_window_data_requirements(
            self.config.stepper
        )

    def _get_valid_window_data_requirements(self) -> CoupledDataRequirements:
        return self.config.stepper.get_evaluation_window_data_requirements(
            self.config.n_coupled_steps
        )

    def get_train_data(self) -> GriddedData:
        data_requirements = self._get_train_window_data_requirements()
        return get_gridded_train_data(
            self.config.train_loader,
            requirements=data_requirements,
        )

    def get_validation_data(self) -> GriddedData:
        data_requirements = self._get_valid_window_data_requirements()
        return get_gridded_data(
            self.config.validation_loader,
            requirements=data_requirements,
            train=False,
        )

    def get_inference_data(
        self,
    ) -> list[tuple[InlineInferenceConfig, InferenceGriddedData, str]]:
        names = self.config.inference_names
        initial_condition = self.config.stepper.get_prognostic_state_data_requirements()
        entries: list[tuple[InlineInferenceConfig, InferenceGriddedData, str]] = []
        for entry, name in zip(self.config.inference_list, names):
            window_requirements = (
                self.config.stepper.get_evaluation_window_data_requirements(
                    entry.coupled_steps_in_memory
                )
            )
            data = get_inference_data(
                config=entry.loader,
                total_coupled_steps=entry.n_coupled_steps,
                window_requirements=window_requirements,
                initial_condition=initial_condition,
            )
            entries.append((entry, data, name))
        return entries

    def get_optimization(self, parameters) -> Optimization:
        return self.config.optimization.build(parameters, self.config.max_epochs)

    @property
    def atmosphere_timestep(self) -> datetime.timedelta:
        return self.config.stepper.atmosphere_timestep

    @property
    def ocean_timestep(self) -> datetime.timedelta:
        return self.config.stepper.ocean_timestep

    def get_stepper(self, dataset_info: CoupledDatasetInfo) -> CoupledTrainStepper:
        return self.config.stepper_training.get_train_stepper(
            stepper_config=self.config.stepper,
            dataset_info=dataset_info,
        )

    def get_ema(self, modules) -> EMATracker:
        return self.config.ema.build(modules)

    def get_end_of_batch_ops(
        self, modules: list[torch.nn.Module]
    ) -> EndOfBatchCallback:
        return lambda: None
