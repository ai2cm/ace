import dataclasses
import logging
import os
from collections.abc import Callable, Sequence

import torch

from fme.core.cli import ResumeResultsConfig
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.lr_tuning import LRTuningConfig, ValidateStepper
from fme.core.generics.train_stepper import TrainStepperABC
from fme.core.generics.trainer import (
    AggregatorBuilderABC,
    EndOfBatchCallback,
    InferenceCallback,
    InferenceTask,
    Trainer,
    TrainerParams,
    ValidationCallback,
    ValidationTask,
    build_inference_callback,
    build_validation_callback,
)
from fme.core.generics.validation import run_validation_loop
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.rand import set_seed
from fme.core.typing_ import Slice
from fme.core.weight_ops import CopyWeightsConfig
from fme.coupled.aggregator import (
    InferenceEvaluatorAggregator,
    InferenceEvaluatorAggregatorConfig,
    OneStepAggregator,
    TrainAggregator,
)
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
    CoupledTrainOutput,
    CoupledTrainStepper,
    CoupledTrainStepperConfig,
)
from fme.coupled.typing_ import CoupledOptionalInt, CoupledTensorMapping


def _validate_n_steps(
    n_coupled_steps: int,
    n_inner_steps: int,
    component_n_steps_max: CoupledOptionalInt,
) -> None:
    """Ensure each component's ``n_steps`` upper bound fits within the rollout
    horizon implied by ``n_coupled_steps`` and the atmosphere/ocean step ratio.

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
            f"ocean n_steps max "
            f"({component_n_steps_max.ocean}) exceeds n_coupled_steps "
            f"({n_coupled_steps})."
        )
    if (
        component_n_steps_max.atmosphere is not None
        and component_n_steps_max.atmosphere > atmos_limit
    ):
        errors.append(
            f"atmosphere n_steps max "
            f"({component_n_steps_max.atmosphere}) exceeds n_coupled_steps * "
            f"n_inner_steps ({n_coupled_steps} * {n_inner_steps} = "
            f"{atmos_limit})."
        )
    if errors:
        raise ValueError("Incompatible n_steps: " + " ".join(errors))


@dataclasses.dataclass
class InlineValidationConfig:
    """
    Parameters:
        loader: configuration for the data loader used during validation
        name: name used as wandb log prefix and output subdirectory. If None,
            defaults to "val" when there is a single validation config
            and "val_{i}" when there are multiple. Note: adding a second
            unnamed config will rename the first from "val" to
            "val_0", changing its wandb keys and output directory.
        weight: weight for this validation's loss in the combined checkpoint
            selection metric. Must be non-negative.
    """

    loader: CoupledDataLoaderConfig
    name: str | None = None
    weight: float = 1.0

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(
                f"InlineValidationConfig weight must be non-negative, got {self.weight}"
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

    def build_aggregator_factory(
        self,
        data: InferenceGriddedData,
        name: str,
        stepper: CoupledTrainStepper,
        dataset_info: CoupledDatasetInfo,
        output_dir: str,
        save_per_epoch_diagnostics: bool,
    ) -> Callable[[], InferenceEvaluatorAggregator]:
        def factory():
            batch = next(iter(data.loader))
            anchor_data = batch.ocean_data or batch.ice_data or batch.atmosphere_data
            assert anchor_data is not None, "Batch has no component data"
            initial_times = anchor_data.time.isel(time=0)
            n_timesteps_ocean = self.n_coupled_steps + stepper.ocean.n_ic_timesteps
            n_timesteps_atmosphere = (
                self.n_coupled_steps * stepper.n_inner_steps
                + stepper.atmosphere.n_ic_timesteps
            )
            return self.aggregator.build(
                dataset_info=dataset_info,
                n_timesteps_ocean=n_timesteps_ocean,
                n_timesteps_atmosphere=n_timesteps_atmosphere,
                initial_time=initial_times,
                ocean_normalize=stepper.ocean.normalizer.normalize,
                atmosphere_normalize=stepper.atmosphere.normalizer.normalize,
                save_diagnostics=save_per_epoch_diagnostics,
                output_dir=os.path.join(output_dir, name),
            )

        return factory


def _get_validation_callback(
    validation_entries: Sequence[tuple[InlineValidationConfig, GriddedData, str]],
    stepper: TrainStepperABC,
    dataset_info: CoupledDatasetInfo,
    loss_scaling: CoupledTensorMapping,
    save_per_epoch_diagnostics: bool,
    output_dir: str,
) -> ValidationCallback:
    def make_factory(name: str) -> Callable[[], OneStepAggregator]:
        def factory():
            return OneStepAggregator(
                dataset_info=dataset_info,
                save_diagnostics=save_per_epoch_diagnostics,
                output_dir=os.path.join(output_dir, name),
                loss_scaling=loss_scaling,
            )

        return factory

    tasks: list[ValidationTask] = [
        ValidationTask(
            name=name,
            data=data,
            aggregator_factory=make_factory(name),
            weight=entry_config.weight,
        )
        for entry_config, data, name in validation_entries
    ]
    return build_validation_callback(tasks=tasks, stepper=stepper)


def _get_validate_stepper_callback(
    validation_entries: Sequence[tuple[InlineValidationConfig, GriddedData, str]],
    dataset_info: CoupledDatasetInfo,
    loss_scaling: CoupledTensorMapping,
    validate_using_ema: bool,
) -> ValidateStepper:
    # LR tuning passes trial stepper/EMA instances distinct from the Trainer's
    # own stepper, so this callback manages its own EMA via run_validation_loop
    # rather than relying on the Trainer's validation_context().
    def validate_stepper(
        stepper: TrainStepperABC, ema: EMATracker, epoch: int
    ) -> float:
        weighted_loss = 0.0
        for entry_config, data, name in validation_entries:
            data.set_epoch(epoch)
            aggregator = OneStepAggregator(
                dataset_info=dataset_info,
                save_diagnostics=False,
                output_dir="",
                loss_scaling=loss_scaling,
            )
            run_validation_loop(
                stepper=stepper,
                valid_data=data,
                aggregator=aggregator,
                ema=ema,
                validate_using_ema=validate_using_ema,
            )
            if entry_config.weight > 0:
                summary = aggregator.get_summary(label=name)
                if summary.loss is not None:
                    weighted_loss += entry_config.weight * summary.loss
        return weighted_loss

    return validate_stepper


def _get_inference_callback(
    inference_entries: Sequence[
        tuple[InlineInferenceConfig, InferenceGriddedData, str]
    ],
    inference_epochs: Sequence[int],
    inference_epoch_sets: Sequence[set[int]],
    stepper: CoupledTrainStepper,
    dataset_info: CoupledDatasetInfo,
    output_dir: str,
    save_per_epoch_diagnostics: bool,
) -> InferenceCallback:
    tasks: list[InferenceTask] = []
    for i, (entry_config, data, name) in enumerate(inference_entries):
        tasks.append(
            InferenceTask(
                name=name,
                data=data,
                aggregator_factory=entry_config.build_aggregator_factory(
                    data=data,
                    name=name,
                    stepper=stepper,
                    dataset_info=dataset_info,
                    output_dir=output_dir,
                    save_per_epoch_diagnostics=save_per_epoch_diagnostics,
                ),
                epoch_set=frozenset(inference_epoch_sets[i]),
                weight=entry_config.weight,
            )
        )

    return build_inference_callback(
        tasks=tasks,
        inference_epochs=inference_epochs,
        stepper=stepper,
    )


@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for training a coupled model.

    Attributes:
        train_loader: Configuration for the coupled training data loader.
        validation: Configuration(s) for inline validation runs. Accepts a single
            InlineValidationConfig or a list of them. The weighted sum of each
            run's loss is used for checkpoint selection. Each entry can specify
            a name (used as wandb log prefix) and weight.
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
    validation: InlineValidationConfig | list[InlineValidationConfig]
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
        if not self.validation_list:
            raise ValueError("At least one validation entry is required.")
        resolved_validation_names = self.validation_names
        if len(resolved_validation_names) != len(set(resolved_validation_names)):
            raise ValueError(f"Duplicate validation names: {resolved_validation_names}")
        resolved_inference_names = self.inference_names
        if len(resolved_inference_names) != len(set(resolved_inference_names)):
            raise ValueError(f"Duplicate inference names: {resolved_inference_names}")
        reserved_overlap = set(resolved_inference_names) & self._RESERVED_NAMES
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
        _validate_n_steps(
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
    def validation_list(self) -> list[InlineValidationConfig]:
        if isinstance(self.validation, InlineValidationConfig):
            return [self.validation]
        return self.validation

    @property
    def validation_names(self) -> list[str]:
        validation = self.validation_list
        names = []
        for i, entry in enumerate(validation):
            if entry.name is not None:
                names.append(entry.name)
            elif len(validation) == 1:
                names.append("val")
            else:
                names.append(f"val_{i}")
        return names

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

    def _get_train_window_data_requirements(self) -> CoupledTrainDataRequirements:
        return self.stepper_training.get_train_window_data_requirements(self.stepper)

    def _get_valid_window_data_requirements(self) -> CoupledDataRequirements:
        return self.stepper.get_evaluation_window_data_requirements(
            self.n_coupled_steps
        )

    def _get_train_data(self) -> GriddedData:
        data_requirements = self._get_train_window_data_requirements()
        return get_gridded_train_data(
            self.train_loader,
            requirements=data_requirements,
        )

    def _get_validation_data(
        self,
    ) -> list[tuple[InlineValidationConfig, GriddedData, str]]:
        data_requirements = self._get_valid_window_data_requirements()
        names = self.validation_names
        entries: list[tuple[InlineValidationConfig, GriddedData, str]] = []
        for entry, name in zip(self.validation_list, names):
            data = get_gridded_data(
                entry.loader,
                requirements=data_requirements,
                train=False,
            )
            entries.append((entry, data, name))
        return entries

    def _get_inference_data(
        self,
    ) -> list[tuple[InlineInferenceConfig, InferenceGriddedData, str]]:
        names = self.inference_names
        initial_condition = self.stepper.get_prognostic_state_data_requirements()
        entries: list[tuple[InlineInferenceConfig, InferenceGriddedData, str]] = []
        for entry, name in zip(self.inference_list, names):
            window_requirements = self.stepper.get_evaluation_window_data_requirements(
                entry.coupled_steps_in_memory
            )
            data = get_inference_data(
                config=entry.loader,
                total_coupled_steps=entry.n_coupled_steps,
                window_requirements=window_requirements,
                initial_condition=initial_condition,
            )
            entries.append((entry, data, name))
        return entries

    def _get_optimization(self, parameters) -> Optimization:
        return self.optimization.build(parameters, self.max_epochs)

    def _get_stepper(self, dataset_info: CoupledDatasetInfo) -> CoupledTrainStepper:
        return self.stepper_training.get_train_stepper(
            stepper_config=self.stepper,
            dataset_info=dataset_info,
        )

    def _get_ema(self, modules) -> EMATracker:
        return self.ema.build(modules)

    def _get_end_of_batch_ops(
        self, modules: list[torch.nn.Module]
    ) -> EndOfBatchCallback:
        return lambda: None

    def build_trainer(self) -> Trainer:
        logging.info("Initializing training data loader")
        train_data = self._get_train_data()

        variable_metadata = (
            get_derived_variable_metadata() | train_data.variable_metadata
        )
        dataset_info = train_data.dataset_info.update_variable_metadata(
            variable_metadata
        )

        logging.info("Initializing validation data loaders")
        validation_entries = self._get_validation_data()

        for data, name in zip(
            [train_data] + [data for _, data, _ in validation_entries],
            ["train"] + [name for _, _, name in validation_entries],
        ):
            data.log_info(name)

        if self.inference_list:
            logging.info("Initializing inline inference data loaders")
        else:
            logging.info("Skipping inline inference")
        inference_entries = self._get_inference_data()
        inference_epochs = self.get_inference_epochs()
        inference_epoch_sets = self.get_inference_epoch_sets()

        logging.info("Starting model initialization")
        stepper = self._get_stepper(train_data.dataset_info)
        end_of_batch_ops = self._get_end_of_batch_ops(stepper.modules)

        loss_scaling = stepper.effective_loss_scaling
        aggregator_builder = CoupledAggregatorBuilder(
            dataset_info=dataset_info,
            loss_scaling=loss_scaling,
            save_per_epoch_diagnostics=self.save_per_epoch_diagnostics,
            output_dir=self.output_dir,
        )

        validation_callback = _get_validation_callback(
            validation_entries=validation_entries,
            stepper=stepper,
            dataset_info=dataset_info,
            loss_scaling=loss_scaling,
            save_per_epoch_diagnostics=self.save_per_epoch_diagnostics,
            output_dir=self.output_dir,
        )

        validate_stepper: ValidateStepper | None = None
        if self.lr_tuning is not None:
            validate_stepper = _get_validate_stepper_callback(
                validation_entries=validation_entries,
                dataset_info=dataset_info,
                loss_scaling=loss_scaling,
                validate_using_ema=self.validate_using_ema,
            )

        inference_callback = _get_inference_callback(
            inference_entries=inference_entries,
            inference_epochs=inference_epochs,
            inference_epoch_sets=inference_epoch_sets,
            stepper=stepper,
            dataset_info=dataset_info,
            output_dir=self.output_dir,
            save_per_epoch_diagnostics=self.save_per_epoch_diagnostics,
        )

        return Trainer(
            train_data=train_data,
            stepper=stepper,
            build_optimization=self._get_optimization,
            build_ema=self._get_ema,
            params=self.get_trainer_params(),
            aggregator_builder=aggregator_builder,
            validation_callback=validation_callback,
            end_of_batch_callback=end_of_batch_ops,
            inference_callback=inference_callback,
            validate_stepper=validate_stepper,
        )

    def get_trainer_params(self) -> TrainerParams:
        """Package the scalar training parameters read by the Trainer."""
        return TrainerParams(
            experiment_dir=self.experiment_dir,
            checkpoint_dir=self.checkpoint_dir,
            max_epochs=self.max_epochs,
            save_checkpoint=self.save_checkpoint,
            validate_using_ema=self.validate_using_ema,
            log_train_every_n_batches=self.log_train_every_n_batches,
            train_evaluation_batches=self.train_evaluation_batches,
            checkpoint_every_n_batches=self.checkpoint_every_n_batches,
            segment_epochs=self.segment_epochs,
            checkpoint_save_epochs=self.checkpoint_save_epochs,
            ema_checkpoint_save_epochs=self.ema_checkpoint_save_epochs,
            evaluate_before_training=self.evaluate_before_training,
            save_best_inference_epoch_checkpoints=(
                self.save_best_inference_epoch_checkpoints
            ),
            lr_tuning=self.lr_tuning,
        )


class CoupledAggregatorBuilder(AggregatorBuilderABC[CoupledTrainOutput]):
    def __init__(
        self,
        dataset_info: CoupledDatasetInfo,
        output_dir: str,
        loss_scaling: CoupledTensorMapping,
        save_per_epoch_diagnostics: bool = False,
    ):
        self.dataset_info = dataset_info
        self.output_dir = output_dir
        self.loss_scaling = loss_scaling
        self.save_per_epoch_diagnostics = save_per_epoch_diagnostics

    def get_train_aggregator(self) -> TrainAggregator:
        return TrainAggregator()
