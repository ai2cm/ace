import dataclasses
import datetime
import os

import torch

from fme.core.cli import ResumeResultsConfig
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.trainer import EndOfBatchCallback
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.rand import set_seed
from fme.core.typing_ import Slice
from fme.core.weight_ops import CopyWeightsConfig
from fme.coupled.aggregator import InferenceEvaluatorAggregatorConfig
from fme.coupled.data_loading.config import CoupledDataLoaderConfig
from fme.coupled.data_loading.getters import get_gridded_data, get_inference_data
from fme.coupled.data_loading.gridded_data import GriddedData, InferenceGriddedData
from fme.coupled.data_loading.inference import InferenceDataLoaderConfig
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.requirements import (
    CoupledDataRequirements,
    CoupledPrognosticStateDataRequirements,
)
from fme.coupled.stepper import CoupledStepper, CoupledStepperConfig


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

    def __post_init__(self):
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
        save_checkpoint: Whether to save checkpoints.
        experiment_dir: Directory where checkpoints and logs are saved.
        inference: Configuration for inline inference.
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
    optimization: OptimizationConfig
    logging: LoggingConfig
    max_epochs: int
    save_checkpoint: bool
    experiment_dir: str
    inference: InlineInferenceConfig
    n_coupled_steps: int
    seed: int | None = None
    copy_weights_after_batch: CopyWeightsConfig = dataclasses.field(
        default_factory=lambda: CopyWeightsConfig(exclude=["*"])
    )
    ema: EMAConfig = dataclasses.field(default_factory=lambda: EMAConfig())
    validate_using_ema: bool = False
    checkpoint_save_epochs: Slice | None = None
    ema_checkpoint_save_epochs: Slice | None = None
    log_train_every_n_batches: int = 100
    checkpoint_every_n_batches: int = 1000
    segment_epochs: int | None = None
    save_per_epoch_diagnostics: bool = False
    evaluate_before_training: bool = True
    save_best_inference_epoch_checkpoints: bool = False
    resume_results: ResumeResultsConfig | None = None

    @property
    def n_forward_steps(self) -> int:
        return self.n_coupled_steps

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.experiment_dir, "training_checkpoints")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.experiment_dir, "output")

    @property
    def inference_aggregator(self) -> InferenceEvaluatorAggregatorConfig:
        return self.inference.aggregator

    @property
    def inference_n_coupled_steps(self) -> int:
        return self.inference.n_coupled_steps

    def set_random_seed(self):
        if self.seed is not None:
            set_seed(self.seed)

    def get_inference_epochs(self) -> list[int]:
        start_epoch = 0 if self.evaluate_before_training else 1
        all_epochs = list(range(start_epoch, self.max_epochs + 1))
        return all_epochs[self.inference.epochs.slice]


class TrainBuilders:
    def __init__(self, config: TrainConfig):
        self.config = config

    def _get_train_window_data_requirements(self) -> CoupledDataRequirements:
        return self.config.stepper.get_evaluation_window_data_requirements(
            self.config.n_coupled_steps
        )

    def _get_evaluation_window_data_requirements(self) -> CoupledDataRequirements:
        return self.config.stepper.get_evaluation_window_data_requirements(
            self.config.inference.coupled_steps_in_memory
        )

    def _get_initial_condition_data_requirements(
        self,
    ) -> CoupledPrognosticStateDataRequirements:
        return self.config.stepper.get_prognostic_state_data_requirements()

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

    def get_evaluation_inference_data(
        self,
    ) -> InferenceGriddedData:
        return get_inference_data(
            config=self.config.inference.loader,
            total_coupled_steps=self.config.inference.n_coupled_steps,
            window_requirements=self._get_evaluation_window_data_requirements(),
            initial_condition=self._get_initial_condition_data_requirements(),
        )

    def get_optimization(self, parameters) -> Optimization:
        return self.config.optimization.build(parameters, self.config.max_epochs)

    @property
    def atmosphere_timestep(self) -> datetime.timedelta:
        return self.config.stepper.atmosphere_timestep

    @property
    def ocean_timestep(self) -> datetime.timedelta:
        return self.config.stepper.ocean_timestep

    def get_stepper(self, dataset_info: CoupledDatasetInfo) -> CoupledStepper:
        return self.config.stepper.get_stepper(dataset_info)

    def get_ema(self, modules) -> EMATracker:
        return self.config.ema.build(modules)

    def get_end_of_batch_ops(
        self, modules: list[torch.nn.Module]
    ) -> EndOfBatchCallback:
        return lambda: None
