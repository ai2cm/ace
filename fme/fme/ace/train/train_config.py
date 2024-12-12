import dataclasses
import datetime
import os
from typing import List, Optional, Tuple, Union

import torch

from fme.ace.aggregator import InferenceEvaluatorAggregatorConfig
from fme.ace.data_loading.batch_data import GriddedData, InferenceGriddedData
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.getters import get_data_loader, get_inference_data
from fme.ace.data_loading.inference import InferenceDataLoaderConfig
from fme.ace.requirements import PrognosticStateDataRequirements
from fme.ace.stepper import (
    ExistingStepperConfig,
    SingleModuleStepper,
    SingleModuleStepperConfig,
)
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset.requirements import DataRequirements
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.trainer import EndOfBatchCallback
from fme.core.gridded_ops import GriddedOperations
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import Optimization, OptimizationConfig
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
        epochs: epochs on which to run inference, where the first epoch is
            defined as epoch 0 (unlike in logs which show epochs as starting
            from 1). By default runs inference every epoch.
        aggregator: configuration of inline inference aggregator.
    """

    loader: InferenceDataLoaderConfig
    n_forward_steps: int = 2
    forward_steps_in_memory: int = 2
    epochs: Slice = Slice(start=0, stop=None, step=1)
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
    Configuration for training a model.

    Arguments:
        train_loader: Configuration for the training data loader.
        validation_loader: Configuration for the validation data loader.
        stepper: Configuration for the stepper.
        optimization: Configuration for the optimization.
        logging: Configuration for logging.
        max_epochs: Total number of epochs to train for.
        save_checkpoint: Whether to save checkpoints.
        experiment_dir: Directory where checkpoints and logs are saved.
        inference: Configuration for inline inference.
        n_forward_steps: Number of forward steps to take gradient over.
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
        segment_epochs: Exit after training for at most this many epochs
            in current job, without exceeding `max_epochs`. Use this if training
            must be run in segments, e.g. due to wall clock limit.
    """

    train_loader: DataLoaderConfig
    validation_loader: DataLoaderConfig
    stepper: Union[SingleModuleStepperConfig, ExistingStepperConfig]
    optimization: OptimizationConfig
    logging: LoggingConfig
    max_epochs: int
    save_checkpoint: bool
    experiment_dir: str
    inference: InlineInferenceConfig
    n_forward_steps: int
    copy_weights_after_batch: CopyWeightsConfig = dataclasses.field(
        default_factory=lambda: CopyWeightsConfig(exclude=["*"])
    )
    ema: EMAConfig = dataclasses.field(default_factory=lambda: EMAConfig())
    validate_using_ema: bool = False
    checkpoint_save_epochs: Optional[Slice] = None
    ema_checkpoint_save_epochs: Optional[Slice] = None
    log_train_every_n_batches: int = 100
    segment_epochs: Optional[int] = None

    @property
    def inference_n_forward_steps(self) -> int:
        return self.inference.n_forward_steps

    @property
    def inference_aggregator(self) -> InferenceEvaluatorAggregatorConfig:
        return self.inference.aggregator

    @property
    def checkpoint_dir(self) -> str:
        """
        The directory where checkpoints are saved.
        """
        return os.path.join(self.experiment_dir, "training_checkpoints")

    def clean_wandb(self, experiment_dir: str) -> None:
        self.logging.clean_wandb(experiment_dir=experiment_dir)

    def get_inference_epochs(self) -> List[int]:
        return list(range(0, self.max_epochs))[self.inference.epochs.slice]


class TrainBuilders:
    def __init__(self, config: TrainConfig):
        self.config = config

    def _get_train_window_data_requirements(self) -> DataRequirements:
        return self.config.stepper.get_evaluation_window_data_requirements(
            self.config.n_forward_steps
        )

    def _get_evaluation_window_data_requirements(self) -> DataRequirements:
        return self.config.stepper.get_evaluation_window_data_requirements(
            self.config.inference.forward_steps_in_memory
        )

    def _get_initial_condition_data_requirements(
        self,
    ) -> PrognosticStateDataRequirements:
        return self.config.stepper.get_prognostic_state_data_requirements()

    def get_train_data(self) -> GriddedData:
        data_requirements = self._get_train_window_data_requirements()
        return get_data_loader(
            self.config.train_loader,
            requirements=data_requirements,
            train=True,
        )

    def get_validation_data(self) -> GriddedData:
        data_requirements = self._get_train_window_data_requirements()
        return get_data_loader(
            self.config.validation_loader,
            requirements=data_requirements,
            train=False,
        )

    def get_evaluation_inference_data(
        self,
    ) -> InferenceGriddedData:
        return get_inference_data(
            config=self.config.inference.loader,
            total_forward_steps=self.config.inference_n_forward_steps,
            window_requirements=self._get_evaluation_window_data_requirements(),
            initial_condition=self._get_initial_condition_data_requirements(),
        )

    def get_optimization(self, modules: torch.nn.ModuleList) -> Optimization:
        return self.config.optimization.build(modules, self.config.max_epochs)

    def get_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: HybridSigmaPressureCoordinate,
        timestep: datetime.timedelta,
    ) -> SingleModuleStepper:
        return self.config.stepper.get_stepper(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=timestep,
        )

    def get_ema(self, modules) -> EMATracker:
        return self.config.ema.build(modules)

    def get_end_of_batch_ops(
        self, modules: List[torch.nn.Module]
    ) -> EndOfBatchCallback:
        base_weights = self.config.stepper.get_base_weights()
        if base_weights is not None:
            copy_after_batch = self.config.copy_weights_after_batch
            return lambda: copy_after_batch.apply(weights=base_weights, modules=modules)
        return lambda: None
