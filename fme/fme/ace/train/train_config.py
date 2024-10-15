import abc
import dataclasses
import datetime
import logging
import os
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Tuple, Union

import torch

from fme.core.aggregator import InferenceEvaluatorAggregatorConfig
from fme.core.data_loading.batch_data import GriddedData
from fme.core.data_loading.config import DataLoaderConfig, Slice
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.data_loading.getters import get_data_loader, get_inference_data
from fme.core.data_loading.inference import InferenceDataLoaderConfig
from fme.core.data_loading.requirements import DataRequirements
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.gridded_ops import GriddedOperations
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.stepper import (
    ExistingStepperConfig,
    SingleModuleStepper,
    SingleModuleStepperConfig,
)
from fme.core.weight_ops import CopyWeightsConfig


@dataclasses.dataclass
class InlineInferenceConfig:
    """
    Attributes:
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
            logging.warning(
                "Both of log_global_mean_time_series and "
                "log_global_mean_norm_time_series must be False for inline inference. "
                "Setting them to False."
            )
            self.aggregator.log_global_mean_time_series = False
            self.aggregator.log_global_mean_norm_time_series = False


class EndOfBatchCallback(Protocol):
    def __call__(self) -> None:
        ...


@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for training a model.

    Attributes:
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

    def get_inference_epochs(self) -> List[int]:
        return list(range(0, self.max_epochs))[self.inference.epochs.slice]


class TrainBuildersABC(abc.ABC):
    @abc.abstractmethod
    def get_train_data(self) -> GriddedData:
        ...

    @abc.abstractmethod
    def get_validation_data(self) -> GriddedData:
        ...

    @abc.abstractmethod
    def get_inference_data(self) -> GriddedData:
        ...

    @abc.abstractmethod
    def get_optimization(self, parameters) -> Optimization:
        ...

    @abc.abstractmethod
    def get_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
    ) -> SingleModuleStepper:
        ...

    @abc.abstractmethod
    def get_ema(self, modules) -> EMATracker:
        ...

    @abc.abstractmethod
    def get_end_of_batch_ops(
        self, modules: List[torch.nn.Module]
    ) -> EndOfBatchCallback:
        ...


class TrainBuilders(TrainBuildersABC):
    def __init__(self, config: TrainConfig):
        self.config = config

    def _get_data_requirements(self) -> DataRequirements:
        return self.config.stepper.get_data_requirements(self.config.n_forward_steps)

    def get_train_data(self) -> GriddedData:
        data_requirements = self._get_data_requirements()
        return get_data_loader(
            self.config.train_loader,
            requirements=data_requirements,
            train=True,
        )

    def get_validation_data(self) -> GriddedData:
        data_requirements = self._get_data_requirements()
        return get_data_loader(
            self.config.validation_loader,
            requirements=data_requirements,
            train=False,
        )

    def get_inference_data(self) -> GriddedData:
        data_requirements = self._get_data_requirements()
        inference_data_requirements = dataclasses.replace(data_requirements)
        inference_data_requirements.n_timesteps = (
            self.config.inference.n_forward_steps + 1
        )
        return get_inference_data(
            self.config.inference.loader,
            self.config.inference.forward_steps_in_memory,
            inference_data_requirements,
        )

    def get_optimization(self, parameters) -> Optimization:
        return self.config.optimization.build(parameters, self.config.max_epochs)

    def get_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
    ) -> SingleModuleStepper:
        return self.config.stepper.get_stepper(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            sigma_coordinates=sigma_coordinates,
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


class TrainConfigProtocol(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]

    @property
    def experiment_dir(self) -> str:
        ...

    @property
    def checkpoint_dir(self) -> str:
        ...

    @property
    def max_epochs(self) -> int:
        ...

    @property
    def save_checkpoint(self) -> bool:
        ...

    @property
    def validate_using_ema(self) -> bool:
        ...

    @property
    def n_forward_steps(self) -> int:
        ...

    @property
    def log_train_every_n_batches(self) -> int:
        ...

    @property
    def inference_aggregator(self) -> InferenceEvaluatorAggregatorConfig:
        ...

    @property
    def inference_n_forward_steps(self) -> int:
        ...

    @property
    def segment_epochs(self) -> Optional[int]:
        ...

    @property
    def checkpoint_save_epochs(self) -> Optional[Slice]:
        ...

    @property
    def ema_checkpoint_save_epochs(self) -> Optional[Slice]:
        ...

    @property
    def logging(self) -> LoggingConfig:
        ...

    def get_inference_epochs(self) -> List[int]:
        ...
