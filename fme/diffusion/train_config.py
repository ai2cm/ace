import dataclasses
import datetime
import os
from collections.abc import Callable, Mapping
from typing import Any

import torch

from fme.ace.aggregator import InferenceEvaluatorAggregatorConfig
from fme.ace.aggregator.one_step.main import OneStepAggregator
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.getters import get_data_loader, get_inference_data
from fme.ace.data_loading.gridded_data import GriddedData, InferenceGriddedData
from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements
from fme.ace.train.train_config import InlineInferenceConfig
from fme.core.coordinates import VerticalCoordinate
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.trainer import EndOfBatchCallback, EndOfEpochCallback
from fme.core.gridded_ops import GriddedOperations
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.timing import GlobalTimer
from fme.core.typing_ import Slice
from fme.core.weight_ops import CopyWeightsConfig
from fme.diffusion.stepper import DiffusionStepper, DiffusionStepperConfig


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
        save_per_epoch_diagnostics: Whether to save per-epoch diagnostics from
            training, validation and inline inference aggregators.
        evaluate_before_training: Whether to run validation and inline inference before
            any training is done.
    """

    train_loader: DataLoaderConfig
    validation_loader: DataLoaderConfig
    stepper: DiffusionStepperConfig
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
    checkpoint_save_epochs: Slice | None = None
    ema_checkpoint_save_epochs: Slice | None = None
    log_train_every_n_batches: int = 100
    segment_epochs: int | None = None
    save_per_epoch_diagnostics: bool = False
    evaluate_before_training: bool = False

    def __post_init__(self):
        if self.n_forward_steps != 1:
            raise NotImplementedError("Only n_forward_steps=1 is currently supported")

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

    @property
    def output_dir(self) -> str:
        """
        The directory where output files are saved.
        """
        return os.path.join(self.experiment_dir, "output")

    def get_inference_epochs(self) -> list[int]:
        start_epoch = 0 if self.evaluate_before_training else 1
        all_epochs = list(range(start_epoch, self.max_epochs + 1))
        return all_epochs[self.inference.epochs.slice]


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
        img_shape: tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        timestep: datetime.timedelta,
    ) -> DiffusionStepper:
        return self.config.stepper.get_stepper(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=timestep,
        )

    def get_ema(self, modules) -> EMATracker:
        return self.config.ema.build(modules)

    def get_end_of_batch_ops(
        self,
        modules: list[torch.nn.Module],
        base_weights: list[Mapping[str, Any]] | None,
    ) -> EndOfBatchCallback:
        if base_weights is not None:
            copy_after_batch = self.config.copy_weights_after_batch
            return lambda: copy_after_batch.apply(weights=base_weights, modules=modules)
        return lambda: None

    def get_end_of_epoch_ops(
        self,
        stepper: DiffusionStepper,
        validation_data: GriddedData,
        get_validation_aggregator: Callable[[], OneStepAggregator],
    ) -> EndOfEpochCallback:
        def end_of_epoch_ops(epoch: int) -> Mapping[str, Any]:
            aggregator = get_validation_aggregator()
            with torch.no_grad(), GlobalTimer():
                for batch in validation_data.loader:
                    stepped = stepper.generate_on_batch(
                        batch,
                        compute_derived_variables=True,
                    )
                    aggregator.record_batch(
                        batch=stepped,
                    )
            return aggregator.get_logs(label="generation")

        return end_of_epoch_ops
