import dataclasses
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from fme.ace.aggregator import (
    InferenceEvaluatorAggregatorConfig,
    OneStepAggregatorConfig,
)
from fme.ace.aggregator.inference.main import InferenceEvaluatorAggregator
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.getters import get_gridded_data, get_inference_data
from fme.ace.data_loading.gridded_data import (
    ErrorInferenceData,
    GriddedData,
    InferenceGriddedData,
)
from fme.ace.data_loading.inference import InferenceDataLoaderConfig
from fme.ace.requirements import (
    DataRequirements,
    NullDataRequirements,
    PrognosticStateDataRequirements,
)
from fme.ace.stepper import ExistingStepperConfig, SingleModuleStepperConfig, Stepper
from fme.ace.stepper.single_module import StepperConfig
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.trainer import EndOfBatchCallback, EndOfEpochCallback
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.typing_ import Slice, TensorDict, TensorMapping
from fme.core.weight_ops import CopyWeightsConfig


@dataclasses.dataclass
class WeatherEvaluationConfig:
    """
    Parameters:
        loader: configuration for the data loader used during weather evaluation
        n_forward_steps: number of forward steps to take
        forward_steps_in_memory: number of forward steps to take before
            re-reading data from disk
        epochs: epochs on which to run weather evaluation. By default runs
            weather evaluation every epoch.
        aggregator: configuration of weather evaluation aggregator.
    """

    loader: InferenceDataLoaderConfig
    n_forward_steps: int = 2
    forward_steps_in_memory: int = 2
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

    def get_inference_data(
        self,
        window_requirements: DataRequirements,
        initial_condition: PrognosticStateDataRequirements,
    ) -> InferenceGriddedData:
        return get_inference_data(
            config=self.loader,
            total_forward_steps=self.n_forward_steps,
            window_requirements=window_requirements,
            initial_condition=initial_condition,
        )


@dataclasses.dataclass
class InlineInferenceConfig:
    """
    Parameters:
        loader: configuration for the data loader used during inference
        n_forward_steps: number of forward steps to take
        forward_steps_in_memory: number of forward steps to take before
            re-reading data from disk
        epochs: epochs on which to run inference. By default runs inference every epoch.
        aggregator: configuration of inline inference aggregator.
    """

    loader: InferenceDataLoaderConfig
    n_forward_steps: int = 2
    forward_steps_in_memory: int = 2
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

    def get_inference_data(
        self,
        window_requirements: DataRequirements,
        initial_condition: PrognosticStateDataRequirements,
    ) -> InferenceGriddedData:
        return get_inference_data(
            config=self.loader,
            total_forward_steps=self.n_forward_steps,
            window_requirements=window_requirements,
            initial_condition=initial_condition,
        )


@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for training a model.

    Arguments:
        train_loader: Configuration for the training data loader.
        validation_loader: Configuration for the validation data loader.
        stepper: Configuration for the stepper. SingleModuleStepperConfig is
            deprecated and will be removed in a future version. Use StepperConfig
            instead.
        optimization: Configuration for the optimization.
        logging: Configuration for logging.
        max_epochs: Total number of epochs to train for.
        save_checkpoint: Whether to save checkpoints.
        experiment_dir: Directory where checkpoints and logs are saved.
        inference: Configuration for inline inference.
            If None, no inline inference is run,
            and no "best_inline_inference" checkpoint will be saved.
        weather_evaluation: Configuration for weather evaluation.
            If None, no weather evaluation is run. Weather evaluation is not
            used to select checkpoints, but is used to provide metrics.
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
        validation_aggregator: Configuration for the validation aggregator.
        evaluate_before_training: Whether to run validation and inline inference before
            any training is done.
    """

    train_loader: DataLoaderConfig
    validation_loader: DataLoaderConfig
    stepper: SingleModuleStepperConfig | ExistingStepperConfig | StepperConfig
    optimization: OptimizationConfig
    logging: LoggingConfig
    max_epochs: int
    save_checkpoint: bool
    experiment_dir: str
    inference: InlineInferenceConfig | None
    n_forward_steps: int
    copy_weights_after_batch: list[CopyWeightsConfig] = dataclasses.field(
        default_factory=list
    )
    ema: EMAConfig = dataclasses.field(default_factory=lambda: EMAConfig())
    weather_evaluation: WeatherEvaluationConfig | None = None
    validate_using_ema: bool = False
    checkpoint_save_epochs: Slice | None = None
    ema_checkpoint_save_epochs: Slice | None = None
    log_train_every_n_batches: int = 100
    segment_epochs: int | None = None
    save_per_epoch_diagnostics: bool = False
    validation_aggregator: OneStepAggregatorConfig = dataclasses.field(
        default_factory=lambda: OneStepAggregatorConfig()
    )
    evaluate_before_training: bool = False

    def __post_init__(self):
        if isinstance(self.stepper, SingleModuleStepperConfig):
            warnings.warn(
                "SingleModuleStepperConfig is deprecated. Use StepperConfig instead.",
                DeprecationWarning,
            )

    @property
    def inference_n_forward_steps(self) -> int:
        if self.inference is None:
            return 0
        return self.inference.n_forward_steps

    @property
    def inference_aggregator(self) -> InferenceEvaluatorAggregatorConfig | None:
        if self.inference is None:
            return None
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
        if self.inference is None:
            return []
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
        if self.config.inference is None:
            return NullDataRequirements
        return self.config.stepper.get_evaluation_window_data_requirements(
            self.config.inference.forward_steps_in_memory
        )

    def _get_initial_condition_data_requirements(
        self,
    ) -> PrognosticStateDataRequirements:
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
        if self.config.inference is None:
            return ErrorInferenceData()  # type: ignore
        else:
            return self.config.inference.get_inference_data(
                window_requirements=self._get_evaluation_window_data_requirements(),
                initial_condition=self._get_initial_condition_data_requirements(),
            )

    def get_optimization(self, modules: torch.nn.ModuleList) -> Optimization:
        return self.config.optimization.build(modules, self.config.max_epochs)

    def get_stepper(
        self,
        dataset_info: DatasetInfo,
    ) -> Stepper:
        return self.config.stepper.get_stepper(
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

    def get_end_of_epoch_callback(
        self,
        inference_one_epoch: Callable[
            [InferenceGriddedData, InferenceEvaluatorAggregator, str, int],
            Mapping[str, Any],
        ],
        normalize: Callable[[TensorMapping], TensorDict],
        output_dir: str,
        variable_metadata: Mapping[str, VariableMetadata],
        channel_mean_names: Sequence[str],
        save_diagnostics: bool,
        n_ic_timesteps: int,
    ) -> EndOfEpochCallback:
        if self.config.weather_evaluation is not None:
            data = self.config.weather_evaluation.get_inference_data(
                window_requirements=self._get_evaluation_window_data_requirements(),
                initial_condition=self._get_initial_condition_data_requirements(),
            )
            dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)
            aggregator = self.config.weather_evaluation.aggregator.build(
                dataset_info=dataset_info,
                n_timesteps=self.config.weather_evaluation.n_forward_steps
                + n_ic_timesteps,
                initial_time=data.initial_time,
                normalize=normalize,
                output_dir=output_dir,
                record_step_20=self.config.weather_evaluation.n_forward_steps >= 20,
                channel_mean_names=channel_mean_names,
                save_diagnostics=save_diagnostics,
            )

            def end_of_epoch_ops(epoch: int) -> Mapping[str, Any]:
                if self.config.weather_evaluation is not None:
                    if self.config.weather_evaluation.epochs.contains(epoch):
                        return inference_one_epoch(
                            data,
                            aggregator,
                            "weather_eval",
                            epoch,
                        )
                return {}

            return end_of_epoch_ops

        return lambda epoch: {}
