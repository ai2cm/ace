import dataclasses
import logging
import os
from typing import Any, Dict, Optional, Union

from fme.core.aggregator import InferenceEvaluatorAggregatorConfig
from fme.core.data_loading.config import DataLoaderConfig, Slice
from fme.core.data_loading.inference import InferenceDataLoaderConfig
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import OptimizationConfig
from fme.core.stepper import ExistingStepperConfig, SingleModuleStepperConfig
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
    def checkpoint_dir(self) -> str:
        """
        The directory where checkpoints are saved.
        """
        return os.path.join(self.experiment_dir, "training_checkpoints")

    @property
    def latest_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "ckpt.tar")

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "best_ckpt.tar")

    @property
    def best_inference_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "best_inference_ckpt.tar")

    @property
    def ema_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "ema_ckpt.tar")

    def epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ckpt_{epoch:04d}.tar")

    def ema_epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ema_ckpt_{epoch:04d}.tar")

    def epoch_checkpoint_enabled(self, epoch: int) -> bool:
        return epoch_checkpoint_enabled(
            epoch, self.max_epochs, self.checkpoint_save_epochs
        )

    def ema_epoch_checkpoint_enabled(self, epoch: int) -> bool:
        return epoch_checkpoint_enabled(
            epoch, self.max_epochs, self.ema_checkpoint_save_epochs
        )

    @property
    def resuming(self) -> bool:
        checkpoint_file_exists = os.path.isfile(self.latest_checkpoint_path)
        resuming = True if checkpoint_file_exists else False
        return resuming

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, env_vars: Optional[Dict[str, Any]] = None, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_wandb(config=config, env_vars=env_vars, **kwargs)

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info(str(self))
        logging.info("---------------------------------------------------")

    def clean_wandb(self):
        self.logging.clean_wandb(experiment_dir=self.experiment_dir)


def epoch_checkpoint_enabled(
    epoch: int, max_epochs: int, save_epochs: Optional[Slice]
) -> bool:
    if save_epochs is None:
        return False
    return epoch in range(max_epochs)[save_epochs.slice]
