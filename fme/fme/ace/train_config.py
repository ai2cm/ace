import dataclasses
import logging
import os
import warnings
from typing import Any, Mapping, Optional, Union

from fme.core import SingleModuleStepperConfig
from fme.core.aggregator import InferenceAggregatorConfig
from fme.core.data_loading.config import DataLoaderConfig, Slice
from fme.core.data_loading.inference import InferenceDataLoaderConfig
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.ema import EMATracker
from fme.core.optimization import OptimizationConfig
from fme.core.stepper import ExistingStepperConfig
from fme.core.wandb import WandB
from fme.core.weight_ops import CopyWeightsConfig


@dataclasses.dataclass
class LoggingConfig:
    project: str = "ace"
    entity: str = "ai2cm"
    log_to_screen: bool = True
    log_to_file: bool = True
    log_to_wandb: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        self._dist = Distributed.get_instance()

    def configure_logging(self, experiment_dir: str, log_filename: str):
        """
        Configure the global `logging` module based on this LoggingConfig.
        """
        if self.log_to_screen and self._dist.is_root():
            logging.basicConfig(format=self.log_format, level=logging.INFO)
        elif self._dist.is_root():
            logging.basicConfig(level=logging.WARNING)
        else:  # we are not root
            logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger()
        if self.log_to_file and self._dist.is_root():
            if not os.path.exists(experiment_dir):
                raise ValueError(
                    f"experiment directory {experiment_dir} does not exist, "
                    "cannot log files to it"
                )
            log_path = os.path.join(experiment_dir, log_filename)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(self.log_format))
            logger.addHandler(fh)

    def configure_wandb(self, config: Mapping[str, Any], **kwargs):
        # must ensure wandb.configure is called before wandb.init
        wandb = WandB.get_instance()
        wandb.configure(log_to_wandb=self.log_to_wandb)
        wandb.init(
            config=config,
            project=self.project,
            entity=self.entity,
            dir=config["experiment_dir"],
            **kwargs,
        )

    def clean_wandb(self, experiment_dir: str):
        wandb = WandB.get_instance()
        wandb.clean_wandb_dir(experiment_dir=experiment_dir)


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
    aggregator: InferenceAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceAggregatorConfig()
    )

    def __post_init__(self):
        if self.n_forward_steps % self.forward_steps_in_memory != 0:
            raise ValueError(
                "n_forward_steps must be divisible by steps_in_memory, "
                f"got {self.n_forward_steps} and {self.forward_steps_in_memory}"
            )
        dist = Distributed.get_instance()
        if self.loader.start_indices.n_initial_conditions % dist.world_size != 0:
            raise ValueError(
                "Number of inference initial conditions must be divisible by the "
                "number of parallel workers, got "
                f"{self.loader.start_indices.n_initial_conditions} and "
                f"{dist.world_size}."
            )


@dataclasses.dataclass
class EMAConfig:
    """
    Configuration for exponential moving average of model weights.

    Attributes:
        decay: decay rate for the moving average
    """

    decay: float = 0.9999

    def build(self, model):
        return EMATracker(model, decay=self.decay, faster_decay_at_start=True)


@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for training a model.

    Attributes:
        train_loader: configuration for the training data loader
        validation_loader: configuration for the validation data loader
        stepper: configuration for the stepper
        optimization: configuration for the optimization
        logging: configuration for logging
        max_epochs: total number of epochs to train for
        save_checkpoint: whether to save checkpoints
        experiment_dir: directory where checkpoints and logs are saved
        inference: configuration for inline inference
        n_forward_steps: number of forward steps to take gradient over
        copy_weights_after_batch: Configuration for copying weights from the
            base model to the training model after each batch. This is used
            to achieve an effect of freezing model parameters that can freeze
            a subset of each weight that comes from a smaller base weight.
            This is less efficient than true parameter freezing, but layer
            freezing is all-or-nothing for each parameter. By default, no
            weights are copied.
        ema: configuration for exponential moving average of model weights
        validate_using_ema: whether to validate using the EMA model
        checkpoint_save_epochs: how often to save epoch-based checkpoints,
            if save_checkpoint is True. If None, checkpoints are only saved
            for the most recent epoch
            (and the best epochs if validate_using_ema == False).
        ema_checkpoint_save_epochs: how often to save epoch-based EMA checkpoints,
            if save_checkpoint is True. If None, EMA checkpoints are only saved
            for the most recent epoch
            (and the best epochs if validate_using_ema == True).
        log_train_every_n_batches: how often to log batch_loss during training
        segment_epochs: (optional) exit after training for at most this many epochs
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
    monthly_reference_data: Optional[str] = None

    def __post_init__(self):
        if self.monthly_reference_data is not None:
            warnings.warn(
                "monthly_reference_data is deprecated, use "
                "inference.aggregator.monthly_reference_data instead.",
                category=DeprecationWarning,
            )
            self.inference.aggregator.monthly_reference_data = (
                self.monthly_reference_data
            )

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

    def configure_wandb(self, env_vars: Optional[Mapping[str, str]] = None, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        if "environment" in config:
            logging.warning("Not recording env vars since 'environment' is in config.")
        elif env_vars is not None:
            config["environment"] = env_vars
        self.logging.configure_wandb(config=config, **kwargs)

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
