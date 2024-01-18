import dataclasses
import logging
import os
import warnings
from typing import Any, Mapping, Optional, Union

from fme.core import SingleModuleStepperConfig
from fme.core.data_loading.params import DataLoaderParams, Slice
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.ema import EMATracker
from fme.core.optimization import OptimizationConfig
from fme.core.stepper import ExistingStepperConfig
from fme.core.wandb import WandB


@dataclasses.dataclass
class LoggingConfig:
    project: str = "fcn_training"
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


@dataclasses.dataclass
class InlineInferenceConfig:
    """
    Attributes:
        data: configuration for the data loader used during inference
        n_forward_steps: number of forward steps to take
        forward_steps_in_memory: number of forward steps to take before
            re-reading data from disk
        epochs: epochs on which to run inference, where the first epoch is
            defined as epoch 0 (unlike in logs which show epochs as starting
            from 1). By default runs inference every epoch.
    """

    data: DataLoaderParams
    n_forward_steps: int = 2
    forward_steps_in_memory: int = 2
    epochs: Slice = Slice(start=0, stop=None, step=1)

    def __post_init__(self):
        if self.n_forward_steps % self.forward_steps_in_memory != 0:
            raise ValueError(
                "n_forward_steps must be divisible by steps_in_memory, "
                f"got {self.n_forward_steps} and {self.forward_steps_in_memory}"
            )
        dist = Distributed.get_instance()
        if self.data.batch_size % dist.world_size != 0:
            raise ValueError(
                "batch_size must be divisible by the number of parallel "
                f"workers, got {self.batch_size} and {dist.world_size}"
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
        train_data: configuration for the training data loader
        validation_data: configuration for the validation data loader
        stepper: configuration for the stepper
        optimization: configuration for the optimization
        logging: configuration for logging
        max_epochs: total number of epochs to train for
        save_checkpoint: whether to save checkpoints
        experiment_dir: directory where checkpoints and logs are saved
        inference: configuration for inline inference
        checkpoint_save_epochs: how often to save epoch-based checkpoints,
            if save_checkpoint is True. If None, checkpoints are only saved
            for the most recent epoch and the best epoch.
        log_train_every_n_batches: how often to log batch_loss during training
        segment_epochs: (optional) exit after training for at most this many epochs
            in current job, without exceeding `max_epochs`. Use this if training
            must be run in segments, e.g. due to wall clock limit.
    """

    train_data: DataLoaderParams
    validation_data: DataLoaderParams
    stepper: Union[SingleModuleStepperConfig, ExistingStepperConfig]
    optimization: OptimizationConfig
    logging: LoggingConfig
    max_epochs: int
    save_checkpoint: bool
    experiment_dir: str
    inference: InlineInferenceConfig
    n_forward_steps: int
    ema: EMAConfig = dataclasses.field(default_factory=lambda: EMAConfig())
    validate_using_ema: bool = False
    checkpoint_save_epochs: Optional[Slice] = None
    log_train_every_n_batches: int = 100
    segment_epochs: Optional[int] = None
    checkpoint_every_n_epochs: Optional[int] = None
    parallel: Optional[bool] = None

    def __post_init__(self):
        if self.checkpoint_every_n_epochs is not None:
            warnings.warn(
                "checkpoint_every_n_epochs is deprecated, use"
                "checkpoint_save_epochs instead.",
                category=DeprecationWarning,
            )
            self.checkpoint_save_epochs = Slice(
                start=0,
                stop=self.max_epochs,
                step=self.checkpoint_every_n_epochs,
            )
        if self.parallel is not None:
            if self.parallel:
                warnings.warn(
                    (
                        "The 'parallel' argument is deprecated and will be ignored. "
                        "Inline inference is now always performed in parallel. "
                        "There's no need to specify this argument in future uses "
                        "of this function."
                    ),
                    category=DeprecationWarning,
                )
            elif not self.parallel:
                raise ValueError("parallel=False is no longer supported")

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
    def ema_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "ema_ckpt.tar")

    def epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ckpt_{epoch:04d}.tar")

    def epoch_checkpoint_enabled(self, epoch: int) -> bool:
        return epoch_checkpoint_enabled(
            epoch, self.max_epochs, self.checkpoint_save_epochs
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


def epoch_checkpoint_enabled(
    epoch: int, max_epochs: int, save_epochs: Optional[Slice]
) -> bool:
    if save_epochs is None:
        return False
    return epoch in range(max_epochs)[save_epochs.slice]
