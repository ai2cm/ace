import logging
import os
from typing import Any, Mapping
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed

from fme.fcn_training.utils.data_loader_multifiles import (
    DataLoaderParams,
)

import dataclasses
from fme.core import SingleModuleStepperConfig
from fme.core.wandb import WandB

wandb = WandB.get_instance()


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
                os.makedirs(experiment_dir)
            log_path = os.path.join(experiment_dir, log_filename)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(self.log_format))
            logger.addHandler(fh)

    def configure_wandb(self, config: Mapping[str, Any]):
        # must ensure wandb.configure is called before wandb.init
        wandb.configure(log_to_wandb=self.log_to_wandb)
        wandb.init(
            config=config,
            project=self.project,
            entity=self.entity,
            resume=True,
            dir=config["experiment_dir"],
        )


@dataclasses.dataclass
class TrainConfig:
    train_data: DataLoaderParams
    validation_data: DataLoaderParams
    stepper: SingleModuleStepperConfig
    logging: LoggingConfig
    max_epochs: int
    save_checkpoint: bool
    experiment_dir: str
    log_train_every_n_batches: int = 100
    # parameters only for inference
    prediction_length: int = 2

    def __post_init__(self):
        scheduler_type = self.stepper.optimization.scheduler.type
        scheduler_kwargs = self.stepper.optimization.scheduler.kwargs
        # work-around so we don't need to specify T_max
        # in the yaml file for this scheduler
        if scheduler_type == "CosineAnnealingLR" and "T_max" not in scheduler_kwargs:
            self.stepper.optimization.scheduler.kwargs["T_max"] = self.max_epochs

    @property
    def checkpoint_dir(self) -> str:
        """
        The directory where checkpoints are saved.
        """
        return os.path.join(self.experiment_dir, "training_checkpoints")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "ckpt.tar")

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "best_ckpt.tar")

    @property
    def resuming(self) -> bool:
        checkpoint_file_exists = os.path.isfile(self.checkpoint_path)  # type: ignore
        resuming = True if checkpoint_file_exists else False
        return resuming

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self):
        self.logging.configure_wandb(config=to_flat_dict(dataclasses.asdict(self)))

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info(str(self))
        logging.info("---------------------------------------------------")
