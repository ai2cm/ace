import logging
import os

from fme.fcn_training.utils.data_loader_multifiles import (
    DataLoaderParams,
)

import dataclasses
from fme.core import SingleModuleStepperConfig


@dataclasses.dataclass
class TrainConfig:
    train_data: DataLoaderParams
    validation_data: DataLoaderParams
    stepper: SingleModuleStepperConfig
    max_epochs: int
    save_checkpoint: bool
    log_to_screen: bool
    log_to_wandb: bool
    project: str
    entity: str
    experiment_dir: str
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

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info(str(self))
        logging.info("---------------------------------------------------")
