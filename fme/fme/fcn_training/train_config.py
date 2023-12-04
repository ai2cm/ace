import dataclasses
import logging
import os
from typing import Any, Mapping

from fme.core.distributed import Distributed
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

    def configure_wandb(self, config: Mapping[str, Any], resume: bool):
        # must ensure wandb.configure is called before wandb.init
        wandb = WandB.get_instance()
        wandb.configure(log_to_wandb=self.log_to_wandb)
        wandb.init(
            config=config,
            project=self.project,
            entity=self.entity,
            resume=resume,
            dir=config["experiment_dir"],
        )
