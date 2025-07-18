import contextlib
import dataclasses
import logging
import os
from collections.abc import Mapping
from typing import Any

from fme.core.distributed import Distributed
from fme.core.wandb import WandB

ENV_VAR_NAMES = (
    "BEAKER_EXPERIMENT_ID",
    "SLURM_JOB_ID",
    "SLURM_JOB_USER",
    "FME_TRAIN_DIR",
    "FME_VALID_DIR",
    "FME_STATS_DIR",
    "FME_CHECKPOINT_DIR",
    "FME_OUTPUT_DIR",
    "FME_IMAGE",
)

DEFAULT_TMP_DIR = "/tmp"


@dataclasses.dataclass
class LoggingConfig:
    """
    Configuration for logging.

    Parameters:
        project: Name of the project in Weights & Biases.
        entity: Name of the entity in Weights & Biases.
        log_to_screen: Whether to log to the screen.
        log_to_file: Whether to log to a file.
        log_to_wandb: Whether to log to Weights & Biases.
        log_format: Format of the log messages.
        level: Sets the logging level.
    """

    project: str = "ace"
    entity: str = "ai2cm"
    log_to_screen: bool = True
    log_to_file: bool = True
    log_to_wandb: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level: str | int = logging.INFO

    def __post_init__(self):
        self._dist = Distributed.get_instance()

    def configure_logging(self, experiment_dir: str, log_filename: str):
        """
        Configure the global `logging` module based on this LoggingConfig.
        """
        if self.log_to_screen and self._dist.is_root():
            logging.basicConfig(format=self.log_format, level=self.level)
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
            fh.setLevel(self.level)
            fh.setFormatter(logging.Formatter(self.log_format))
            logger.addHandler(fh)

    def configure_wandb(
        self,
        config: Mapping[str, Any],
        env_vars: Mapping[str, Any] | None = None,
        wandb_dir: str | None = DEFAULT_TMP_DIR,
        resumable: bool = True,
        resume: Any = None,
        **kwargs,
    ):
        if resume is not None:
            raise ValueError(
                "The 'resume' argument is no longer supported, "
                "please pass 'resumable' instead."
            )
        config_copy = {**config}
        if "environment" in config_copy:
            logging.warning(
                "Not recording environmental variables since 'environment' key is "
                "already present in config."
            )
        elif env_vars is not None:
            config_copy["environment"] = env_vars
        # must ensure wandb.configure is called before wandb.init
        wandb = WandB.get_instance()
        wandb.configure(log_to_wandb=self.log_to_wandb)
        wandb.init(
            config=config_copy,
            project=self.project,
            entity=self.entity,
            experiment_dir=config["experiment_dir"],
            resumable=resumable,
            dir=wandb_dir,
            **kwargs,
        )


def log_versions():
    import torch

    logging.info("--------------- Versions ---------------")
    logging.info("Torch: " + str(torch.__version__))
    logging.info("----------------------------------------")


def retrieve_env_vars(names=ENV_VAR_NAMES) -> dict[str, str]:
    """Return a dictionary of specific environmental variables."""
    output = {}
    for name in names:
        try:
            value = os.environ[name]
        except KeyError:
            logging.warning(f"Environmental variable {name} not found.")
        else:
            output[name] = value
            logging.info(f"Environmental variable {name}={value}.")
    return output


def log_beaker_url(beaker_id=None):
    """Log the Beaker ID and URL for the current experiment.

    beaker_id: The Beaker ID of the experiment. If None, uses the env variable
    `BEAKER_EXPERIMENT_ID`.

    Returns the Beaker URL.
    """
    if beaker_id is None:
        try:
            beaker_id = os.environ["BEAKER_EXPERIMENT_ID"]
        except KeyError:
            logging.warning("Beaker Experiment ID not found.")
            return None

    beaker_url = f"https://beaker.org/ex/{beaker_id}"
    logging.info(f"Beaker ID: {beaker_id}")
    logging.info(f"Beaker URL: {beaker_url}")
    return beaker_url


@contextlib.contextmanager
def log_level(level):
    """Temporarily set the log level of the global logger."""
    logger = logging.getLogger()  # presently, data loading uses the root logger
    old_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)
