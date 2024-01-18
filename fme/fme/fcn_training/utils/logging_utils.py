import contextlib
import logging
import os
from typing import Dict

ENV_VAR_NAMES = (
    "BEAKER_EXPERIMENT_ID",
    "SLURM_JOB_ID",
    "SLURM_JOB_USER",
    "FME_TRAIN_DIR",
    "FME_VALID_DIR",
    "FME_STATS_DIR",
    "FME_CHECKPOINT_DIR",
    "FME_OUTPUT_DIR",
)


def log_versions():
    import torch

    logging.info("--------------- Versions ---------------")
    logging.info("Torch: " + str(torch.__version__))
    logging.info("----------------------------------------")


def retrieve_env_vars(names=ENV_VAR_NAMES) -> Dict[str, str]:
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
