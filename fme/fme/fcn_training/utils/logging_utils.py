import os
import logging

_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def config_logger(log_level=logging.INFO):
    logging.basicConfig(format=_format, level=log_level)


def log_to_file(
    logger_name=None, log_level=logging.INFO, log_filename="tensorflow.log"
):
    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)


def log_versions():
    import torch

    logging.info("--------------- Versions ---------------")
    logging.info("Torch: " + str(torch.__version__))
    logging.info("----------------------------------------")


def log_beaker_url(beaker_id=None):
    """Log the Beaker ID and URL for the current experiment.

    beaker_id: The Beaker ID of the experiment. If None, uses the env variable
    `BEAKER_EXPERIMENT_ID`.

    Returns the Beaker ID.
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
    return beaker_id
