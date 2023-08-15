import logging
import os


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
