import logging
import os
import subprocess

from fme.core.distributed import Distributed

KEYFILE_VAR = "GOOGLE_APPLICATION_CREDENTIALS"


def authenticate(keyfile_var: str = KEYFILE_VAR):
    dist = Distributed.get_instance()
    if dist.is_root():
        try:
            keyfile = os.environ[keyfile_var]
        except KeyError:
            logging.info(
                "No keyfile found in environment, skipping gcloud authentication."
            )
        else:
            output = subprocess.check_output(
                ["gcloud", "auth", "activate-service-account", "--key-file", keyfile]
            )
            logging.info(output.decode("utf-8"))
