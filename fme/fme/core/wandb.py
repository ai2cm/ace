import glob
import os
import shutil
from typing import Any, Mapping, Optional

import numpy as np
import wandb

from fme.core.distributed import Distributed

singleton: Optional["WandB"] = None


class WandB:
    """
    A singleton class to interface with Weights and Biases (WandB).
    """

    @classmethod
    def get_instance(cls) -> "WandB":
        """
        Get the singleton instance of the WandB class.
        """
        global singleton
        if singleton is None:
            singleton = cls()
        return singleton

    def __init__(self):
        self._enabled = False
        self._configured = False

    def configure(self, log_to_wandb: bool):
        dist = Distributed.get_instance()
        self._enabled = log_to_wandb and dist.is_root()
        self._configured = True

    def init(self, **kwargs):
        """kwargs are passed to wandb.init"""
        if not self._configured:
            raise RuntimeError(
                "must call WandB.configure before WandB init can be called"
            )
        if self._enabled:
            wandb.init(**kwargs)

    def watch(self, modules):
        if self._enabled:
            wandb.watch(modules)

    def log(self, data: Mapping[str, Any], step=None):
        if self._enabled:
            wandb.log(data, step=step)

    def Image(self, data_or_path, *args, **kwargs):
        if isinstance(data_or_path, np.ndarray):
            data_or_path = scale_image(data_or_path)

        return wandb.Image(data_or_path, *args, **kwargs)

    def clean_wandb_dir(self, experiment_dir: str):
        # this is necessary because wandb does not remove run media directories
        # after a run is synced; see https://github.com/wandb/wandb/issues/3564
        if self._enabled:
            wandb.run.finish()  # necessary to ensure the run directory is synced
            wandb_dir = os.path.join(experiment_dir, "wandb")
            remove_media_dirs(wandb_dir)

    @property
    def Video(self):
        return wandb.Video

    @property
    def Table(self):
        return wandb.Table


def scale_image(
    image_data: np.ndarray,
) -> np.ndarray:
    """
    Given an array of scalar data, rescale the data to the range [0, 255].
    """
    data_min = np.nanmin(image_data)
    data_max = np.nanmax(image_data)
    # video data is brightness values on a 0-255 scale
    image_data = 255 * (image_data - data_min) / (data_max - data_min)
    image_data = np.minimum(image_data, 255)
    image_data = np.maximum(image_data, 0)
    image_data[np.isnan(image_data)] = 0
    return image_data


def remove_media_dirs(wandb_dir: str, media_dir_pattern: str = "run-*-*/files/media"):
    """
    Remove the media directories in the wandb run directories.
    """
    glob_pattern = os.path.join(wandb_dir, media_dir_pattern)
    media_dirs = glob.glob(glob_pattern)
    for media_dir in media_dirs:
        if os.path.isdir(media_dir):
            shutil.rmtree(media_dir)
