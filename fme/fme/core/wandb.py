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

    def init(
        self,
        config: Mapping[str, Any],
        project: str,
        entity: str,
        resume: bool,
        dir: str,
    ):
        if not self._configured:
            raise RuntimeError(
                "must call WandB.configure before WandB init can be called"
            )
        if self._enabled:
            wandb.init(
                config=config,
                project=project,
                entity=entity,
                resume=resume,
                dir=dir,
            )

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
