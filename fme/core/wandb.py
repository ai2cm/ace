import logging
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import wandb

from fme.core.distributed import Distributed

WANDB_RUN_ID_FILE = "wandb_run_id"


class DirectInitializationError(RuntimeError):
    pass


class Histogram(wandb.Histogram):
    def __init__(
        self,
        *args,
        direct_access=True,
        **kwargs,
    ):
        if direct_access:
            raise DirectInitializationError(
                "must initialize from `wandb = WandB.get_instance()`, "
                "not directly from `import fme.core.wandb`"
            )
        super().__init__(*args, **kwargs)


Histogram.__doc__ = wandb.Histogram.__doc__
Histogram.__init__.__doc__ = wandb.Histogram.__init__.__doc__


class Table(wandb.Table):
    def __init__(
        self,
        *args,
        direct_access=True,
        **kwargs,
    ):
        if direct_access:
            raise DirectInitializationError(
                "must initialize from `wandb = WandB.get_instance()`, "
                "not directly from `import fme.core.wandb`"
            )
        super().__init__(*args, **kwargs)


Table.__doc__ = wandb.Table.__doc__
Table.__init__.__doc__ = wandb.Table.__init__.__doc__


class Video(wandb.Video):
    def __init__(
        self,
        *args,
        direct_access=True,
        **kwargs,
    ):
        if direct_access:
            raise DirectInitializationError(
                "must initialize from `wandb = WandB.get_instance()`, "
                "not directly from `import fme.core.wandb`"
            )
        super().__init__(*args, **kwargs)


Video.__doc__ = wandb.Video.__doc__
Video.__init__.__doc__ = wandb.Video.__init__.__doc__


class Image(wandb.Image):
    def __init__(
        self,
        *args,
        direct_access=True,
        **kwargs,
    ):
        if direct_access:
            raise DirectInitializationError(
                "must initialize from `wandb = WandB.get_instance()`, "
                "not directly from `import fme.core.wandb`"
            )
        super().__init__(*args, **kwargs)


Image.__doc__ = wandb.Image.__doc__
Image.__init__.__doc__ = wandb.Image.__init__.__doc__


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
        self._id = None

    def configure(self, log_to_wandb: bool):
        dist = Distributed.get_instance()
        self._enabled = log_to_wandb and dist.is_root()
        self._configured = True

    def init(
        self,
        resumable: bool = False,
        experiment_dir: str | None = None,
        **kwargs,
    ):
        """
        Initialize wandb, potentially with resumption logic.

        Args:
            resumable: If True, attempt to resume the run in the experiment directory,
                or start a new run there if no run is found.
            experiment_dir: The directory where the experiment is being run. Required if
                `resumable` is True.
            **kwargs: Passed to wandb.init.
        """
        if not self._configured:
            raise RuntimeError(
                "must call WandB.configure before WandB init can be called"
            )
        if self._enabled:
            if resumable:
                if experiment_dir is None:
                    raise ValueError(
                        "must provide `experiment_dir` when `resumable` is True"
                    )
                else:
                    id_ = init_wandb_with_resumption(
                        experiment_dir, direct_access=False, **kwargs
                    )
            else:
                wandb.init(**kwargs)
                if wandb.run is None:
                    raise RuntimeError("wandb.init did not return a run")
                else:
                    id_ = wandb.run.id
                logging.info(f"New non-resuming wandb run with id: {id_}.")
            self._id = id_

    def watch(self, modules):
        if self._enabled:
            wandb.watch(modules)

    def log(self, data: Mapping[str, Any], step=None, sleep=None):
        if self._enabled:
            wandb.log(dict(data), step=step)
            if sleep is not None:
                time.sleep(sleep)

    def Image(self, data_or_path, *args, **kwargs) -> Image:
        if isinstance(data_or_path, np.ndarray):
            data_or_path = scale_image(data_or_path)

        return Image(data_or_path, *args, direct_access=False, **kwargs)

    def Video(self, *args, **kwargs) -> Video:
        return Video(*args, direct_access=False, **kwargs)

    def Table(self, *args, **kwargs) -> Table:
        return Table(*args, direct_access=False, **kwargs)

    def Histogram(self, *args, **kwargs) -> Histogram:
        return Histogram(*args, direct_access=False, **kwargs)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def configured(self) -> bool:
        return self._configured

    def get_id(self) -> str | None:
        return self._id


singleton: WandB | None = None


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


def init_wandb_with_resumption(
    experiment_dir: str,
    direct_access=True,
    wandb_run_id_file: str = WANDB_RUN_ID_FILE,
    wandb_init=None,
    wandb_id=None,
    **kwargs: Any,
) -> str:
    """
    Initialize wandb with resumption logic. If wandb has previously
    been initialized in the experiment directory, resume the run. Otherwise,
    start a new run.

    The reason we implement our own resumption logic is that wandb uses the same
    location to save temporary media files and the information necessary for
    resumption. We want to save these things in different places.

    Args:
        experiment_dir: The directory where the experiment is being run.
        direct_access: If True, raise an error if this function is called directly.
        wandb_run_id_file: The file where the wandb run id is stored.
        wandb_init: The wandb.init function to use (for testing).
        wandb_id: A function returning the wandb run_id (for testing).
        **kwargs: Arguments to pass to `wandb.init`.

    Returns:
        The wandb run id.
    """
    if direct_access:
        raise DirectInitializationError(
            "Must access this function by calling `wandb.init` after "
            "`wandb = WandB.get_instance()`. It should not be called from anywhere "
            "else."
        )

    if wandb_init is None:
        wandb_init = wandb.init

    if wandb_id is None:

        def wandb_id():
            if wandb.run is None:
                raise RuntimeError("wandb does not have an active run")
            return wandb.run.id

    if not os.path.exists(os.path.join(experiment_dir, wandb_run_id_file)):
        # new run
        kwargs.update({"resume": "never"})
        wandb_init(**kwargs)
        logging.info(f"New resumable wandb run with id: {wandb_id()}.")
        with open(os.path.join(experiment_dir, wandb_run_id_file), "w") as f:
            f.write(wandb_id())
    else:
        # resuming
        with open(os.path.join(experiment_dir, wandb_run_id_file)) as f:
            wandb_run_id = f.read().strip()
        kwargs.update({"resume": "must", "id": wandb_run_id})
        wandb_init(**kwargs)
        logging.info(f"Resuming wandb run with id: {wandb_id()}")
    return wandb_id()
