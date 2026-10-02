import logging
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import wandb

from fme.core.disk_metric_logger import DiskMetricLogger, read_metrics_by_step
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
        self._disk_logger: DiskMetricLogger | None = None

    def configure(self, log_to_wandb: bool, metrics_log_dir: str | None = None):
        dist = Distributed.get_instance()
        self._enabled = log_to_wandb and dist.is_root()
        self._configured = True
        if self._disk_logger is not None:
            self._disk_logger.close()
            self._disk_logger = None
        if metrics_log_dir is not None and dist.is_root():
            self._disk_logger = DiskMetricLogger(metrics_log_dir)

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

    def finish(self):
        """End the active run so the next `init` starts a new run rather than
        reusing it (wandb returns the active run by default in scripts).
        """
        if self._enabled:
            wandb.finish()
        self._id = None

    def watch(self, modules):
        if self._enabled:
            wandb.watch(modules)

    def log(
        self,
        data: Mapping[str, Any],
        step: int,
        sleep: float | None = None,
        commit: bool | None = None,
    ):
        if self._enabled:
            wandb.log(dict(data), step=step, commit=commit)
            if sleep is not None:
                time.sleep(sleep)
        if self._disk_logger is not None:
            self._disk_logger.log(dict(data), step=step)
        dist = Distributed.get_instance()
        dist.barrier()

    @property
    def disk_metrics_offset(self) -> int | None:
        """Size in bytes of the disk metrics logged so far, which a checkpoint
        records so a job resuming from it can ``restore_disk_metrics``. None if
        disk metric logging is disabled.
        """
        if self._disk_logger is None:
            return None
        return self._disk_logger.offset

    def restore_disk_metrics(
        self, offset: int | None, resume_step: int, step_continues: bool
    ):
        """Restore the disk metrics a previous job logged before a checkpoint,
        and re-log to wandb the ones it never received.

        wandb uploads logs in the background, so a job killed (e.g. preempted)
        shortly after logging loses whatever was still queued. Only scalars are
        on disk, so figures in lost logs stay lost.

        Args:
            offset: The checkpoint's ``disk_metrics_offset``.
            resume_step: The checkpoint's step.
            step_continues: Whether this job logs more metrics at
                ``resume_step``, in which case the metrics recovered at that
                step are not committed to wandb, so the new ones join them.
        """
        if self._disk_logger is None:
            return
        if offset is None:
            logging.warning(
                "The checkpoint was saved with disk metric logging disabled, so "
                "no disk metrics are restored"
            )
            return
        if self._disk_logger.restore(offset):
            self._relog_unsynced_disk_metrics(resume_step, step_continues)

    def restore_disk_metrics_through_step(self, last_step: int):
        """Like ``restore_disk_metrics``, for a checkpoint saved before
        checkpoints recorded ``disk_metrics_offset``.

        Restores the previous job's disk metrics through ``last_step``, which
        must be a step this job does not log metrics at again.
        """
        if self._disk_logger is None:
            return
        logging.warning(
            "The checkpoint does not record disk_metrics_offset, so disk metrics "
            f"are restored through step {last_step}"
        )
        if self._disk_logger.restore_through_step(last_step):
            self._relog_unsynced_disk_metrics(last_step, step_continues=False)

    def _relog_unsynced_disk_metrics(self, resume_step: int, step_continues: bool):
        """Re-log the restored disk metrics wandb never received.

        A resumed wandb run continues from the step after the last one it
        received and rejects logs at earlier steps, so disk metrics from that
        step on are re-logged at their original steps.
        """
        if not self._enabled or self._disk_logger is None or wandb.run is None:
            return
        if wandb.run.offline:
            logging.info(
                "wandb is offline, so disk metrics are not re-logged: the "
                "previous job's offline wandb files hold them"
            )
            return
        first_step = wandb.run.step
        logging.info(
            f"Checking disk metrics for steps wandb lacks: wandb resumed at "
            f"step {first_step}, checkpoint step {resume_step}"
        )
        if step_continues and first_step > resume_step:
            logging.warning(
                f"wandb already received step {resume_step}, so it will reject "
                "the metrics this job logs at that step"
            )
        metrics_by_step = read_metrics_by_step(self._disk_logger.directory, first_step)
        for step, data in metrics_by_step.items():
            commit = not (step_continues and step == resume_step)
            wandb.log(data, step=step, commit=commit)
        if metrics_by_step:
            steps = list(metrics_by_step)
            epochs = [
                data["epoch"] for data in metrics_by_step.values() if "epoch" in data
            ]
            logging.info(
                f"Recovered wandb logs for {len(steps)} steps from disk "
                f"(steps {steps[0]} to {steps[-1]}, epochs {epochs})"
            )

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
