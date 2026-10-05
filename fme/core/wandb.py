import dataclasses
import logging
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import wandb

from fme.core.cloud import is_local
from fme.core.disk_metric_logger import (
    CheckpointMark,
    DiskMetricLogger,
    read_metrics_by_step,
)
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
        # only the root rank logs metrics to disk, see build_disk_logger
        self._disk_logger: DiskMetricLogger | None = None

    def configure(self, log_to_wandb: bool, metrics_log_dir: str):
        """Set up logging to wandb and to disk. Must be called before ``init``.

        Args:
            log_to_wandb: Whether to log to Weights & Biases.
            metrics_log_dir: Directory to write scalar metrics to on disk, so
                a resumed job can recover any that wandb lost.
        """
        dist = Distributed.get_instance()
        self._enabled = log_to_wandb and dist.is_root()
        self._configured = True
        if self._disk_logger is not None:
            self._disk_logger.close()
        self._disk_logger = build_disk_logger(metrics_log_dir)

    def init(
        self,
        resumable: bool = False,
        experiment_dir: str | None = None,
        **kwargs,
    ):
        """
        Initialize wandb, potentially with resumption logic.

        If `resumable`, restores the metrics logged to disk before the last
        checkpoint, and re-logs any that the resumed wandb run did not
        receive. The disk restore happens whether or not logging to wandb is
        enabled. Must be called on all ranks.

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
        if resumable:
            self._restore_disk_metrics()
            Distributed.get_instance().barrier()

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

    def mark_checkpoint(self):
        """Record that a checkpoint has been saved.

        Call right after saving a checkpoint that training can resume from.
        wandb uploads logs in the background, so a job killed shortly after
        logging loses the logs not yet uploaded. A job resuming from this
        checkpoint restores the metrics logged to disk up to this point and
        re-logs to wandb the ones it did not receive. Only scalars are logged
        to disk, so figures are not recovered.

        Safe to call from the termination listener's thread, since it only
        writes a local file (see
        `fme.core.distributed.shutdown.add_post_abort_callback`).
        """
        if self._disk_logger is not None:
            self._disk_logger.write_checkpoint_mark()

    def _restore_disk_metrics(self):
        """Restore the metrics logged to disk before the last checkpoint, and
        if this job resumed the previous wandb run, re-log to it any metrics
        it did not receive. Nothing is re-logged to a new wandb run (e.g. with
        ``resume_wandb: false``), since it has not lost any logs.
        """
        if self._disk_logger is None:
            return
        mark = self._disk_logger.restore_to_checkpoint_mark()
        if mark is None:
            return
        if wandb.run is None:  # this rank does not log to wandb
            return
        if wandb.run.offline:
            logging.info(
                "wandb is offline, so disk metrics are not re-logged: the "
                "previous job's offline wandb files hold them"
            )
            return
        # wandb.run.resumed, not the run id file, tells a resumed run from a
        # new one: init_wandb_with_resumption writes the file for a new run
        # too, so by this point it always exists.
        if not wandb.run.resumed:
            return
        first_step = wandb.run.step
        logging.info(
            f"Checking disk metrics for steps wandb lacks: wandb resumed at "
            f"step {first_step}, checkpoint mark at step {mark.last_step}"
        )
        calls = metrics_to_relog(self._disk_logger.directory, first_step, mark)
        for call in calls:
            wandb.log(call.data, step=call.step, commit=call.commit)
        if calls:
            epochs = [call.data["epoch"] for call in calls if "epoch" in call.data]
            logging.info(
                f"Recovered wandb logs for {len(calls)} steps from disk "
                f"(steps {calls[0].step} to {calls[-1].step}, epochs {epochs})"
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


def build_disk_logger(metrics_log_dir: str) -> DiskMetricLogger | None:
    """Build a DiskMetricLogger for this rank.

    Returns None on non-root ranks, which do not log metrics, and when
    ``metrics_log_dir`` is not on a local file system.
    """
    if not Distributed.get_instance().is_root():
        return None
    if not is_local(metrics_log_dir):
        logging.warning(
            f"Disk metric logging is only supported on a local file system. "
            f"Got metrics_log_dir={metrics_log_dir!r}, so no metrics will be "
            f"saved to disk and none can be recovered on resume."
        )
        return None
    return DiskMetricLogger(metrics_log_dir)


@dataclasses.dataclass(frozen=True)
class WandBLogCall:
    """The arguments of one call to ``wandb.log``."""

    data: dict[str, Any]
    step: int
    commit: bool


def metrics_to_relog(
    directory: str | os.PathLike, first_step: int, mark: CheckpointMark
) -> list[WandBLogCall]:
    """Determine the calls to ``wandb.log`` that give a resumed run the
    metrics it did not receive from the previous job.

    Args:
        directory: Directory holding the restored metrics file.
        first_step: The step the resumed run continues at, i.e. the step after
            the last one it received. wandb rejects logs at earlier steps.
        mark: The checkpoint mark the metrics were restored to.

    Returns:
        One call per step from ``first_step`` on, in step order, with the
        metrics logged at that step merged. The call at the mark's last step
        has ``commit=False``, so that if the resumed job logs at that step
        again (e.g. the end-of-epoch metrics after a checkpoint saved before
        validation) those logs are added to the step instead of rejected.
    """
    return [
        WandBLogCall(data=data, step=step, commit=step != mark.last_step)
        for step, data in read_metrics_by_step(directory, first_step).items()
    ]


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
