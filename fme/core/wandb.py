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
        # None on non-root ranks and for a non-local metrics_log_dir (see
        # build_disk_logger); the None-guards below are for those cases only
        self._disk_logger: DiskMetricLogger | None = None

    def configure(self, log_to_wandb: bool, metrics_log_dir: str):
        """
        Args:
            log_to_wandb: Whether to log to Weights & Biases.
            metrics_log_dir: Directory to write scalar metrics to disk, which
                resumable runs restore from on resume.
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

        A resumable init also restores the disk metrics a previous job logged up
        to its last ``mark_checkpoint``, and, if it resumes a wandb run rather
        than starting a new one, re-logs to that run the ones it never received.
        Must be called on all ranks.

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
        resumed_run = False
        if self._enabled:
            if resumable:
                if experiment_dir is None:
                    raise ValueError(
                        "must provide `experiment_dir` when `resumable` is True"
                    )
                else:
                    resumed_run = os.path.exists(
                        os.path.join(experiment_dir, WANDB_RUN_ID_FILE)
                    )
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
            self._restore_disk_metrics(relog=resumed_run)
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
        """Record that a resume checkpoint holds the training logged so far.

        Call right after the checkpoint is saved. When a later job resumes the
        wandb run, ``init`` restores the disk metrics logged up to this mark
        and re-logs the ones wandb never received: wandb uploads logs in the
        background, so a job killed (e.g. preempted) shortly after logging
        loses whatever was still queued, and wandb holds the last logged step
        uncommitted until a later step is logged. Only scalars are on disk, so
        figures in lost logs stay lost.

        A no-op unless disk metric logging is enabled, which it is only on the
        root rank. Safe to call on the termination listener's thread: it only
        writes a local file (see
        `fme.core.distributed.shutdown.add_post_abort_callback`).
        """
        if self._disk_logger is not None:
            self._disk_logger.write_checkpoint_mark()

    def _restore_disk_metrics(self, relog: bool):
        """Restore the disk metrics a previous job logged up to its last
        checkpoint mark, and if ``relog``, re-log to the resumed wandb run the
        ones it never received. A new wandb run (e.g. ``resume_wandb: false``)
        did not lose them, so it is not given them.
        """
        if self._disk_logger is None:
            return
        mark = self._disk_logger.restore_to_checkpoint_mark()
        if mark is None or not relog or wandb.run is None:
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
            f"step {first_step}, checkpoint mark at step {mark.last_step}"
        )
        rows = metrics_to_relog(self._disk_logger.directory, first_step, mark)
        for step, data, commit in rows:
            wandb.log(data, step=step, commit=commit)
        if rows:
            epochs = [data["epoch"] for _, data, _ in rows if "epoch" in data]
            logging.info(
                f"Recovered wandb logs for {len(rows)} steps from disk "
                f"(steps {rows[0][0]} to {rows[-1][0]}, epochs {epochs})"
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
    """Build the disk metric logger for this rank, or None if it has none: only
    the root rank logs metrics to disk, and only to a local file system.
    """
    if not Distributed.get_instance().is_root():
        return None
    if not is_local(metrics_log_dir):
        # reachable only through a non-local experiment directory, since
        # LoggingConfig rejects a non-local metrics_log_dir itself
        logging.warning(
            f"Disk metric logging is only supported on a local file system. "
            f"Got metrics_log_dir={metrics_log_dir!r}, so no metrics will be "
            f"saved to disk and none can be recovered on resume."
        )
        return None
    return DiskMetricLogger(metrics_log_dir)


def metrics_to_relog(
    directory: str | os.PathLike, first_step: int, mark: CheckpointMark
) -> list[tuple[int, dict[str, Any], bool]]:
    """The restored disk metrics to re-log to a resumed wandb run, as
    ``(step, data, commit)``.

    A resumed wandb run continues at ``first_step``, the step after the last one
    it received, and rejects logs at earlier steps, so the metrics from that
    step on are re-logged at their original steps, merged by step. The mark's
    last step is left uncommitted: the resumed job may log at it again, e.g.
    the end-of-epoch logs after a checkpoint saved before validation, and those
    logs then update it rather than being rejected.
    """
    return [
        (step, data, step != mark.last_step)
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
