import json
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, TextIO

import numpy as np
import torch
import wandb

from fme.core.disk_metric_logger import DiskMetricLogger
from fme.core.distributed import Distributed

WANDB_RUN_ID_FILE = "wandb_run_id"
RECENT_ROWS_FILE = "wandb_recent_rows.jsonl"
# wandb uploads rows in order within seconds of logging them, so the rows a
# killed job never sent are among the last few it logged
N_RECENT_ROWS = 16
# appended lines before the cache file is rewritten as its merged rows
COMPACT_EVERY_N_LINES = 4 * N_RECENT_ROWS


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
        self._recent_rows: RecentRows | None = None
        self._previous_rows: dict[int, dict[str, Any]] | None = None

    def configure(self, log_to_wandb: bool, metrics_log_dir: str | None = None):
        dist = Distributed.get_instance()
        self._enabled = log_to_wandb and dist.is_root()
        self._configured = True
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
                    path = os.path.join(experiment_dir, RECENT_ROWS_FILE)
                    self._previous_rows = read_recent_rows(path)
                    if self._recent_rows is not None:
                        self._recent_rows.close()
                    self._recent_rows = RecentRows(path, self._previous_rows)
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
        if self._recent_rows is not None:
            self._recent_rows.close()
        self._id = None
        self._recent_rows = None
        self._previous_rows = None

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
        if self._recent_rows is not None:
            if self._previous_rows is not None:
                # rows from a previous job can only be restored before logging
                self._previous_rows = None
                self._recent_rows.drop_after(step - 1)
            self._recent_rows.record(data, step)
        if self._disk_logger is not None:
            self._disk_logger.log(dict(data), step=step)
        dist = Distributed.get_instance()
        dist.barrier()

    def restore_unsent(self, resume_step: int):
        """Re-log the rows a previous job of this resumed run logged but wandb
        never received, e.g. because the job was killed before wandb's
        background uploader sent them.

        Must be called on all ranks after restoring the training checkpoint
        and before logging anything else. Rows from ``resume_step`` on are
        logged again by the resumed job: the row at ``resume_step`` is
        re-logged uncommitted, so the resumed job's logs at that step merge
        into it, and later rows are dropped. Only scalars are restored.

        Args:
            resume_step: The step of the restored checkpoint.
        """
        previous_rows = self._previous_rows
        self._previous_rows = None
        if self._recent_rows is not None:
            # the resumed job logs these steps again
            self._recent_rows.drop_after(resume_step)
        if (
            self._enabled
            and previous_rows
            and wandb.run is not None
            and not wandb.run.offline
        ):
            # the step after the last one wandb received, read before logging
            restored = rows_to_restore(previous_rows, wandb.run.step, resume_step)
            for step, row, commit in restored:
                wandb.log(row, step=step, commit=commit)
            if restored:
                logging.info(
                    f"Re-logged {len(restored)} rows wandb did not receive "
                    f"(steps {restored[0][0]} to {restored[-1][0]})"
                )
        Distributed.get_instance().barrier()

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


def _scalars(data: Mapping[str, Any]) -> dict[str, int | float | bool]:
    """Return the scalar values of ``data``, dropping figures, tables, etc."""
    # log values are heterogeneous by design, so filter by type
    scalars: dict[str, int | float | bool] = {}
    for key, value in data.items():
        if isinstance(value, bool | int | float):
            scalars[key] = value
        elif isinstance(value, np.generic | np.ndarray | torch.Tensor) and (
            value.ndim == 0
        ):
            item = value.item()
            if isinstance(item, bool | int | float):
                scalars[key] = item
    return scalars


class RecentRows:
    """The scalars of the last ``N_RECENT_ROWS`` steps logged, appended to a
    file on every log so they survive the job being killed.

    The file holds one ``{"step": ..., "logs": ...}`` line per log call,
    merged by step when read. It is rewritten as the merged rows every
    ``COMPACT_EVERY_N_LINES`` lines to bound its size.
    """

    def __init__(self, path: str, rows: Mapping[int, dict[str, Any]]):
        self._path = path
        self._rows = {step: dict(row) for step, row in rows.items()}
        self._file: TextIO | None = None
        self._n_appended = 0
        self._compact()

    def record(self, data: Mapping[str, Any], step: int):
        scalars = _scalars(data)
        if not scalars:
            return
        self._rows.setdefault(step, {}).update(scalars)
        _trim(self._rows)
        assert self._file is not None
        self._file.write(json.dumps({"step": step, "logs": scalars}) + "\n")
        # the SIGTERM path ends in os._exit, which skips flushing buffers
        self._file.flush()
        self._n_appended += 1
        if self._n_appended >= COMPACT_EVERY_N_LINES:
            self._compact()

    def drop_after(self, step: int):
        """Forget the rows after ``step``."""
        for later_step in [s for s in self._rows if s > step]:
            del self._rows[later_step]
        self._compact()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def _compact(self):
        self.close()
        temporary_location = f"{self._path}.tmp"
        with open(temporary_location, "w") as f:
            for step in sorted(self._rows):
                f.write(json.dumps({"step": step, "logs": self._rows[step]}) + "\n")
        os.replace(temporary_location, self._path)
        self._file = open(self._path, "a")
        self._n_appended = 0


def _trim(rows: dict[int, dict[str, Any]]):
    for old_step in sorted(rows)[:-N_RECENT_ROWS]:
        del rows[old_step]


def read_recent_rows(path: str) -> dict[int, dict[str, Any]]:
    """Read the rows written by ``RecentRows``, merged by step, or none if
    there is no file.
    """
    rows: dict[int, dict[str, Any]] = {}
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            step, logs = int(entry["step"]), dict(entry["logs"])
        except (ValueError, KeyError, TypeError) as err:
            if i == len(lines) - 1:
                # a job killed mid-write leaves a torn last line
                logging.debug(f"Skipping the torn last line of {path}: {err}")
            else:
                logging.warning(f"Skipping invalid line {i + 1} of {path}: {err}")
            continue
        rows.setdefault(step, {}).update(logs)
    _trim(rows)
    return rows


def rows_to_restore(
    rows: Mapping[int, dict[str, Any]], next_step: int, resume_step: int
) -> list[tuple[int, dict[str, Any], bool]]:
    """The (step, row, commit) to re-log of the previous job's ``rows``, for
    a wandb run that has received the steps before ``next_step`` and a job
    resuming at ``resume_step``.
    """
    restored = []
    for step in sorted(rows):
        if next_step <= step < resume_step:
            restored.append((step, rows[step], True))
        elif next_step <= step == resume_step:
            restored.append((step, rows[step], False))
    return restored


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
