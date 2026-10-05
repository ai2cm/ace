import collections
import contextlib
import os
import random
import string
from collections.abc import Mapping
from typing import Any, Literal

from fme.core import wandb
from fme.core.disk_metric_logger import DiskMetricLogger
from fme.core.distributed import Distributed


class MockWandB:
    def __init__(self):
        self._enabled = False
        self._configured = False
        self._logs: dict[int, dict[str, Any]] = collections.defaultdict(dict)
        self._last_step = 0
        self._last_received_step: int | None = None
        self._id: str | None = None
        self._disk_logger: DiskMetricLogger | None = None
        self._runs: list[dict[str, Any]] = []
        # wandb reads WANDB_NAME only on the first init; model that one-time
        # snapshot so an explicit `name` is needed to rename subsequent runs.
        self._env_name_snapshot: str | None = None
        self._env_name_snapshot_taken = False

    def configure(self, log_to_wandb: bool, metrics_log_dir: str):
        dist = Distributed.get_instance()
        self._enabled = log_to_wandb and dist.is_root()
        self._configured = True
        if self._disk_logger is not None:
            self._disk_logger.close()
        self._disk_logger = wandb.build_disk_logger(metrics_log_dir)

    def init(
        self,
        resumable: bool = False,
        experiment_dir: str | None = None,
        **kwargs,
    ):
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
                        os.path.join(experiment_dir, wandb.WANDB_RUN_ID_FILE)
                    )
                    wandb.init_wandb_with_resumption(
                        experiment_dir,
                        direct_access=False,
                        wandb_init=self._wandb_init,
                        wandb_id=self.get_id,
                        **kwargs,
                    )
            else:
                self._wandb_init(resume="never", **kwargs)
        if resumable:
            self._restore_disk_metrics(relog=resumed_run)

    def _wandb_init(
        self,
        resume: Literal["must", "never"],
        id: str | None = None,
        name: str | None = None,
        **kwargs,
    ):
        """
        Mocks the `wandb.init` behavior, specifically around initializing
        a run with `resume` and `id`.
        See https://docs.wandb.ai/guides/runs/resuming/.
        """
        # Snapshot WANDB_NAME on the first init only; an explicit `name` wins.
        if not self._env_name_snapshot_taken:
            self._env_name_snapshot = os.environ.get("WANDB_NAME")
            self._env_name_snapshot_taken = True
        if resume == "must":
            if id is None:
                raise ValueError("resume='must' and id is None")
            else:
                if id != self._id:
                    raise ValueError("resume='must' and id does not match previous id")
        else:
            if id is not None:
                raise ValueError("resume='never' and id is not None")
            else:
                if self._id is not None:
                    raise ValueError(
                        "resume='never' and id is None but previous id exists"
                    )
            self._id = _mock_wandb_id()
            run_name = name if name is not None else self._env_name_snapshot
            self._runs.append({"id": self._id, "name": run_name})

    def get_id(self) -> str:
        if self._id is None:
            raise ValueError("mock wandb id is None")
        return self._id

    def set_id(self, id: str):
        self._id = id

    def set_last_received_step(self, step: int):
        """Simulate resuming a wandb run that received logs through ``step``
        in a previous job, whose logs this mock does not hold.
        """
        self._last_received_step = step

    def finish(self):
        # Reset per-run state so the next init starts fresh; the env-name
        # snapshot persists, mirroring wandb's setup singleton across finish().
        self._id = None
        self._last_step = 0

    @property
    def runs(self) -> list[dict[str, Any]]:
        """The runs started so far, each as ``{"id": ..., "name": ...}``."""
        return self._runs

    def watch(self, modules):
        if self._enabled:
            # wandb.watch(modules)
            pass

    def log(self, data: Mapping[str, Any], step: int, sleep=None):
        if step < self._last_step:
            raise ValueError(
                f"step {step} is less than last step {self._last_step}, "
                "steps must be logged in order"
            )
        self._last_step = step
        # sleep arg is ignored since we don't want to sleep in tests
        if self._enabled:
            self._logs[step].update(data)
        if self._disk_logger is not None:
            self._disk_logger.log(dict(data), step=step)

    def mark_checkpoint(self):
        if self._disk_logger is not None:
            self._disk_logger.write_checkpoint_mark()

    def _restore_disk_metrics(self, relog: bool):
        """Mirror wandb: a resumed run continues after the last step it
        received, and a committed step rejects later logs at that step.
        """
        if self._disk_logger is None:
            return
        mark = self._disk_logger.restore_to_checkpoint_mark()
        if mark is None or not relog:
            return
        received_steps = list(self._logs)
        if self._last_received_step is not None:
            received_steps.append(self._last_received_step)
        first_step = max(received_steps, default=-1) + 1
        for step, data, commit in wandb.metrics_to_relog(
            self._disk_logger.directory, first_step, mark
        ):
            self._last_step = step + 1 if commit else step
            self._logs[step].update(data)

    def drop_logs_after(self, step: int):
        """Simulate wandb never receiving logs after ``step``, e.g. because
        the job was killed before its background uploader synced them.
        """
        for logged_step in [s for s in self._logs if s > step]:
            del self._logs[logged_step]
        self._last_step = min(self._last_step, step)

    def get_logs(self) -> list[dict[str, Any]]:
        if len(self._logs) == 0:
            return []
        n_logs = max(self._logs.keys())
        return_value: list[dict[str, Any]] = [dict() for _ in range(n_logs + 1)]
        for step, log in self._logs.items():
            return_value[step] = log
        return return_value

    def Image(self, *args, **kwargs) -> wandb.Image:
        return wandb.Image(*args, direct_access=False, **kwargs)

    def Video(self, *args, **kwargs) -> wandb.Video:
        return wandb.Video(*args, direct_access=False, **kwargs)

    def Table(self, *args, **kwargs) -> wandb.Table:
        return wandb.Table(*args, direct_access=False, **kwargs)

    def Histogram(self, *args, **kwargs) -> wandb.Histogram:
        return wandb.Histogram(*args, direct_access=False, **kwargs)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def configured(self) -> bool:
        return self._configured


@contextlib.contextmanager
def mock_wandb():
    """
    Mock the distributed singleton to return a MockDistributed object.

    This is useful for testing that metrics are reduced across processes.

    It will make it so that when any tensor is reduced, it is filled with
    the given fill_value, which can be checked for in tests.
    """
    original = wandb.singleton
    mock = MockWandB()
    wandb.singleton = mock  # type: ignore
    try:
        yield mock
    finally:
        if mock._disk_logger is not None:
            mock._disk_logger.close()
        wandb.singleton = original


def _mock_wandb_id(n_chars: int = 8) -> str:
    return "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(n_chars)
    )
