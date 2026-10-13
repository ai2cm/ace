import dataclasses
import logging
import os
from collections.abc import Callable
from typing import Any

from fme.core.cloud import is_local
from fme.core.disk_metric_logger import (
    CheckpointMark,
    DiskMetricLogger,
    read_metrics_by_step,
)
from fme.core.distributed import Distributed


@dataclasses.dataclass(frozen=True)
class WandBLogCall:
    """The arguments of one call to ``wandb.log``."""

    data: dict[str, Any]
    step: int
    commit: bool


class MetricBackup:
    """A backup on disk of the scalar metrics this process logs, used by a
    resumed job to recover the metrics a wandb run never received.

    In distributed training every process has a rank, and only the root rank
    (rank 0) logs metrics, to wandb and to disk, so that one process owns the
    metrics file and the checkpoint mark. On other ranks, or when the metrics
    directory is not on a local file system, the backup is disabled and every
    method does nothing.

    Parameters:
        logger: The disk logger to write to, or None to disable the backup.
    """

    def __init__(self, logger: DiskMetricLogger | None = None):
        self._logger = logger

    @classmethod
    def from_directory(cls, metrics_log_dir: str) -> "MetricBackup":
        """Build this process's backup, writing to ``metrics_log_dir`` on the
        root rank. Disabled on other ranks, and on the root rank (with a
        warning) when ``metrics_log_dir`` is not on a local file system.
        """
        if not Distributed.get_instance().is_root():
            return cls()
        if not is_local(metrics_log_dir):
            logging.warning(
                f"Disk metric logging is only supported on a local file system. "
                f"Got metrics_log_dir={metrics_log_dir!r}, so no metrics will be "
                f"saved to disk and none can be recovered on resume."
            )
            return cls()
        return cls(DiskMetricLogger(metrics_log_dir))

    @property
    def enabled(self) -> bool:
        return self._logger is not None

    def log(self, data: dict[str, Any], step: int):
        if self._logger is not None:
            self._logger.log(data, step=step)

    def mark_checkpoint(self):
        """Write the checkpoint mark file, recording the current size of the
        metrics file and the last step logged.

        Call right after saving a checkpoint that training can resume from.
        Does no logging and no collectives, so it is safe to call from the
        termination listener's thread (see
        `fme.core.distributed.shutdown.add_post_abort_callback`).
        """
        if self._logger is not None:
            self._logger.write_checkpoint_mark()

    def restore(
        self,
        first_unreceived_step: Callable[[], int | None],
        log: Callable[[WandBLogCall], None],
    ):
        """Truncate the metrics file back to the checkpoint mark, then re-log
        the rows wandb never received (see ``metrics_to_relog``).

        Does nothing if the backup is disabled or no mark exists.

        Args:
            first_unreceived_step: Returns the first step the wandb run has
                not received, or None if nothing should be re-logged (e.g.
                no wandb run on this rank, an offline run, or a new run
                rather than a resumed one). Called only if a mark exists.
            log: Sends one re-logged row to wandb.
        """
        if self._logger is None:
            return
        mark = self._logger.restore_to_checkpoint_mark()
        if mark is None:
            return
        first_step = first_unreceived_step()
        if first_step is None:
            return
        logging.info(
            f"Checking disk metrics for steps wandb lacks: wandb resumed at "
            f"step {first_step}, checkpoint mark at step {mark.last_step}"
        )
        calls = metrics_to_relog(self._logger.directory, first_step, mark)
        for call in calls:
            log(call)
        if calls:
            epochs = [call.data["epoch"] for call in calls if "epoch" in call.data]
            logging.info(
                f"Recovered wandb logs for {len(calls)} steps from disk "
                f"(steps {calls[0].step} to {calls[-1].step}, epochs {epochs})"
            )

    def close(self):
        if self._logger is not None:
            self._logger.close()


def metrics_to_relog(
    directory: str | os.PathLike, first_step: int, mark: CheckpointMark
) -> list[WandBLogCall]:
    """Build the ``wandb.log`` calls that fill the gap between what wandb
    received before the previous job was killed and what that job logged to
    disk.

    For example, with these lines in the restored metrics file, a checkpoint
    mark with ``last_step=30``, and wandb having received step 10 so that
    ``first_step=11``::

        {"step": 10, "batch_loss": 0.5}
        {"step": 20, "batch_loss": 0.4}
        {"step": 20, "val_loss": 0.3}
        {"step": 30, "batch_loss": 0.2}

    the result is::

        [
            WandBLogCall({"batch_loss": 0.4, "val_loss": 0.3}, step=20, commit=True),
            WandBLogCall({"batch_loss": 0.2}, step=30, commit=False),
        ]

    Step 10 is dropped since wandb has it, the two step-20 lines are merged
    into one call, and step 30 is left uncommitted since it is the mark's
    ``last_step``.

    Args:
        directory: Directory holding the restored metrics file.
        first_step: The first step wandb does not have. On resume this is
            ``wandb.run.step``. Earlier steps are skipped, since wandb rejects
            logs at steps it already passed.
        mark: The checkpoint mark written by
            ``DiskMetricLogger.write_checkpoint_mark`` when the checkpoint
            being resumed from was saved. Only ``mark.last_step`` is used:
            the ``step`` passed to the last ``WandB.log`` call before that
            checkpoint was saved, i.e. the step the resumed job starts
            logging at again.

    Returns:
        One call per step from ``first_step`` on, in ascending order, each
        holding every metric logged at that step. Every call has
        ``commit=True`` except the one at ``mark.last_step``, the step the
        resumed job starts logging at again (e.g. end-of-epoch metrics after
        a checkpoint saved before validation). That call has
        ``commit=False`` so wandb holds the step open and the resumed job's
        logs merge into it instead of being rejected.
    """
    return [
        WandBLogCall(data=data, step=step, commit=step != mark.last_step)
        for step, data in read_metrics_by_step(directory, first_step).items()
    ]
