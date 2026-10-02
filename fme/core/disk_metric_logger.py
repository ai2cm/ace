import dataclasses
import io
import json
import logging
import os
import time
from typing import Any

METRICS_FILENAME = "metrics.jsonl"
CHECKPOINT_MARK_FILENAME = "checkpoint_mark.json"


@dataclasses.dataclass(frozen=True)
class CheckpointMark:
    """Where the metrics file stood when a resume checkpoint was saved.

    Parameters:
        offset: Size in bytes of the metrics file.
        last_step: The last step logged, or None if nothing was logged.
    """

    offset: int
    last_step: int | None


class DiskMetricLogger:
    """Logs scalar metrics to a JSONL file on disk.

    Each line in the file is a JSON object with a "step" key and scalar metric
    key-value pairs. On construction, a metrics file a previous job left in the
    directory is moved aside, so the file holds only this job's metrics.
    ``write_checkpoint_mark`` records in the directory where the file stands
    when a resume checkpoint is saved, and a job resuming from that checkpoint
    calls ``restore_to_checkpoint_mark`` to bring back the metrics logged up to
    it.

    Non-JSON-serializable values (e.g. images, tensors) are silently dropped.
    """

    def __init__(self, directory: str | os.PathLike):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self._path = os.path.join(directory, METRICS_FILENAME)
        self._mark_path = os.path.join(directory, CHECKPOINT_MARK_FILENAME)
        self._previous_path: str | None = None
        if os.path.exists(self._path) and os.path.getsize(self._path) > 0:
            self._previous_path = _unused_path(f"{self._path}.{_timestamp()}")
            os.rename(self._path, self._previous_path)
            logging.info(
                "Moved metrics from a previous job in this directory to %s",
                self._previous_path,
            )
        self._file: io.BufferedWriter | None = open(self._path, "wb")
        self._offset = 0
        self._last_step: int | None = None

    @property
    def offset(self) -> int:
        """Size in bytes of the metrics file."""
        return self._offset

    def log(self, data: dict[str, Any], step: int) -> None:
        """Log scalar metrics for a given step.

        Non-serializable values are dropped.
        """
        self._last_step = step
        scalars = _extract_serializable(data)
        if not scalars:
            return
        line = (json.dumps({"step": step, **scalars}) + "\n").encode()
        if self._file is None:
            raise RuntimeError("DiskMetricLogger is closed")
        self._file.write(line)
        self._file.flush()
        self._offset += len(line)

    def write_checkpoint_mark(self) -> None:
        """Record that a resume checkpoint holds the training logged so far.

        Writes the current offset and last logged step to the checkpoint mark
        file, replacing it atomically. This runs on the termination listener's
        thread when a job is preempted, so it must not use the logging module
        (see `fme.core.distributed.shutdown.add_post_abort_callback`).
        """
        tmp_path = f"{self._mark_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump({"offset": self._offset, "last_step": self._last_step}, f)
        os.replace(tmp_path, self._mark_path)

    def restore_to_checkpoint_mark(self) -> CheckpointMark | None:
        """Restore the metrics logged before the last checkpoint mark.

        Keeps the metrics file up to the mark, plus the lines right after it at
        the mark's last step, which the resumed job does not necessarily log
        again. If this logger has not logged anything, the file is the one a
        previous job left, which is moved back into place first. The lines
        after those are from training the resumed job redoes, so they are moved
        to a separate file rather than kept.

        Returns:
            The mark the metrics were restored to, or None if they were not
            restored: there is no mark, no metrics file to restore, or the file
            is shorter than the mark's offset.
        """
        mark = self._read_checkpoint_mark()
        if mark is None:
            return None
        source_path = self._restore_source_path()
        if source_path is None:
            logging.warning(
                "No metrics file to restore in %s, so no disk metrics are restored",
                self.directory,
            )
            return None
        source_size = os.path.getsize(source_path)
        if source_size < mark.offset:
            logging.warning(
                "Metrics file %s has %d bytes but the checkpoint mark expects at "
                "least %d, so it is not the checkpoint's metrics file and no "
                "disk metrics are restored",
                source_path,
                source_size,
                mark.offset,
            )
            return None
        offset = mark.offset
        with open(source_path, "rb") as f:
            f.seek(offset)
            for line in f:
                if not line.endswith(b"\n"):
                    break
                record = _parse_record(line)
                if record is not None and record["step"] != mark.last_step:
                    break
                offset += len(line)
        self._restore(source_path, offset)
        self._last_step = mark.last_step
        return mark

    def _read_checkpoint_mark(self) -> CheckpointMark | None:
        try:
            with open(self._mark_path) as f:
                mark = json.load(f)
        except FileNotFoundError:
            return None
        return CheckpointMark(offset=mark["offset"], last_step=mark["last_step"])

    def _restore(self, source_path: str, offset: int) -> None:
        """Make ``source_path`` this logger's file, cut at ``offset``."""
        self.close()
        if source_path != self._path:
            os.replace(source_path, self._path)
            self._previous_path = None
        with open(self._path, "r+b") as f:
            f.seek(offset)
            discarded = f.read()
            if discarded:
                discarded_path = _unused_path(f"{self._path}.discarded.{_timestamp()}")
                with open(discarded_path, "wb") as discarded_file:
                    discarded_file.write(discarded)
                logging.info(
                    "Moved metrics logged after the checkpoint to %s",
                    discarded_path,
                )
            f.truncate(offset)
        self._file = open(self._path, "ab")
        self._offset = offset

    def _restore_source_path(self) -> str | None:
        """The metrics file to restore: this logger's own if it has logged
        anything, otherwise the one a previous job left. None if that file
        does not exist.
        """
        source_path = self._path if self._offset > 0 else self._previous_path
        if source_path is None or not os.path.exists(source_path):
            return None
        return source_path

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _unused_path(path: str) -> str:
    candidate = path
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{path}.{suffix}"
        suffix += 1
    return candidate


def _parse_record(line: str | bytes) -> dict[str, Any] | None:
    """Parse a metrics line, or return None if it is not a valid record."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or not isinstance(record.get("step"), int):
        return None
    return record


def _extract_serializable(data: dict[str, Any]) -> dict[str, Any]:
    """Return only JSON-serializable entries from *data*."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, int | float | bool):
            result[key] = value
        elif isinstance(value, str):
            result[key] = value
        else:
            try:
                json.dumps(value)
            except (TypeError, ValueError, OverflowError):
                logging.debug(
                    f"DiskMetricLogger: skipping non-serializable key '{key}'"
                )
            else:
                result[key] = value
    return result


def read_metrics(directory: str | os.PathLike) -> list[dict[str, Any]]:
    """Read all metric records from a metrics JSONL file.

    Returns a list of dicts, one per logged line, in file order. Lines that are
    not valid records (e.g. cut off by the job being killed mid-write, or
    missing an integer "step") are skipped.
    """
    path = os.path.join(directory, METRICS_FILENAME)
    records: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            record = _parse_record(line)
            if record is not None:
                records.append(record)
    return records


def read_metrics_by_step(
    directory: str | os.PathLike, first_step: int
) -> dict[int, dict[str, Any]]:
    """Read metric records with ``step >= first_step``.

    Records logged at the same step are merged, since a step may be logged in
    several calls. Returns a dict from step to metrics, excluding the "step"
    key, sorted by step.
    """
    by_step: dict[int, dict[str, Any]] = {}
    for record in read_metrics(directory):
        record = dict(record)
        step = record.pop("step")
        if step >= first_step:
            by_step.setdefault(step, {}).update(record)
    return dict(sorted(by_step.items()))
