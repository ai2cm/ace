import io
import json
import logging
import os
import time
from typing import Any

METRICS_FILENAME = "metrics.jsonl"


class DiskMetricLogger:
    """Logs scalar metrics to a JSONL file on disk.

    Each line in the file is a JSON object with a "step" key and scalar metric
    key-value pairs. On construction, a metrics file a previous job left in the
    directory is moved aside, so the file holds only this job's metrics. A job
    resuming from a checkpoint calls ``restore`` to bring back the metrics
    logged up to the checkpoint.

    Non-JSON-serializable values (e.g. images, tensors) are silently dropped.
    """

    def __init__(self, directory: str | os.PathLike):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self._path = os.path.join(directory, METRICS_FILENAME)
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

    @property
    def offset(self) -> int:
        """Size in bytes of the metrics file, which a checkpoint records so a
        job resuming from it can ``restore`` the metrics logged before it.
        """
        return self._offset

    def log(self, data: dict[str, Any], step: int) -> None:
        """Log scalar metrics for a given step.

        Non-serializable values are dropped.
        """
        scalars = _extract_serializable(data)
        if not scalars:
            return
        line = (json.dumps({"step": step, **scalars}) + "\n").encode()
        if self._file is None:
            raise RuntimeError("DiskMetricLogger is closed")
        self._file.write(line)
        self._file.flush()
        self._offset += len(line)

    def restore(self, offset: int) -> bool:
        """Restore the metrics logged before a checkpoint.

        Cuts the metrics file at ``offset``, the checkpoint's ``offset``. If
        this logger has not logged anything, the file is the one a previous job
        left, which is moved back into place first. Metrics logged after the
        checkpoint are from training the resumed job redoes, so they are moved
        to a separate file rather than kept.

        Returns:
            Whether the metrics were restored. They are not if there is no
            metrics file to restore, or if it is shorter than ``offset`` (e.g.
            ``metrics_log_dir`` changed since the checkpoint was saved).
        """
        source_path = self._path if self._offset > 0 else self._previous_path
        if source_path is None:
            logging.warning(
                "No metrics file from a previous job in %s, so no disk metrics "
                "are restored",
                self.directory,
            )
            return False
        source_size = os.path.getsize(source_path)
        if source_size < offset:
            logging.warning(
                "Metrics file %s has %d bytes but the checkpoint expects at "
                "least %d, so it is not the checkpoint's metrics file and no "
                "disk metrics are restored",
                source_path,
                source_size,
                offset,
            )
            return False
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
        return True

    def restore_through_step(self, last_step: int) -> bool:
        """Restore the previous job's metrics through ``last_step``, for a
        checkpoint that does not record an ``offset``.

        The metrics are cut before the first line with a step after
        ``last_step``. See ``restore`` for the rest.
        """
        source_path = self._path if self._offset > 0 else self._previous_path
        if source_path is None:
            return self.restore(0)
        offset = 0
        with open(source_path, "rb") as f:
            for line in f:
                if not line.endswith(b"\n"):
                    break
                record = _parse_record(line)
                if record is not None and record["step"] > last_step:
                    break
                offset += len(line)
        return self.restore(offset)

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
