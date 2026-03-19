import io
import json
import logging
import os
from typing import Any

METRICS_FILENAME = "metrics.jsonl"


class DiskMetricLogger:
    """Logs scalar metrics to a JSONL file on disk.

    Each line in the file is a JSON object with a "step" key and scalar metric
    key-value pairs. On construction, any existing file is read to determine the
    high-water mark (maximum step already logged). Subsequent calls to ``log``
    with a step at or below that mark are silently skipped, which makes this
    logger safe for job resumption.

    Non-JSON-serializable values (e.g. images, tensors) are silently dropped.
    """

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self._path = os.path.join(directory, METRICS_FILENAME)
        self._high_water_mark: int | None = None
        self._file: io.TextIOWrapper | None = None
        self._read_high_water_mark()
        self._file = open(self._path, "a")

    def _read_high_water_mark(self):
        """Read the existing file (if any) to find the max step logged."""
        if not os.path.exists(self._path):
            return
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = record.get("step")
                if isinstance(step, int):
                    if self._high_water_mark is None or step > self._high_water_mark:
                        self._high_water_mark = step

    def log(self, data: dict[str, Any], step: int) -> None:
        """Log scalar metrics for a given step.

        If ``step`` is at or below the high-water mark from a previous run,
        the call is silently skipped.  Non-serializable values are dropped.
        """
        if self._high_water_mark is not None and step <= self._high_water_mark:
            return
        scalars = _extract_scalars(data)
        if not scalars:
            return
        record = {"step": step, **scalars}
        assert self._file is not None
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def _extract_scalars(data: dict[str, Any]) -> dict[str, Any]:
    """Return only JSON-serializable scalar entries from *data*."""
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


def read_metrics(directory: str) -> list[dict[str, Any]]:
    """Read all metric records from a metrics JSONL file.

    Returns a list of dicts, one per logged line, in file order.
    """
    path = os.path.join(directory, METRICS_FILENAME)
    records: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
