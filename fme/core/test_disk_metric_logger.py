import json
import math
import os

import pytest

from fme.core.disk_metric_logger import METRICS_FILENAME, DiskMetricLogger, read_metrics


@pytest.fixture
def log_dir(tmp_path):
    return str(tmp_path / "metrics")


def test_creates_directory(log_dir):
    assert not os.path.exists(log_dir)
    logger = DiskMetricLogger(log_dir)
    assert os.path.isdir(log_dir)
    logger.close()


def test_log_writes_jsonl(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 0.5, "lr": 1e-3}, step=0)
    logger.log({"loss": 0.3, "lr": 1e-4}, step=1)
    logger.close()

    records = read_metrics(log_dir)
    assert len(records) == 2
    assert records[0] == {"step": 0, "loss": 0.5, "lr": 1e-3}
    assert records[1] == {"step": 1, "loss": 0.3, "lr": 1e-4}


def test_log_flushes_each_line(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 0.5}, step=0)
    # Read before close — data should be on disk due to flush
    records = read_metrics(log_dir)
    assert len(records) == 1
    logger.close()


def test_resume_skips_steps_at_or_below_high_water_mark(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 0.5}, step=0)
    logger.log({"loss": 0.4}, step=1)
    logger.log({"loss": 0.3}, step=2)
    logger.close()

    # Simulate resume: re-create logger, re-log steps 1 and 2,
    # then continue with step 3
    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 0.45}, step=1)  # skipped
    logger.log({"loss": 0.35}, step=2)  # skipped
    logger.log({"loss": 0.2}, step=3)  # written
    logger.close()

    records = read_metrics(log_dir)
    assert len(records) == 4
    # Original steps preserved
    assert records[0]["step"] == 0
    assert records[1]["step"] == 1
    assert records[1]["loss"] == 0.4  # original, not overwritten
    assert records[2]["step"] == 2
    assert records[2]["loss"] == 0.3  # original, not overwritten
    # New step appended
    assert records[3] == {"step": 3, "loss": 0.2}


def test_resume_catches_up_then_continues(log_dir):
    """Steps exactly at the high-water mark are skipped; one above is written."""
    logger = DiskMetricLogger(log_dir)
    logger.log({"a": 1}, step=5)
    logger.close()

    logger = DiskMetricLogger(log_dir)
    logger.log({"a": 2}, step=5)  # skipped (== high water mark)
    logger.log({"a": 3}, step=6)  # written
    logger.close()

    records = read_metrics(log_dir)
    assert len(records) == 2
    assert records[0] == {"step": 5, "a": 1}
    assert records[1] == {"step": 6, "a": 3}


def test_non_scalar_values_are_skipped(log_dir):
    logger = DiskMetricLogger(log_dir)

    class NotSerializable:
        pass

    logger.log(
        {"loss": 0.5, "image": NotSerializable(), "count": 10},
        step=0,
    )
    logger.close()

    records = read_metrics(log_dir)
    assert len(records) == 1
    assert records[0] == {"step": 0, "loss": 0.5, "count": 10}


def test_all_non_scalar_skips_entire_line(log_dir):
    """If all values are non-serializable, no line is written."""

    class NotSerializable:
        pass

    logger = DiskMetricLogger(log_dir)
    logger.log({"image": NotSerializable()}, step=0)
    logger.close()

    records = read_metrics(log_dir)
    assert len(records) == 0


def test_empty_data_skips_line(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.log({}, step=0)
    logger.close()

    records = read_metrics(log_dir)
    assert len(records) == 0


def test_string_values_are_logged(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.log({"phase": "train", "loss": 0.5}, step=0)
    logger.close()

    records = read_metrics(log_dir)
    assert records[0] == {"step": 0, "phase": "train", "loss": 0.5}


def test_bool_values_are_logged(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.log({"converged": True}, step=0)
    logger.close()

    records = read_metrics(log_dir)
    assert records[0] == {"step": 0, "converged": True}


def test_no_existing_file(log_dir):
    """Logger works when directory exists but no metrics file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = DiskMetricLogger(log_dir)
    logger.log({"x": 1}, step=0)
    logger.close()

    records = read_metrics(log_dir)
    assert len(records) == 1


def test_corrupt_line_is_skipped(log_dir):
    """A corrupt line in the file doesn't prevent reading or resuming."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, METRICS_FILENAME)
    with open(path, "w") as f:
        f.write(json.dumps({"step": 0, "loss": 0.5}) + "\n")
        f.write("NOT VALID JSON\n")
        f.write(json.dumps({"step": 2, "loss": 0.3}) + "\n")

    logger = DiskMetricLogger(log_dir)
    # High water mark should be 2 despite the corrupt line
    logger.log({"loss": 0.1}, step=2)  # skipped
    logger.log({"loss": 0.05}, step=3)  # written
    logger.close()

    records = read_metrics(log_dir)
    # read_metrics also skips corrupt lines
    assert len(records) == 3
    assert records[0]["step"] == 0
    assert records[1]["step"] == 2
    assert records[2]["step"] == 3


def test_read_metrics_empty_directory(tmp_path):
    """read_metrics returns empty list when no file exists."""
    assert read_metrics(str(tmp_path)) == []


def test_multiple_logs_same_step(log_dir):
    """Multiple log calls at the same step each produce a line."""
    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 0.5}, step=0)
    logger.log({"lr": 1e-3}, step=0)
    logger.close()

    records = read_metrics(log_dir)
    assert len(records) == 2
    assert records[0] == {"step": 0, "loss": 0.5}
    assert records[1] == {"step": 0, "lr": 1e-3}


def test_nan_and_inf_values_are_logged(log_dir):
    """NaN and Inf are valid floats and get logged.

    Note: Python's json.dumps produces non-standard NaN/Infinity tokens.
    Use read_metrics (which uses json.loads) to round-trip these values
    rather than strict JSON parsers like jq.
    """
    logger = DiskMetricLogger(log_dir)
    logger.log({"nan_val": float("nan"), "inf_val": float("inf")}, step=0)
    logger.close()

    records = read_metrics(log_dir)
    assert len(records) == 1
    assert math.isnan(records[0]["nan_val"])
    assert records[0]["inf_val"] == float("inf")
