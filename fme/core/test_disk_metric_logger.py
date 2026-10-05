import json
import logging
import math
import os

import pytest

from fme.core import disk_metric_logger
from fme.core.disk_metric_logger import (
    CHECKPOINT_MARK_FILENAME,
    METRICS_FILENAME,
    DiskMetricLogger,
    read_metrics,
    read_metrics_by_step,
)


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


def _log_steps(log_dir: str, steps: range, mark_after: int | None = None):
    """Log one record per step, writing a checkpoint mark after ``mark_after``."""
    logger = DiskMetricLogger(log_dir)
    for step in steps:
        logger.log({"loss": float(step)}, step=step)
        if step == mark_after:
            logger.write_checkpoint_mark()
    logger.close()


def _other_files(log_dir: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(log_dir)
        if name not in (METRICS_FILENAME, CHECKPOINT_MARK_FILENAME)
    )


def _read_lines(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _read_mark(log_dir: str) -> dict:
    with open(os.path.join(log_dir, CHECKPOINT_MARK_FILENAME)) as f:
        return json.load(f)


def _write_mark(log_dir: str, offset: int, last_step: int | None):
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, CHECKPOINT_MARK_FILENAME), "w") as f:
        json.dump({"offset": offset, "last_step": last_step}, f)


def test_previous_job_metrics_are_moved_aside(log_dir):
    _log_steps(log_dir, range(3))

    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 9.0}, step=0)
    logger.close()

    assert read_metrics(log_dir) == [{"step": 0, "loss": 9.0}]
    (previous,) = _other_files(log_dir)
    assert [r["step"] for r in _read_lines(os.path.join(log_dir, previous))] == [
        0,
        1,
        2,
    ]


def test_moved_aside_files_get_unique_names(log_dir):
    for _ in range(3):
        _log_steps(log_dir, range(1))
    DiskMetricLogger(log_dir).close()
    assert len(_other_files(log_dir)) == 3


def test_checkpoint_mark_records_offset_and_last_step(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 0.0}, step=0)
    logger.log({"image": object()}, step=1)  # nothing on disk, but logged
    logger.write_checkpoint_mark()
    assert _read_mark(log_dir) == {"offset": logger.offset, "last_step": 1}
    logger.close()
    assert sorted(os.listdir(log_dir)) == sorted(
        [METRICS_FILENAME, CHECKPOINT_MARK_FILENAME]
    )


def test_checkpoint_mark_before_any_log(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.write_checkpoint_mark()
    logger.close()
    assert _read_mark(log_dir) == {"offset": 0, "last_step": None}


def test_checkpoint_mark_is_replaced_atomically(log_dir, monkeypatch):
    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 0.0}, step=0)
    logger.write_checkpoint_mark()
    previous_mark = _read_mark(log_dir)
    logger.log({"loss": 1.0}, step=1)

    def fail_mid_write(obj, f):
        f.write('{"offset": ')
        raise OSError("killed mid-write")

    monkeypatch.setattr(disk_metric_logger.json, "dump", fail_mid_write)
    with pytest.raises(OSError):
        logger.write_checkpoint_mark()
    monkeypatch.undo()
    logger.close()
    assert _read_mark(log_dir) == previous_mark


def test_restore_to_checkpoint_mark_cuts_and_continues(log_dir):
    _log_steps(log_dir, range(5), mark_after=2)

    logger = DiskMetricLogger(log_dir)
    mark = logger.restore_to_checkpoint_mark()
    assert mark is not None and mark.last_step == 2
    assert logger.offset == mark.offset
    logger.log({"loss": 30.0}, step=3)
    logger.close()

    assert read_metrics(log_dir) == [
        {"step": 0, "loss": 0.0},
        {"step": 1, "loss": 1.0},
        {"step": 2, "loss": 2.0},
        {"step": 3, "loss": 30.0},
    ]
    (discarded,) = _other_files(log_dir)
    assert ".discarded." in discarded
    assert _read_lines(os.path.join(log_dir, discarded)) == [
        {"step": 3, "loss": 3.0},
        {"step": 4, "loss": 4.0},
    ]


def test_restore_keeps_records_at_the_mark_step_logged_after_it(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.log({"batch_loss": 1.0}, step=5)
    logger.write_checkpoint_mark()
    logger.log({"timing": 2.0}, step=5)  # logged after the checkpoint
    logger.log({"batch_loss": 3.0}, step=6)
    logger.close()

    logger = DiskMetricLogger(log_dir)
    assert logger.restore_to_checkpoint_mark() is not None
    logger.log({"val_loss": 4.0}, step=5)
    logger.close()

    assert read_metrics_by_step(log_dir, first_step=0) == {
        5: {"batch_loss": 1.0, "timing": 2.0, "val_loss": 4.0}
    }
    (discarded,) = _other_files(log_dir)
    assert _read_lines(os.path.join(log_dir, discarded)) == [
        {"step": 6, "batch_loss": 3.0}
    ]


def test_restore_after_logging_cuts_current_file(log_dir):
    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 0.0}, step=0)
    logger.write_checkpoint_mark()
    logger.log({"loss": 1.0}, step=1)
    assert logger.restore_to_checkpoint_mark() is not None
    logger.log({"loss": 10.0}, step=1)
    logger.close()

    assert read_metrics(log_dir) == [
        {"step": 0, "loss": 0.0},
        {"step": 1, "loss": 10.0},
    ]


def test_restore_without_checkpoint_mark_keeps_previous_file_aside(log_dir):
    _log_steps(log_dir, range(2))

    logger = DiskMetricLogger(log_dir)
    assert logger.restore_to_checkpoint_mark() is None
    logger.close()

    assert read_metrics(log_dir) == []
    assert len(_other_files(log_dir)) == 1


def test_restore_without_previous_file_warns(log_dir, caplog):
    _write_mark(log_dir, offset=10, last_step=0)
    logger = DiskMetricLogger(log_dir)
    with caplog.at_level(logging.WARNING):
        assert logger.restore_to_checkpoint_mark() is None
    logger.close()
    assert "no disk metrics are restored" in caplog.text


def test_restore_after_metrics_file_is_deleted_warns(log_dir, caplog):
    logger = DiskMetricLogger(log_dir)
    logger.log({"loss": 0.0}, step=0)
    logger.write_checkpoint_mark()
    os.remove(os.path.join(log_dir, METRICS_FILENAME))
    with caplog.at_level(logging.WARNING):
        assert logger.restore_to_checkpoint_mark() is None
    logger.close()
    assert "no disk metrics are restored" in caplog.text


def test_restore_from_shorter_file_warns_and_keeps_it_aside(log_dir, caplog):
    _log_steps(log_dir, range(2))
    size = os.path.getsize(os.path.join(log_dir, METRICS_FILENAME))
    _write_mark(log_dir, offset=size + 1, last_step=1)

    logger = DiskMetricLogger(log_dir)
    with caplog.at_level(logging.WARNING):
        assert logger.restore_to_checkpoint_mark() is None
    logger.log({"loss": 5.0}, step=0)
    logger.close()

    assert "not the checkpoint's metrics file" in caplog.text
    assert read_metrics(log_dir) == [{"step": 0, "loss": 5.0}]
    assert len(_other_files(log_dir)) == 1


def test_restore_drops_line_cut_off_mid_write(log_dir):
    _log_steps(log_dir, range(2), mark_after=1)
    with open(os.path.join(log_dir, METRICS_FILENAME), "a") as f:
        f.write('{"step": 1, "lo')

    logger = DiskMetricLogger(log_dir)
    assert logger.restore_to_checkpoint_mark() is not None
    logger.log({"loss": 2.0}, step=2)
    logger.close()

    assert [r["step"] for r in read_metrics(log_dir)] == [0, 1, 2]


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
    """Corrupt lines, and lines without an integer step, are skipped."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, METRICS_FILENAME)
    with open(path, "w") as f:
        f.write(json.dumps({"step": 0, "loss": 0.5}) + "\n")
        f.write("NOT VALID JSON\n")
        f.write(json.dumps({"loss": 0.4}) + "\n")
        f.write(json.dumps({"step": "1", "loss": 0.4}) + "\n")
        f.write(json.dumps([1, 2]) + "\n")
        f.write(json.dumps({"step": 2, "loss": 0.3}) + "\n")

    records = read_metrics(log_dir)
    assert records == [{"step": 0, "loss": 0.5}, {"step": 2, "loss": 0.3}]
    assert read_metrics_by_step(log_dir, first_step=1) == {2: {"loss": 0.3}}


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
