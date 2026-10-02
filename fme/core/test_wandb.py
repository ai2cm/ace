import logging
from types import SimpleNamespace

import numpy as np
import pytest

import fme.core.wandb
from fme.core.disk_metric_logger import DiskMetricLogger, read_metrics
from fme.core.testing.wandb import mock_wandb
from fme.core.wandb import DirectInitializationError, Image, WandB


def test_image_is_image_instance():
    wandb = WandB.get_instance()
    img = wandb.Image(np.zeros((10, 10)))
    assert isinstance(img, Image)


def test_wandb_direct_initialization_raises():
    with pytest.raises(DirectInitializationError):
        Image(np.zeros((10, 10)))


class TestDiskLoggingIntegration:
    def test_metrics_written_to_disk_via_mock_wandb(self, tmp_path):
        log_dir = str(tmp_path / "metrics")
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True, metrics_log_dir=log_dir)
            wandb.log({"loss": 0.5, "lr": 1e-3}, step=0)
            wandb.log({"loss": 0.3, "lr": 1e-4}, step=1)

        records = read_metrics(log_dir)
        assert len(records) == 2
        assert records[0] == {"step": 0, "loss": 0.5, "lr": 1e-3}
        assert records[1] == {"step": 1, "loss": 0.3, "lr": 1e-4}

    def test_no_disk_logging_when_dir_is_none(self, tmp_path):
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True, metrics_log_dir=None)
            wandb.log({"loss": 0.5}, step=0)
        assert wandb._disk_logger is None

    def test_disk_logging_rerun_starts_fresh(self, tmp_path):
        log_dir = str(tmp_path / "metrics")
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True, metrics_log_dir=log_dir)
            wandb.log({"loss": 0.5}, step=0)
            wandb.log({"loss": 0.3}, step=1)

        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True, metrics_log_dir=log_dir)
            wandb.log({"loss": 0.45}, step=0)

        assert read_metrics(log_dir) == [{"step": 0, "loss": 0.45}]

    def test_disk_logging_independent_of_wandb_enabled(self, tmp_path):
        """Disk logging works even when log_to_wandb is False."""
        log_dir = str(tmp_path / "metrics")
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=False, metrics_log_dir=log_dir)
            wandb.log({"loss": 0.5}, step=0)

        records = read_metrics(log_dir)
        assert len(records) == 1
        assert records[0] == {"step": 0, "loss": 0.5}


def _log_previous_job(log_dir: str, records: list[tuple[dict, int]]) -> list[int]:
    """Log records as a previous job would, returning the offset after each."""
    previous_job = DiskMetricLogger(log_dir)
    offsets = []
    for data, step in records:
        previous_job.log(data, step=step)
        offsets.append(previous_job.offset)
    previous_job.close()
    return offsets


def _resumed_wandb(
    monkeypatch, log_dir: str, run_step: int, offline: bool = False
) -> tuple[WandB, list[tuple[dict, int, bool | None]]]:
    """A WandB whose resumed wandb run continues at ``run_step``, and the list
    its calls to wandb.log are recorded in.
    """
    logged: list[tuple[dict, int, bool | None]] = []
    monkeypatch.setattr(
        fme.core.wandb.wandb, "run", SimpleNamespace(step=run_step, offline=offline)
    )
    monkeypatch.setattr(
        fme.core.wandb.wandb,
        "log",
        lambda data, step, commit: logged.append((data, step, commit)),
    )
    wandb = WandB()
    wandb.configure(log_to_wandb=True, metrics_log_dir=log_dir)
    return wandb, logged


def _close_disk_logger(wandb: WandB):
    wandb.configure(log_to_wandb=False, metrics_log_dir=None)


PREVIOUS_JOB_RECORDS = [
    ({"batch_loss": 0.5}, 10),
    ({"batch_loss": 0.4}, 20),
    ({"val_loss": 0.3, "epoch": 2}, 20),
    ({"batch_loss": 0.2}, 30),
]


def test_restore_disk_metrics_relogs_rows_wandb_lacks(tmp_path, monkeypatch, caplog):
    log_dir = str(tmp_path / "metrics")
    offsets = _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS)
    # wandb received step 10, and the checkpoint was saved at step 20
    wandb, logged = _resumed_wandb(monkeypatch, log_dir, run_step=11)
    try:
        with caplog.at_level(logging.INFO):
            wandb.restore_disk_metrics(offsets[2], resume_step=20, step_continues=False)
        assert wandb.disk_metrics_offset == offsets[2]
    finally:
        _close_disk_logger(wandb)
    assert logged == [({"batch_loss": 0.4, "val_loss": 0.3, "epoch": 2}, 20, True)]
    assert (
        "Recovered wandb logs for 1 steps from disk (steps 20 to 20, epochs [2])"
        in caplog.messages
    )
    assert [r["step"] for r in read_metrics(log_dir)] == [10, 20, 20]


def test_restore_disk_metrics_leaves_continued_step_uncommitted(tmp_path, monkeypatch):
    log_dir = str(tmp_path / "metrics")
    offsets = _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS)
    wandb, logged = _resumed_wandb(monkeypatch, log_dir, run_step=0)
    try:
        wandb.restore_disk_metrics(offsets[1], resume_step=20, step_continues=True)
    finally:
        _close_disk_logger(wandb)
    assert logged == [
        ({"batch_loss": 0.5}, 10, True),
        ({"batch_loss": 0.4}, 20, False),
    ]


def test_restore_disk_metrics_warns_if_wandb_has_continued_step(
    tmp_path, monkeypatch, caplog
):
    log_dir = str(tmp_path / "metrics")
    offsets = _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS)
    wandb, logged = _resumed_wandb(monkeypatch, log_dir, run_step=21)
    try:
        with caplog.at_level(logging.WARNING):
            wandb.restore_disk_metrics(offsets[1], resume_step=20, step_continues=True)
    finally:
        _close_disk_logger(wandb)
    assert logged == []
    assert "wandb already received step 20" in caplog.text


def test_restore_disk_metrics_does_not_relog_to_offline_wandb(tmp_path, monkeypatch):
    log_dir = str(tmp_path / "metrics")
    offsets = _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS)
    wandb, logged = _resumed_wandb(monkeypatch, log_dir, run_step=0, offline=True)
    try:
        wandb.restore_disk_metrics(offsets[2], resume_step=20, step_continues=False)
    finally:
        _close_disk_logger(wandb)
    assert logged == []
    assert [r["step"] for r in read_metrics(log_dir)] == [10, 20, 20]


def test_restore_disk_metrics_without_offset_warns(tmp_path, monkeypatch, caplog):
    log_dir = str(tmp_path / "metrics")
    _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS)
    wandb, logged = _resumed_wandb(monkeypatch, log_dir, run_step=0)
    try:
        with caplog.at_level(logging.WARNING):
            wandb.restore_disk_metrics(None, resume_step=20, step_continues=False)
    finally:
        _close_disk_logger(wandb)
    assert logged == []
    assert "disk metric logging disabled" in caplog.text
    assert read_metrics(log_dir) == []


def test_restore_disk_metrics_through_step(tmp_path, monkeypatch, caplog):
    log_dir = str(tmp_path / "metrics")
    _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS)
    wandb, logged = _resumed_wandb(monkeypatch, log_dir, run_step=11)
    try:
        with caplog.at_level(logging.WARNING):
            wandb.restore_disk_metrics_through_step(20)
    finally:
        _close_disk_logger(wandb)
    assert logged == [({"batch_loss": 0.4, "val_loss": 0.3, "epoch": 2}, 20, True)]
    assert "does not record disk_metrics_offset" in caplog.text


def test_disk_metrics_offset_is_none_without_disk_logging():
    wandb = WandB()
    wandb.configure(log_to_wandb=False, metrics_log_dir=None)
    assert wandb.disk_metrics_offset is None
