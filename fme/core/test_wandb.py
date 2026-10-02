import json
import logging
import os
import unittest.mock
from types import SimpleNamespace

import numpy as np
import pytest

import fme.core.disk_metric_logger
import fme.core.wandb
from fme.core.disk_metric_logger import CHECKPOINT_MARK_FILENAME, read_metrics
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

    def test_no_disk_logging_for_non_local_dir(self):
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True, metrics_log_dir="memory://b/metrics")
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


def _log_previous_job(log_dir: str, records: list[tuple[dict, int]], mark_after: int):
    """Log records as a previous job would, marking a checkpoint after the
    record at index ``mark_after``.
    """
    previous_job = WandB()
    previous_job.configure(log_to_wandb=False, metrics_log_dir=log_dir)
    for i, (data, step) in enumerate(records):
        previous_job.log(data, step=step)
        if i == mark_after:
            previous_job.mark_checkpoint()
    _close_disk_logger(previous_job)


def _resume_wandb(
    monkeypatch,
    tmp_path,
    log_dir: str,
    run_step: int,
    offline: bool = False,
    resumable: bool = True,
    new_run: bool = False,
) -> list[tuple[dict, int, bool | None]]:
    """Start a job whose resumed wandb run continues at ``run_step``, and
    return the calls it made to wandb.log. With ``new_run``, the experiment
    directory has no wandb run id, so the job starts a new wandb run instead.
    """
    if not new_run:
        with open(tmp_path / fme.core.wandb.WANDB_RUN_ID_FILE, "w") as f:
            f.write("run-id")
    logged: list[tuple[dict, int, bool | None]] = []
    monkeypatch.setattr(
        fme.core.wandb.wandb,
        "run",
        SimpleNamespace(id="run-id", step=run_step, offline=offline),
    )
    monkeypatch.setattr(fme.core.wandb.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        fme.core.wandb.wandb,
        "log",
        lambda data, step, commit: logged.append((data, step, commit)),
    )
    wandb = WandB()
    wandb.configure(log_to_wandb=True, metrics_log_dir=log_dir)
    try:
        wandb.init(resumable=resumable, experiment_dir=str(tmp_path))
    finally:
        _close_disk_logger(wandb)
    return logged


def _close_disk_logger(wandb: WandB):
    if wandb._disk_logger is not None:
        wandb._disk_logger.close()


# the checkpoint mark is after the step-30 batch logs; a timings log at step
# 30 and the next batch's logs at step 40 come after it
PREVIOUS_JOB_RECORDS = [
    ({"batch_loss": 0.5}, 10),
    ({"batch_loss": 0.4}, 20),
    ({"val_loss": 0.3, "epoch": 2}, 20),
    ({"batch_loss": 0.2}, 30),
    ({"epoch_seconds": 5.0}, 30),
    ({"batch_loss": 0.1}, 40),
]
MARK_AFTER = 3


def _discarded_steps(log_dir: str) -> list[int]:
    (discarded,) = [name for name in os.listdir(log_dir) if ".discarded." in name]
    with open(os.path.join(log_dir, discarded)) as f:
        return [json.loads(line)["step"] for line in f]


def test_resume_relogs_rows_wandb_lacks(tmp_path, monkeypatch, caplog):
    log_dir = str(tmp_path / "metrics")
    _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS, MARK_AFTER)
    # wandb received step 10
    with caplog.at_level(logging.INFO):
        logged = _resume_wandb(monkeypatch, tmp_path, log_dir, run_step=11)
    assert logged == [
        ({"batch_loss": 0.4, "val_loss": 0.3, "epoch": 2}, 20, True),
        ({"batch_loss": 0.2, "epoch_seconds": 5.0}, 30, False),
    ]
    assert (
        "Recovered wandb logs for 2 steps from disk (steps 20 to 30, epochs [2])"
        in caplog.messages
    )
    assert [r["step"] for r in read_metrics(log_dir)] == [10, 20, 20, 30, 30]
    assert _discarded_steps(log_dir) == [40]


def test_resume_does_not_relog_mark_step_wandb_received(tmp_path, monkeypatch):
    log_dir = str(tmp_path / "metrics")
    _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS, MARK_AFTER)
    logged = _resume_wandb(monkeypatch, tmp_path, log_dir, run_step=31)
    assert logged == []


def test_new_wandb_run_restores_disk_metrics_without_relogging(tmp_path, monkeypatch):
    log_dir = str(tmp_path / "metrics")
    _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS, MARK_AFTER)
    logged = _resume_wandb(monkeypatch, tmp_path, log_dir, run_step=0, new_run=True)
    assert logged == []
    assert [r["step"] for r in read_metrics(log_dir)] == [10, 20, 20, 30, 30]


def test_resume_without_checkpoint_mark_restores_nothing(tmp_path, monkeypatch):
    log_dir = str(tmp_path / "metrics")
    _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS, mark_after=-1)
    logged = _resume_wandb(monkeypatch, tmp_path, log_dir, run_step=0)
    assert logged == []
    assert read_metrics(log_dir) == []


def test_non_resumable_init_restores_nothing(tmp_path, monkeypatch):
    log_dir = str(tmp_path / "metrics")
    _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS, MARK_AFTER)
    logged = _resume_wandb(monkeypatch, tmp_path, log_dir, run_step=0, resumable=False)
    assert logged == []
    assert read_metrics(log_dir) == []


def test_resume_does_not_relog_to_offline_wandb(tmp_path, monkeypatch):
    log_dir = str(tmp_path / "metrics")
    _log_previous_job(log_dir, PREVIOUS_JOB_RECORDS, MARK_AFTER)
    logged = _resume_wandb(monkeypatch, tmp_path, log_dir, run_step=0, offline=True)
    assert logged == []
    assert [r["step"] for r in read_metrics(log_dir)] == [10, 20, 20, 30, 30]


def test_mark_checkpoint_does_not_use_logging_module(tmp_path, monkeypatch):
    # the mark is written on the termination listener's thread, where the
    # logging module can deadlock (see add_post_abort_callback)
    wandb = WandB()
    wandb.configure(log_to_wandb=False, metrics_log_dir=str(tmp_path / "metrics"))
    wandb.log({"loss": 0.5}, step=0)
    mock_logging = unittest.mock.MagicMock()
    monkeypatch.setattr(fme.core.wandb, "logging", mock_logging)
    monkeypatch.setattr(fme.core.disk_metric_logger, "logging", mock_logging)
    with unittest.mock.patch.object(logging.Logger, "_log") as logger_log:
        wandb.mark_checkpoint()
    _close_disk_logger(wandb)
    assert mock_logging.mock_calls == []
    logger_log.assert_not_called()
    assert os.path.exists(os.path.join(tmp_path, "metrics", CHECKPOINT_MARK_FILENAME))


def test_mark_checkpoint_without_disk_logging_is_a_no_op():
    wandb = WandB()
    wandb.configure(log_to_wandb=False, metrics_log_dir="memory://b/metrics")
    wandb.mark_checkpoint()
    assert wandb._disk_logger is None
