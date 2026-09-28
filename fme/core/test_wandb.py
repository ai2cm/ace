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

    def test_disk_logging_resume_skips_old_steps(self, tmp_path):
        log_dir = str(tmp_path / "metrics")
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True, metrics_log_dir=log_dir)
            wandb.log({"loss": 0.5}, step=0)
            wandb.log({"loss": 0.3}, step=1)

        # Simulate resume
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True, metrics_log_dir=log_dir)
            wandb.log({"loss": 0.45}, step=0)  # skipped
            wandb.log({"loss": 0.35}, step=1)  # skipped
            wandb.log({"loss": 0.2}, step=2)  # written

        records = read_metrics(log_dir)
        assert len(records) == 3
        assert records[0]["loss"] == 0.5  # original preserved
        assert records[1]["loss"] == 0.3  # original preserved
        assert records[2] == {"step": 2, "loss": 0.2}

    def test_disk_logging_independent_of_wandb_enabled(self, tmp_path):
        """Disk logging works even when log_to_wandb is False."""
        log_dir = str(tmp_path / "metrics")
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=False, metrics_log_dir=log_dir)
            wandb.log({"loss": 0.5}, step=0)

        records = read_metrics(log_dir)
        assert len(records) == 1
        assert records[0] == {"step": 0, "loss": 0.5}


def test_log_unsynced_from_disk_relogs_rows_wandb_lacks(tmp_path, monkeypatch, caplog):
    log_dir = str(tmp_path / "metrics")
    previous_job = DiskMetricLogger(log_dir)
    previous_job.log({"batch_loss": 0.5}, step=10)
    previous_job.log({"batch_loss": 0.4}, step=20)
    previous_job.log({"val_loss": 0.3, "epoch": 2}, step=20)
    previous_job.log({"batch_loss": 0.2}, step=30)  # after the checkpoint
    previous_job.close()
    logged: list[tuple[dict, int, bool | None]] = []
    # wandb received step 10, so the resumed run continues from step 11
    monkeypatch.setattr(fme.core.wandb.wandb, "run", SimpleNamespace(step=11))
    monkeypatch.setattr(
        fme.core.wandb.wandb,
        "log",
        lambda data, step, commit: logged.append((data, step, commit)),
    )
    wandb = WandB()
    wandb.configure(log_to_wandb=True, metrics_log_dir=log_dir)
    try:
        with caplog.at_level(logging.INFO):
            wandb.log_unsynced_from_disk(max_step=20)
    finally:
        # reconfiguring closes the disk logger's file
        wandb.configure(log_to_wandb=False, metrics_log_dir=None)
    assert logged == [({"batch_loss": 0.4, "val_loss": 0.3, "epoch": 2}, 20, True)]
    assert (
        "Recovered wandb logs for 1 steps from disk (steps 20 to 20, epochs [2])"
        in caplog.messages
    )
