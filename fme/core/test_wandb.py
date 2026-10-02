import contextlib
import json
import os
import unittest.mock

import numpy as np
import pytest
import torch

from fme.core.disk_metric_logger import read_metrics
from fme.core.testing.wandb import mock_wandb
from fme.core.wandb import (
    COMPACT_EVERY_N_LINES,
    N_RECENT_ROWS,
    RECENT_ROWS_FILE,
    DirectInitializationError,
    Image,
    WandB,
    read_recent_rows,
)


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


def _init_resumable(experiment_dir: str, log_to_wandb: bool = True) -> WandB:
    """A real WandB resuming the run in ``experiment_dir``, for use while
    ``fme.core.wandb.wandb`` is mocked.
    """
    wandb = WandB()
    wandb.configure(log_to_wandb=log_to_wandb)
    wandb.init(resumable=True, experiment_dir=experiment_dir)
    return wandb


def _read_lines(experiment_dir: str) -> list[dict]:
    with open(os.path.join(experiment_dir, RECENT_ROWS_FILE)) as f:
        return [json.loads(line) for line in f]


@contextlib.contextmanager
def _mock_wandb_module(next_step: int = 0, offline: bool = False):
    with unittest.mock.patch("fme.core.wandb.wandb") as module:
        module.run.id = "run_id"
        module.run.step = next_step
        module.run.offline = offline
        yield module


def _logged(module: unittest.mock.MagicMock) -> list[tuple[int, dict, bool | None]]:
    return [
        (call.kwargs["step"], call.args[0], call.kwargs["commit"])
        for call in module.log.call_args_list
    ]


class TestRecentRows:
    def test_log_is_readable_from_file_immediately(self, tmp_path):
        with _mock_wandb_module():
            wandb = _init_resumable(str(tmp_path))
            wandb.log({"a": 1}, step=3)
            # read while the job still holds the file open
            assert _read_lines(str(tmp_path)) == [{"step": 3, "logs": {"a": 1}}]

    def test_logs_at_one_step_merge_into_one_scalar_row(self, tmp_path):
        with _mock_wandb_module():
            wandb = _init_resumable(str(tmp_path))
            wandb.log({"a": 1, "b": 0, "image": object()}, step=3)
            wandb.log({"b": np.float32(2.0), "c": torch.tensor(3.0)}, step=3)
            wandb.log({"a": 4}, step=4)
            path = os.path.join(str(tmp_path), RECENT_ROWS_FILE)
            assert read_recent_rows(path) == {
                3: {"a": 1, "b": 2.0, "c": 3.0},
                4: {"a": 4},
            }

    def test_torn_last_line_is_skipped(self, tmp_path):
        path = os.path.join(str(tmp_path), RECENT_ROWS_FILE)
        with open(path, "w") as f:
            f.write('{"step": 3, "logs": {"a": 1}}\n{"step": 4, "lo')
        assert read_recent_rows(path) == {3: {"a": 1}}

    def test_compaction_keeps_latest_rows_and_appends_after(self, tmp_path):
        n_steps = COMPACT_EVERY_N_LINES + 2
        with _mock_wandb_module():
            wandb = _init_resumable(str(tmp_path))
            for step in range(n_steps):
                wandb.log({"a": step}, step=step)
            lines = _read_lines(str(tmp_path))
            assert len(lines) < COMPACT_EVERY_N_LINES
            wandb.log({"b": 0}, step=n_steps - 1)
            assert _read_lines(str(tmp_path))[-1] == {
                "step": n_steps - 1,
                "logs": {"b": 0},
            }
            rows = read_recent_rows(os.path.join(str(tmp_path), RECENT_ROWS_FILE))
        assert sorted(rows) == list(range(n_steps - N_RECENT_ROWS, n_steps))
        assert rows[n_steps - 1] == {"a": n_steps - 1, "b": 0}

    def test_restore_relogs_unsent_rows_up_to_resume_step(self, tmp_path):
        with _mock_wandb_module():
            previous = _init_resumable(str(tmp_path))
            for step in range(2, 8):
                previous.log({"a": step}, step=step)
        with _mock_wandb_module(next_step=4) as module:
            wandb = _init_resumable(str(tmp_path))
            wandb.restore_unsent(resume_step=6)
            # rows after the resume step are logged again by the resumed job
            assert sorted(
                read_recent_rows(os.path.join(str(tmp_path), RECENT_ROWS_FILE))
            ) == [2, 3, 4, 5, 6]
        assert _logged(module) == [
            (4, {"a": 4}, True),
            (5, {"a": 5}, True),
            (6, {"a": 6}, False),  # the resumed job's logs at 6 merge into it
        ]

    def test_restore_relogs_nothing_wandb_received(self, tmp_path):
        with _mock_wandb_module():
            previous = _init_resumable(str(tmp_path))
            for step in range(2, 8):
                previous.log({"a": step}, step=step)
        with _mock_wandb_module(next_step=7) as module:
            wandb = _init_resumable(str(tmp_path))
            wandb.restore_unsent(resume_step=6)
        module.log.assert_not_called()

    @pytest.mark.parametrize("offline, log_to_wandb", [(True, True), (False, False)])
    def test_restore_is_noop_offline_or_disabled(
        self, tmp_path, offline: bool, log_to_wandb: bool
    ):
        with _mock_wandb_module():
            previous = _init_resumable(str(tmp_path))
            previous.log({"a": 2}, step=2)
        with _mock_wandb_module(offline=offline) as module:
            wandb = _init_resumable(str(tmp_path), log_to_wandb=log_to_wandb)
            wandb.restore_unsent(resume_step=6)
        module.log.assert_not_called()

    def test_log_before_restore_discards_previous_rows(self, tmp_path):
        with _mock_wandb_module():
            previous = _init_resumable(str(tmp_path))
            previous.log({"a": 2}, step=2)
        with _mock_wandb_module() as module:
            wandb = _init_resumable(str(tmp_path))
            wandb.log({"b": 0}, step=0)
            wandb.restore_unsent(resume_step=6)
            assert _read_lines(str(tmp_path)) == [{"step": 0, "logs": {"b": 0}}]
        assert _logged(module) == [(0, {"b": 0}, None)]
