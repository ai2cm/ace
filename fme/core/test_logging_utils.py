import os

from fme.core.disk_metric_logger import read_metrics
from fme.core.logging_utils import LoggingConfig
from fme.core.testing.wandb import mock_wandb


def _configure_and_log(experiment_dir: str, config: LoggingConfig):
    with mock_wandb() as wandb:
        config._configure_wandb(
            experiment_dir=experiment_dir, config={}, resumable=True
        )
        wandb.log({"loss": 1.0}, step=1)


def test_metrics_written_under_experiment_dir_by_default(tmp_path):
    _configure_and_log(str(tmp_path), LoggingConfig(log_to_wandb=True))
    records = read_metrics(os.path.join(tmp_path, "metrics"))
    assert records == [{"step": 1, "loss": 1.0}]


def test_absolute_metrics_log_dir_is_used_as_is(tmp_path):
    experiment_dir = os.path.join(tmp_path, "experiment")
    metrics_log_dir = os.path.join(tmp_path, "elsewhere")
    os.makedirs(experiment_dir)
    _configure_and_log(
        experiment_dir,
        LoggingConfig(log_to_wandb=True, metrics_log_dir=metrics_log_dir),
    )
    assert read_metrics(metrics_log_dir) == [{"step": 1, "loss": 1.0}]


def test_no_metrics_written_when_metrics_log_dir_is_none(tmp_path):
    _configure_and_log(
        str(tmp_path), LoggingConfig(log_to_wandb=True, metrics_log_dir=None)
    )
    assert os.listdir(tmp_path) == ["wandb_run_id"]


def test_no_metrics_written_for_non_local_experiment_dir():
    with mock_wandb() as wandb:
        wandb.configure(
            log_to_wandb=False,
            metrics_log_dir=LoggingConfig()._get_metrics_log_dir("gs://bucket/exp"),
        )
        assert wandb._disk_logger is None
