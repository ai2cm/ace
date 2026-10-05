import os

import dacite
import pytest
import yaml

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


@pytest.mark.parametrize(
    "metrics_log_dir", ["/absolute/metrics", "memory://bucket/metrics"]
)
def test_non_relative_metrics_log_dir_is_rejected(metrics_log_dir):
    with pytest.raises(ValueError, match="relative path"):
        LoggingConfig(metrics_log_dir=metrics_log_dir)


def test_null_metrics_log_dir_is_rejected():
    config = yaml.safe_load("metrics_log_dir: null")
    with pytest.raises(dacite.WrongTypeError, match="metrics_log_dir"):
        dacite.from_dict(LoggingConfig, config, config=dacite.Config(strict=True))


def test_no_metrics_written_for_non_local_experiment_dir():
    with mock_wandb() as wandb:
        LoggingConfig(log_to_wandb=False)._configure_wandb(
            experiment_dir="memory://bucket/exp", config={}, resumable=False
        )
        wandb.log({"loss": 1.0}, step=1)
        assert wandb._disk_logger is None
