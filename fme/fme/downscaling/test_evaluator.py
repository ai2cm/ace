import os
import unittest.mock
from typing import Any, Mapping

import pytest
import yaml

from fme.ace.train_config import LoggingConfig
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import OptimizationConfig
from fme.core.testing.wandb import mock_wandb
from fme.downscaling import evaluator
from fme.downscaling.datasets import DataLoaderConfig
from fme.downscaling.models import DownscalingModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.test_models import LinearDownscaling
from fme.downscaling.test_train import data_paths_helper
from fme.downscaling.train import TrainerConfig, save_checkpoint


def create_evaluator_config(tmp_path, model: Mapping[str, Any]):
    # create toy dataset
    path = tmp_path / "evaluation"
    path.mkdir()
    paths = data_paths_helper(path)

    # create config
    this_file = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{this_file}/configs/test_evaluator_config.yaml"
    experiment_dir = tmp_path / "output"
    experiment_dir.mkdir()
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    config["data"]["path_fine"] = str(paths.fine)
    config["data"]["path_coarse"] = str(paths.coarse)
    config["experiment_dir"] = str(experiment_dir)
    config["model"] = model

    out_path = tmp_path / "evaluator-config.yaml"
    with open(out_path, "w") as file:
        yaml.dump(config, file)
    return out_path


@pytest.mark.parametrize(
    "model_config",
    [
        pytest.param(
            {
                "mode": "nearest",
                "in_names": ["x", "y"],
                "out_names": ["x", "y"],
                "downscale_factor": 2,
            },
            id="interpolation",
        ),
        pytest.param({"checkpoint": "checkpoint.ckpt"}, id="checkpoint"),
    ],
)
def test_evaluator_runs(tmp_path, model_config):
    """Check that evaluator runs with different models."""
    evaluator_config_path = create_evaluator_config(tmp_path, model_config)

    paths = data_paths_helper(tmp_path)

    if "checkpoint" in model_config:
        trainer = TrainerConfig(
            DownscalingModelConfig(
                ModuleRegistrySelector(
                    "prebuilt", {"module": LinearDownscaling(2, (32, 32), n_channels=2)}
                ),
                LossConfig("NaN"),
                ["x", "y"],
                ["x", "y"],
                PairedNormalizationConfig(
                    NormalizationConfig(
                        means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}
                    ),
                    NormalizationConfig(
                        means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}
                    ),
                ),
                use_fine_topography=False,
            ),
            OptimizationConfig(),
            DataLoaderConfig(paths.fine, paths.coarse, "xarray", 2, 1),
            DataLoaderConfig(paths.fine, paths.coarse, "xarray", 2, 1),
            1,
            "/experiment_dir",
            False,
            LoggingConfig(),
            None,
        ).build()

        checkpoint_path = tmp_path / model_config["checkpoint"]
        save_checkpoint(trainer, checkpoint_path)

        with open(evaluator_config_path, "r") as file:
            config = yaml.safe_load(file)
        with open(evaluator_config_path, "w") as file:
            config["model"] = {"checkpoint": str(checkpoint_path)}
            yaml.dump(config, file)

    with unittest.mock.patch(
        "fme.downscaling.aggregators.Aggregator.get_wandb"
    ) as mock_get_wandb:
        with mock_wandb():
            evaluator.main(str(evaluator_config_path))
    mock_get_wandb.assert_called()

    assert os.path.isfile(tmp_path / "output/histogram_diagnostics.nc")
    assert os.path.isfile(tmp_path / "output/time_mean_map_diagnostics.nc")
