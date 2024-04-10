import os
import unittest.mock

import yaml

from fme.core.testing.wandb import mock_wandb
from fme.downscaling import evaluator
from fme.downscaling.test_train import data_paths_helper


def create_evaluator_config(tmp_path):
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
    config["model"]["mode"] = "nearest"
    config["model"]["in_names"] = ["x", "y"]
    config["model"]["out_names"] = ["x", "y"]
    config["model"]["downscale_factor"] = 2

    out_path = tmp_path / "evaluator-config.yaml"
    with open(out_path, "w") as file:
        yaml.dump(config, file)
    return out_path


def test_evaluator(tmp_path):
    """Integration test for model evaluation."""
    evaluator_config_path = create_evaluator_config(tmp_path)
    with unittest.mock.patch(
        "fme.downscaling.aggregators.Aggregator.get_wandb"
    ) as mock_get_wandb:
        with mock_wandb():
            evaluator.main(evaluator_config_path)
    mock_get_wandb.assert_called()

    assert os.path.isfile(tmp_path / "output/histogram_diagnostics.nc")
    assert os.path.isfile(tmp_path / "output/time_mean_map_diagnostics.nc")
