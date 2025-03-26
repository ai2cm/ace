import os
import unittest.mock
from typing import Any, Mapping

import pytest
import yaml

from fme.core.dataset.config import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import OptimizationConfig
from fme.core.testing.wandb import mock_wandb
from fme.downscaling import evaluator
from fme.downscaling.datasets import DataLoaderConfig
from fme.downscaling.models import (
    DiffusionModelConfig,
    DownscalingModelConfig,
    PairedNormalizationConfig,
)
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.test_models import LinearDownscaling
from fme.downscaling.test_train import data_paths_helper
from fme.downscaling.train import TrainerConfig


def create_evaluator_config(tmp_path, model: Mapping[str, Any], n_samples: int):
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
    config["data"]["fine"] = [{"data_path": str(paths.fine)}]
    config["data"]["coarse"] = [{"data_path": str(paths.coarse)}]
    config["experiment_dir"] = str(experiment_dir)
    config["model"] = model
    config["n_samples"] = n_samples

    out_path = tmp_path / "evaluator-config.yaml"
    with open(out_path, "w") as file:
        yaml.dump(config, file)
    return out_path


class LinearDownscalingDiffusion(LinearDownscaling):
    def forward(self, latent, coarse, noise_level):  # type: ignore
        return super().forward(coarse)


def get_trainer_model_config(model_type: str):
    if model_type == "deterministic":
        return DownscalingModelConfig(
            ModuleRegistrySelector(
                "prebuilt",
                {"module": LinearDownscaling(2, fine_img_shape=(16, 16), n_channels=2)},
                expects_interpolated_input=False,
            ),
            loss=LossConfig("NaN"),
            in_names=["x", "y"],
            out_names=["x", "y"],
            normalization=PairedNormalizationConfig(
                NormalizationConfig(
                    means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}
                ),
                NormalizationConfig(
                    means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}
                ),
            ),
        )
    elif model_type == "diffusion":
        return DiffusionModelConfig(
            DiffusionModuleRegistrySelector(
                "prebuilt",
                {
                    "module": LinearDownscalingDiffusion(
                        factor=2,
                        fine_img_shape=(16, 16),
                        n_channels=2,
                    )
                },
                expects_interpolated_input=False,
            ),
            loss=LossConfig("NaN"),
            in_names=["x", "y"],
            out_names=["x", "y"],
            normalization=PairedNormalizationConfig(
                NormalizationConfig(
                    means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}
                ),
                NormalizationConfig(
                    means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}
                ),
            ),
            p_mean=0,
            p_std=1,
            sigma_min=1,
            sigma_max=2,
            churn=1,
            num_diffusion_generation_steps=1,
            predict_residual=True,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")


@pytest.mark.parametrize(
    "evaluator_model_config, model_type, num_samples",
    [
        pytest.param(
            {
                "mode": "nearest",
                "in_names": ["x", "y"],
                "out_names": ["x", "y"],
                "downscale_factor": 2,
            },
            "deterministic",
            1,
            id="interpolation",
        ),
        pytest.param(
            {"checkpoint": "unused value"},
            "deterministic",
            1,
            id="checkpoint_deterministic_model",
        ),
        pytest.param(
            {"checkpoint": "unused value"},
            "diffusion",
            1,
            id="checkpoint_diffusion_model_single_sample",
        ),
        pytest.param(
            {"checkpoint": "unused value"},
            "diffusion",
            2,
            id="checkpoint_diffusion_model_multiple_samples",
        ),
    ],
)
def test_evaluator_runs(tmp_path, evaluator_model_config, model_type, num_samples):
    """Check that evaluator runs with different models."""
    evaluator_config_path = create_evaluator_config(
        tmp_path, evaluator_model_config, num_samples
    )

    paths = data_paths_helper(tmp_path)

    trainer_model_config = get_trainer_model_config(model_type)

    if "checkpoint" in evaluator_model_config:
        with open(evaluator_config_path, "r") as file:
            config = yaml.safe_load(file)

        data_loader_config = DataLoaderConfig(
            fine=[XarrayDataConfig(paths.fine)],
            coarse=[XarrayDataConfig(paths.coarse)],
            batch_size=2,
            num_data_workers=0,
            strict_ensemble=False,
        )
        trainer = TrainerConfig(
            model=trainer_model_config,
            optimization=OptimizationConfig(),
            train_data=data_loader_config,
            validation_data=data_loader_config,
            max_epochs=1,
            experiment_dir=config["experiment_dir"],
            save_checkpoints=False,
            logging=LoggingConfig(),
        ).build()

        trainer.save_all_checkpoints(float("-inf"))

        with open(evaluator_config_path, "w") as file:
            config["model"] = {"checkpoint": trainer.epoch_checkpoint_path}
            yaml.dump(config, file)

    with unittest.mock.patch(
        "fme.downscaling.aggregators.Aggregator.get_wandb"
    ) as mock_get_wandb:
        with mock_wandb():
            evaluator.main(str(evaluator_config_path))
    mock_get_wandb.assert_called()

    assert os.path.isfile(tmp_path / "output/histogram_diagnostics.nc")
    assert os.path.isfile(tmp_path / "output/time_mean_map_diagnostics.nc")
