import os
import unittest.mock
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import OptimizationConfig
from fme.core.testing.wandb import mock_wandb
from fme.downscaling import evaluator
from fme.downscaling.data import PairedDataLoaderConfig
from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.test_models import LinearDownscaling
from fme.downscaling.test_utils import data_paths_helper
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
    with open(file_path) as file:
        config = yaml.safe_load(file)
    config["data"]["fine"] = [{"data_path": str(paths.fine)}]
    config["data"]["coarse"] = [{"data_path": str(paths.coarse)}]
    config["experiment_dir"] = str(experiment_dir)
    config["model"] = model
    config["n_samples"] = n_samples
    config["events"] = [
        {
            "name": "test_event",
            "date": "2000-01-01T00:00",
            "n_samples": n_samples,
        }
    ]

    out_path = tmp_path / "evaluator-config.yaml"
    with open(out_path, "w") as file:
        yaml.dump(config, file)
    return out_path


class LinearDownscalingDiffusion(LinearDownscaling):
    def forward(self, latent, coarse, noise_level):  # type: ignore
        return super().forward(coarse)


def get_trainer_model_config():
    return DiffusionModelConfig(
        DiffusionModuleRegistrySelector(
            "prebuilt",
            {
                "module": LinearDownscalingDiffusion(
                    factor=2,
                    fine_img_shape=(16, 16),
                    n_channels_in=2,
                )
            },
            expects_interpolated_input=False,
        ),
        loss=LossConfig("NaN"),
        in_names=["var0", "var1"],
        out_names=["var0", "var1"],
        normalization=PairedNormalizationConfig(
            NormalizationConfig(
                means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
            ),
            NormalizationConfig(
                means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
            ),
        ),
        p_mean=0,
        p_std=1,
        sigma_min=1,
        sigma_max=2,
        churn=1,
        num_diffusion_generation_steps=2,
        predict_residual=True,
    )


@pytest.mark.parametrize(
    "evaluator_model_config, num_samples",
    [
        pytest.param(
            {"checkpoint_path": "unused value"},
            1,
            id="checkpoint_diffusion_model_single_sample",
        ),
        pytest.param(
            {"checkpoint_path": "unused value"},
            2,
            id="checkpoint_diffusion_model_multiple_samples",
        ),
    ],
)
def test_evaluator_runs(
    tmp_path,
    evaluator_model_config,
    num_samples,
    very_fast_only: bool,
):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    """Check that evaluator runs with different models."""
    evaluator_config_path = create_evaluator_config(
        tmp_path, evaluator_model_config, num_samples
    )

    paths = data_paths_helper(tmp_path)

    trainer_model_config = get_trainer_model_config()

    if "checkpoint_path" in evaluator_model_config:
        with open(evaluator_config_path) as file:
            config = yaml.safe_load(file)

        data_loader_config = PairedDataLoaderConfig(
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

        trainer.save_epoch_checkpoints()

        with open(evaluator_config_path, "w") as file:
            config["model"] = {"checkpoint_path": trainer.epoch_checkpoint_path}
            yaml.dump(config, file)

    with unittest.mock.patch(
        "fme.downscaling.aggregators.GenerationAggregator.get_wandb"
    ) as mock_get_gen_agg_wandb:
        with mock_wandb():
            evaluator.main(str(evaluator_config_path))
    mock_get_gen_agg_wandb.assert_called()


def _mock_evaluator_config(coarse_lon, monkeypatch):
    """EvaluatorConfig over mocked model/data configs, with Evaluator and
    EventEvaluator patched to capture the model that gets handed to them. The
    mock model's with_rolled_lon returns a distinct sentinel so we can observe
    the seam-crossing roll branch in the build methods. Default (no-op) patching
    means the (possibly rolled) base model is passed through unwrapped.
    """
    rolled_model = MagicMock(name="rolled_model")
    rolled_model.coarse_shape = (4, 4)
    model = MagicMock(name="model")
    model.coarse_shape = (4, 4)
    model.with_rolled_lon.return_value = rolled_model

    model_config = MagicMock(name="model_config")
    model_config.build.return_value = model

    dataset = MagicMock(name="dataset")
    dataset.coarse_latlon_coords = LatLonCoordinates(
        lat=torch.tensor([0.0, 1.0]), lon=coarse_lon
    )
    dataset.coarse_shape = (4, 4)

    data_config = MagicMock(name="data_config")
    data_config.build.return_value = dataset

    config = evaluator.EvaluatorConfig(
        model=model_config,
        experiment_dir="unused",
        data=data_config,
        logging=MagicMock(),
    )

    captured: dict[str, Any] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(evaluator, "Evaluator", _capture)
    monkeypatch.setattr(evaluator, "EventEvaluator", _capture)
    return config, model, rolled_model, dataset, captured


@pytest.mark.parametrize("crossing", [True, False])
def test_build_default_evaluator_rolls_for_seam_crossing(monkeypatch, crossing):
    coarse_lon = (
        torch.tensor([-10.0, -5.0, 0.0, 5.0])
        if crossing
        else torch.tensor([0.0, 5.0, 10.0, 15.0])
    )
    config, model, rolled_model, _, captured = _mock_evaluator_config(
        coarse_lon, monkeypatch
    )

    config._build_default_evaluator()

    if crossing:
        model.with_rolled_lon.assert_called_once()
        assert torch.equal(model.with_rolled_lon.call_args[0][0], coarse_lon)
        assert captured["model"] is rolled_model
    else:
        model.with_rolled_lon.assert_not_called()
        assert captured["model"] is model


def test_build_event_evaluator_rolls_for_seam_crossing(monkeypatch):
    coarse_lon = torch.tensor([-10.0, -5.0, 0.0, 5.0])
    config, model, rolled_model, dataset, captured = _mock_evaluator_config(
        coarse_lon, monkeypatch
    )
    event_config = MagicMock()
    event_config.get_paired_gridded_data.return_value = dataset
    event_config.name = "evt"
    event_config.n_samples = 1
    event_config.save_generated_samples = False

    config._build_event_evaluator(event_config)

    model.with_rolled_lon.assert_called_once()
    assert torch.equal(model.with_rolled_lon.call_args[0][0], coarse_lon)
    assert captured["model"] is rolled_model
