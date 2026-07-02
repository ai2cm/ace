import os
import unittest.mock
from collections.abc import Mapping
from typing import Any

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
from fme.downscaling.data import ClosedInterval, PairedDataLoaderConfig, StaticInputs
from fme.downscaling.data.test_config import global_data_paths_helper
from fme.downscaling.models import (
    CheckpointModelConfig,
    DiffusionModelConfig,
    PairedNormalizationConfig,
)
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.test_models import LinearDownscaling
from fme.downscaling.test_utils import cell_centered_coordinate, data_paths_helper
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


def _seam_crossing_model_config(fine_shape: tuple[int, int]) -> DiffusionModelConfig:
    """A (4, 4)-patch model config whose module fits the seam-crossing fine shape."""
    return DiffusionModelConfig(
        DiffusionModuleRegistrySelector(
            "prebuilt",
            {
                "module": LinearDownscalingDiffusion(
                    factor=1,
                    fine_img_shape=fine_shape,
                    n_channels_in=2,
                    n_channels_out=2,
                )
            },
            expects_interpolated_input=True,
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
        use_fine_topography=False,
    )


def test_evaluator_rolls_for_seam_crossing(tmp_path):
    """The evaluator entrypoint rolls the model end-to-end on real data (no mocks):
    a seam-crossing PairedGriddedData yields coarse coords that cross the prime
    meridian, and _build_default_evaluator returns a model rolled to that
    convention. Global coarse grid is 4 lat x 8 lon (45 deg spacing); a (-90, 90)
    lon extent selects 4 of 8 coarse cells across the seam, matching the (4, 4)
    model so no patching is needed.
    """
    coarse_shape = (4, 4)
    downscale_factor = 2
    fine_shape = (
        coarse_shape[0] * downscale_factor,
        coarse_shape[1] * downscale_factor,
    )
    paths = global_data_paths_helper(tmp_path)

    full_fine_coords = LatLonCoordinates(
        lat=cell_centered_coordinate(0.0, 8.0, fine_shape[0]),
        lon=cell_centered_coordinate(0.0, 360.0, fine_shape[1]),
    )
    model = _seam_crossing_model_config(fine_shape).build(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        full_fine_coords=full_fine_coords,
        static_inputs=StaticInputs(fields=[], coords=full_fine_coords),
    )
    experiment_dir = tmp_path / "output"
    experiment_dir.mkdir()
    checkpoint_path = experiment_dir / "latest.ckpt"
    torch.save({"model": model.get_state()}, checkpoint_path)

    config = evaluator.EvaluatorConfig(
        model=CheckpointModelConfig(checkpoint_path=str(checkpoint_path)),
        experiment_dir=str(experiment_dir),
        data=PairedDataLoaderConfig(
            fine=[XarrayDataConfig(paths.fine)],
            coarse=[XarrayDataConfig(paths.coarse)],
            batch_size=2,
            num_data_workers=0,
            strict_ensemble=False,
            lat_extent=ClosedInterval(0.0, 8.0),
            lon_extent=ClosedInterval(-90.0, 90.0),
        ),
        logging=LoggingConfig(),
    )

    built = config._build_default_evaluator()

    # A real roll moved the model's fine grid into the seam-crossing convention.
    assert float(built.model.full_fine_coords.lon.min()) < 0.0


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
@pytest.mark.medium_duration
def test_evaluator_runs(
    tmp_path,
    evaluator_model_config,
    num_samples,
):
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
