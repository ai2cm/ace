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
    """A model config whose module fits the given seam-crossing fine shape."""
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
    """The evaluator entrypoint rolls the model and generates end-to-end on real
    data (no mocks) for a seam-crossing domain. The lon extent straddles the 0/360
    seam and is sized to match the model's coarse patch, so no patching is needed.

    The generated output lands on the model's rolled fine grid, while the target
    is rolled at the data layer via an independent path. We assert those two grids
    are identical -- not merely that both crossed the seam -- then run generation
    and aggregation so the predictions and targets are actually produced and
    compared on that shared grid.
    """
    coarse_shape = (4, 4)
    downscale_factor = 2
    fine_shape = (
        coarse_shape[0] * downscale_factor,
        coarse_shape[1] * downscale_factor,
    )
    paths = global_data_paths_helper(tmp_path)

    # full_fine_coords spans the entire global fine domain, i.e. the coarse grid
    # scaled by downscale_factor, so the fine coordinates the model derives for a
    # batch are at the same resolution as the fine target data and can be compared
    # directly.
    full_fine_coords = LatLonCoordinates(
        lat=cell_centered_coordinate(0.0, 8.0, coarse_shape[0] * downscale_factor),
        lon=cell_centered_coordinate(0.0, 360.0, 8 * downscale_factor),
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
        n_samples=2,
    )

    built = config._build_default_evaluator()

    # A real roll moved the model's fine grid into the seam-crossing convention.
    assert float(built.model.full_fine_coords.lon.min()) < 0.0

    # The generated output uses the model's rolled fine grid; the target is rolled
    # independently at the data layer. Assert the per-batch fine coordinates the
    # model derives exactly equal the target's fine coordinates, so predictions and
    # targets are compared on the same grid rather than mismatched conventions.
    batch = next(iter(built.data.get_generator()))
    target_fine_lon = batch.fine.latlon_coordinates.lon[0]
    assert float(target_fine_lon.min()) < 0.0
    assert torch.allclose(
        built.model.get_fine_coords_for_batch(batch.coarse).lon, target_fine_lon
    )

    # Run the full generation + aggregation entrypoint on the crossing domain to
    # confirm predictions and targets are produced and compared without error.
    with unittest.mock.patch(
        "fme.downscaling.aggregators.GenerationAggregator.get_wandb"
    ):
        with mock_wandb():
            built.run()
    assert (experiment_dir / "evaluator_maps_and_metrics.nc").exists()


def test_evaluator_raises_when_larger_without_patching(tmp_path):
    """The default evaluator refuses a region larger than the model patch when
    patch prediction is not configured.

    The model and data are mocked so the shape check in ``_build_default_evaluator``
    is exercised without building a real model or dataset. The lower-level check is
    covered directly in test_composite.py; this verifies the evaluator wiring.
    """
    model_config = unittest.mock.MagicMock()
    built_model = model_config.build.return_value
    built_model.coarse_shape = (4, 4)
    built_model.with_rolled_lon.return_value = built_model  # same model, no roll

    data_config = unittest.mock.MagicMock()
    dataset = data_config.build.return_value
    dataset.coarse_shape = (4, 8)

    config = evaluator.EvaluatorConfig(
        model=model_config,
        experiment_dir=str(tmp_path),
        data=data_config,
        logging=LoggingConfig(),
        n_samples=2,
    )
    with pytest.raises(ValueError, match="requires patch prediction"):
        config._build_default_evaluator()


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
