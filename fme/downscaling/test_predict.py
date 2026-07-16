import os

import pytest
import torch
import yaml

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.testing.wandb import mock_wandb
from fme.downscaling import predict
from fme.downscaling.data import (
    ClosedInterval,
    DataLoaderConfig,
    StaticInputs,
    coords_require_lon_roll,
    load_static_inputs,
)
from fme.downscaling.data.test_config import global_data_paths_helper
from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.predictors import PatchPredictionConfig
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.test_models import LinearDownscaling
from fme.downscaling.test_utils import cell_centered_coordinate, data_paths_helper


class LinearDownscalingDiffusion(LinearDownscaling):
    def forward(self, latent, coarse, noise_level):  # type: ignore
        return super().forward(coarse)


def get_model_config(
    coarse_shape: tuple[int, int],
    downscale_factor: int,
    use_fine_topography: bool = True,
):
    fine_shape = (
        coarse_shape[0] * downscale_factor,
        coarse_shape[1] * downscale_factor,
    )
    return DiffusionModelConfig(
        DiffusionModuleRegistrySelector(
            "prebuilt",
            {
                "module": LinearDownscalingDiffusion(
                    factor=1,  # will pass coarse input interpolated to fine shape
                    fine_img_shape=fine_shape,
                    n_channels_in=3 if use_fine_topography else 2,
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
        use_fine_topography=use_fine_topography,
    )


def create_predictor_config(
    tmp_path,
    n_samples: int,
    model_renaming: dict | None = None,
    data_paths_helper_kwargs: dict = {},
):
    # create toy dataset
    path = tmp_path / "data"
    path.mkdir()
    paths = data_paths_helper(path, **data_paths_helper_kwargs)

    # create config
    this_file = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{this_file}/configs/test_predictor_config.yaml"
    experiment_dir = tmp_path / "output"
    experiment_dir.mkdir()
    with open(file_path) as file:
        config = yaml.safe_load(file)
    config["data"]["coarse"] = [{"data_path": str(paths.coarse)}]
    config["data"]["lat_extent"] = {"start": 1, "stop": 6}
    config["experiment_dir"] = str(experiment_dir)
    config["model"] = {
        "checkpoint_path": f"{str(experiment_dir)}/checkpoints/latest.ckpt"
    }
    config["n_samples"] = n_samples
    config["events"][0]["name"] = "test_event"
    config["events"][0]["save_generated_samples"] = True
    if model_renaming is not None:
        config["model"]["rename"] = model_renaming

    out_path = tmp_path / "predictor-config.yaml"
    with open(out_path, "w") as file:
        yaml.dump(config, file)
    return out_path, f"{paths.fine}/data.nc"


@pytest.mark.medium_duration
def test_predictor_runs(tmp_path):
    n_samples = 2
    coarse_shape = (4, 4)
    downscale_factor = 2
    predictor_config_path, fine_data_path = create_predictor_config(
        tmp_path,
        n_samples,
    )
    model_config = get_model_config(coarse_shape, downscale_factor=downscale_factor)
    static_inputs = load_static_inputs({"HGTsfc": fine_data_path})
    model = model_config.build(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        full_fine_coords=static_inputs.coords,
        static_inputs=static_inputs,
    )
    with open(predictor_config_path) as f:
        predictor_config = yaml.safe_load(f)
    os.makedirs(
        os.path.join(predictor_config["experiment_dir"], "checkpoints"), exist_ok=True
    )

    torch.save(
        {
            "model": model.get_state(),
        },
        predictor_config["model"]["checkpoint_path"],
    )

    with mock_wandb():
        predict.main(str(predictor_config_path))
    assert os.path.exists(
        f"{predictor_config['experiment_dir']}/generated_maps_and_metrics.nc"
    )
    assert os.path.exists(f"{predictor_config['experiment_dir']}/test_event.nc")


def _build_seam_crossing_model_and_data(tmp_path):
    """A real model on a global fine grid plus a seam-crossing GriddedData.

    The lon extent straddles the 0/360 seam and is sized to match the model's
    coarse patch, so no patching is needed. No HGTsfc field, so StaticInputs is
    empty and the model runs with use_fine_topography=False.
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
    model = get_model_config(
        coarse_shape, downscale_factor, use_fine_topography=False
    ).build(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        full_fine_coords=full_fine_coords,
        static_inputs=StaticInputs(fields=[], coords=full_fine_coords),
    )
    data = DataLoaderConfig(
        coarse=[XarrayDataConfig(str(paths.coarse))],
        batch_size=2,
        num_data_workers=0,
        strict_ensemble=False,
        lat_extent=ClosedInterval(0.0, 8.0),
        lon_extent=ClosedInterval(-90.0, 90.0),
    ).build(
        requirements=DataRequirements(
            fine_names=[], coarse_names=["var0", "var1"], n_timesteps=1
        ),
    )
    return model, data, fine_shape


@pytest.mark.parametrize("cls", [predict.Downscaler, predict.EventDownscaler])
def test_predictor_runs_seam_crossing(tmp_path, cls):
    """Confirm the predict entrypoints roll the model end-to-end on real data (no
    mocks): the real coarse coords cross the prime meridian, with_rolled_lon
    produces a rolled model, and generation runs clean in the rolled convention.
    """
    model, data, fine_shape = _build_seam_crossing_model_and_data(tmp_path)
    kwargs = dict(
        data=data,
        model=model,
        experiment_dir=str(tmp_path / "output"),
        n_samples=2,
        patch=PatchPredictionConfig(),  # region matches model, so no patching
    )
    if cls is predict.EventDownscaler:
        downscaler = cls(event_name="evt", **kwargs)
    else:
        downscaler = cls(**kwargs)

    # The real coarse coords cross the seam, so the entrypoint rolls the model.
    assert coords_require_lon_roll(data.coarse_extent_latlon_coords.lon)
    generation_model = downscaler._get_generation_model()
    assert generation_model is not model  # a real roll produced a new model
    assert float(generation_model.full_fine_coords.lon.min()) < 0.0

    # Generation runs clean on a real batch, in the rolled convention.
    batch = next(iter(data.get_generator()))
    fine_coords = generation_model.get_fine_coords_for_batch(batch)
    assert float(fine_coords.lon.min()) < 0.0
    outputs = generation_model.generate_on_batch_no_target(batch, n_samples=2)
    for value in outputs.values():
        assert value.shape[-2:] == fine_shape


@pytest.mark.medium_duration
def test_predictor_renaming(
    tmp_path,
):
    n_samples = 2
    coarse_shape = (4, 4)
    downscale_factor = 2
    renaming = {"var0": "var0_renamed", "var1": "var1_renamed"}
    predictor_config_path, fine_data_path = create_predictor_config(
        tmp_path,
        n_samples,
        model_renaming=renaming,
        data_paths_helper_kwargs={
            "rename": {"var0": "var0_renamed", "var1": "var1_renamed"}
        },
    )
    model_config = get_model_config(
        coarse_shape, downscale_factor, use_fine_topography=False
    )
    static_inputs = load_static_inputs({"HGTsfc": fine_data_path})
    model = model_config.build(
        coarse_shape=coarse_shape,
        downscale_factor=2,
        full_fine_coords=static_inputs.coords,
        static_inputs=static_inputs,
    )
    with open(predictor_config_path) as f:
        predictor_config = yaml.safe_load(f)
    os.makedirs(
        os.path.join(predictor_config["experiment_dir"], "checkpoints"), exist_ok=True
    )

    torch.save(
        {
            "model": model.get_state(),
        },
        predictor_config["model"]["checkpoint_path"],
    )
    with mock_wandb():
        predict.main(str(predictor_config_path))
