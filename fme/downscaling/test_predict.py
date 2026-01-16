import os

import pytest
import torch
import xarray as xr
import yaml

from fme.core.coordinates import LatLonCoordinates
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.testing.wandb import mock_wandb
from fme.downscaling import predict
from fme.downscaling.data import StaticInputs, Topography
from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.test_models import LinearDownscaling
from fme.downscaling.test_utils import data_paths_helper


class LinearDownscalingDiffusion(LinearDownscaling):
    def forward(self, latent, coarse, noise_level):  # type: ignore
        return super().forward(coarse)


def get_model_config(coarse_shape: tuple[int, int], downscale_factor: int):
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
                    n_channels_in=3,
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
        use_fine_topography=True,
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
    config["data"]["topography"] = f"{paths.fine}/data.nc"
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
    return out_path


@pytest.mark.parametrize("static_inputs_on_model", [True, False])
def test_predictor_runs(static_inputs_on_model, tmp_path, very_fast_only: bool):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    n_samples = 2
    coarse_shape = (4, 4)
    downscale_factor = 2
    predictor_config_path = create_predictor_config(
        tmp_path,
        n_samples,
    )
    model_config = get_model_config(coarse_shape, downscale_factor=downscale_factor)
    model = model_config.build(
        coarse_shape=coarse_shape, downscale_factor=downscale_factor
    )
    with open(predictor_config_path) as f:
        predictor_config = yaml.safe_load(f)
    os.makedirs(
        os.path.join(predictor_config["experiment_dir"], "checkpoints"), exist_ok=True
    )

    if static_inputs_on_model:
        # ensure model static inputs shape is consistent with the test data
        fine_data = xr.load_dataset(predictor_config["data"]["topography"])
        topo_data = fine_data["HGTsfc"]
        model.static_inputs = StaticInputs(
            [
                Topography(
                    data=torch.randn(topo_data.shape[-2:]),
                    coords=LatLonCoordinates(
                        lat=torch.tensor(topo_data.lat.values),
                        lon=torch.tensor(topo_data.lon.values),
                    ),
                )
            ]
        )
        # overwrite dataset removing HGTsfc (fine data path is same as topography path)
        fine_data.drop_vars("HGTsfc").to_netcdf(
            predictor_config["data"]["topography"], mode="w"
        )
        # overwrite config to remove topography path
        predictor_config["data"]["topography"] = None
        with open(predictor_config_path, "w") as f:
            yaml.dump(predictor_config, f)

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


def test_predictor_renaming(
    tmp_path,
    very_fast_only: bool,
):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    n_samples = 2
    coarse_shape = (4, 4)
    downscale_factor = 2
    renaming = {"var0": "var0_renamed", "var1": "var1_renamed"}
    predictor_config_path = create_predictor_config(
        tmp_path,
        n_samples,
        model_renaming=renaming,
        data_paths_helper_kwargs={
            "rename": {"var0": "var0_renamed", "var1": "var1_renamed"}
        },
    )
    model_config = get_model_config(coarse_shape, downscale_factor)
    model = model_config.build(coarse_shape=coarse_shape, downscale_factor=2)
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
