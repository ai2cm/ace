import os

import torch
import yaml

from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.testing.wandb import mock_wandb
from fme.downscaling import predict
from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.test_models import LinearDownscaling
from fme.downscaling.test_train import data_paths_helper


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
                    factor=downscale_factor,
                    fine_img_shape=fine_shape,
                    n_channels=2,
                )
            },
            expects_interpolated_input=False,
        ),
        loss=LossConfig("NaN"),
        in_names=["x", "y"],
        out_names=["x", "y"],
        normalization=PairedNormalizationConfig(
            NormalizationConfig(means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}),
            NormalizationConfig(means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}),
        ),
        p_mean=0,
        p_std=1,
        sigma_min=1,
        sigma_max=2,
        churn=1,
        num_diffusion_generation_steps=2,
        predict_residual=True,
    )


def create_predictor_config(tmp_path, n_samples: int):
    # create toy dataset
    path = tmp_path / "data"
    path.mkdir()
    paths = data_paths_helper(path)

    # create config
    this_file = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{this_file}/configs/test_predictor_config.yaml"
    experiment_dir = tmp_path / "output"
    experiment_dir.mkdir()
    with open(file_path) as file:
        config = yaml.safe_load(file)
    config["data"]["topography"] = {"data_path": str(paths.fine)}
    config["data"]["coarse"] = [{"data_path": str(paths.coarse)}]
    config["experiment_dir"] = str(experiment_dir)
    config["model"] = {
        "checkpoint_path": f"{str(experiment_dir)}/checkpoints/latest.ckpt"
    }
    config["n_samples"] = n_samples

    out_path = tmp_path / "predictor-config.yaml"
    with open(out_path, "w") as file:
        yaml.dump(config, file)
    return out_path


def test_predictor_runs(
    tmp_path,
):
    n_samples = 2
    coarse_shape = (4, 4)
    downscale_factor = 2
    predictor_config_path = create_predictor_config(tmp_path, n_samples)
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
    assert os.path.exists(
        f"{predictor_config['experiment_dir']}/generated_maps_and_metrics.nc"
    )
