import pytest
import torch

from fme.core.device import get_device
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.predictors.cascade import CascadePredictor


def _get_diffusion_model(coarse_shape, downscale_factor):
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )
    return DiffusionModelConfig(
        module=DiffusionModuleRegistrySelector(
            "unet_diffusion_song",
            {"model_channels": 4, "attn_resolutions": []},
        ),
        loss=LossConfig(type="MSE"),
        in_names=["x"],
        out_names=["x"],
        normalization=normalizer,
        p_mean=-1.0,
        p_std=1.0,
        sigma_min=0.1,
        sigma_max=1.0,
        churn=0.5,
        num_diffusion_generation_steps=3,
        predict_residual=True,
        use_fine_topography=False,
    ).build(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
    )


@pytest.mark.parametrize("downscale_factors", [[2, 4], [2, 3, 4]])
def test_CascadePredictor_generate(downscale_factors):
    n_times, n_samples_generate, nside_coarse = 3, 2, 4
    models = []
    input_n_cells = nside_coarse

    for downscale_factor in downscale_factors:
        input_n_cells *= downscale_factor
        models.append(
            _get_diffusion_model(
                coarse_shape=(input_n_cells, input_n_cells),
                downscale_factor=downscale_factor,
            )
        )
    cascade_predictor = CascadePredictor(models=models)
    coarse_input = {
        "x": torch.randn(
            (n_times, nside_coarse, nside_coarse),
            device=get_device(),
            dtype=torch.float32,
        )
    }
    generated, _, _ = cascade_predictor.generate(
        coarse=coarse_input,
        n_samples=n_samples_generate,
    )
    expected_nside = cascade_predictor.downscale_factor * nside_coarse
    assert generated["x"].shape == (
        n_times,
        n_samples_generate,
        expected_nside,
        expected_nside,
    )
