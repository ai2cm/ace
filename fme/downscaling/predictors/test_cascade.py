import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.downscaling.data import Topography
from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.predictors.cascade import CascadePredictor


def _latlon_coords_on_ngrid(n: int, edges=(0, 100)):
    start, end = edges
    dx = (end - start) / n
    midpoints = (start + (torch.arange(n) + 0.5) * dx).to(device=get_device())
    return LatLonCoordinates(lat=midpoints, lon=midpoints)


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
        use_fine_topography=True,
    ).build(coarse_shape=coarse_shape, downscale_factor=downscale_factor)


@pytest.mark.parametrize("downscale_factors", [[2, 4], [2, 3, 4]])
def test_CascadePredictor_generate(downscale_factors):
    n_times, n_samples_generate, nside_coarse = 3, 2, 4
    grid_bounds = (0, 100)
    models = []
    topographies: list[Topography | None] = []
    input_n_cells = nside_coarse

    for downscale_factor in downscale_factors:
        models.append(
            _get_diffusion_model(
                coarse_shape=(input_n_cells, input_n_cells),
                downscale_factor=downscale_factor,
            )
        )
        topographies.append(
            Topography(
                data=torch.randn(
                    input_n_cells * downscale_factor,
                    input_n_cells * downscale_factor,
                    device=get_device(),
                ),
                coords=_latlon_coords_on_ngrid(
                    n=input_n_cells * downscale_factor, edges=grid_bounds
                ),
            )
        )
        input_n_cells *= downscale_factor

    cascade_predictor = CascadePredictor(models=models, topographies=topographies)
    coarse_input = {
        "x": torch.randn(
            (n_times, nside_coarse, nside_coarse),
            device=get_device(),
            dtype=torch.float32,
        )
    }
    generated, _, _ = cascade_predictor.generate(
        coarse=coarse_input, n_samples=n_samples_generate, topographies=topographies
    )
    expected_nside = cascade_predictor.downscale_factor * nside_coarse
    assert generated["x"].shape == (
        n_times,
        n_samples_generate,
        expected_nside,
        expected_nside,
    )


def test_CascadePredictor__subset_topographies():
    nside_coarse = 8
    downscale_factors = [2, 2]
    grid_bounds = (0, 8)
    models = []
    topographies: list[Topography | None] = []
    input_n_cells = nside_coarse

    for downscale_factor in downscale_factors:
        models.append(
            _get_diffusion_model(
                coarse_shape=(input_n_cells, input_n_cells),
                downscale_factor=downscale_factor,
            )
        )
        topographies.append(
            Topography(
                data=torch.randn(
                    input_n_cells * downscale_factor,
                    input_n_cells * downscale_factor,
                    device=get_device(),
                ),
                coords=_latlon_coords_on_ngrid(
                    n=input_n_cells * downscale_factor, edges=grid_bounds
                ),
            )
        )
        input_n_cells *= downscale_factor

    cascade_predictor = CascadePredictor(models=models, topographies=topographies)
    # Coarse grid subset has 1.0 grid spacing and midpoints 1.5 ... 4.5
    coarse_coords = _latlon_coords_on_ngrid(n=4, edges=(1, 5))
    subset_intermediate_topographies = cascade_predictor._get_subset_topographies(
        coarse_coords=coarse_coords
    )

    # First topography grid 0.5 grid spacing
    assert isinstance(subset_intermediate_topographies[0], Topography)
    assert subset_intermediate_topographies[0].shape == (8, 8)
    assert subset_intermediate_topographies[0].coords.lat[0] == 1.25
    assert subset_intermediate_topographies[0].coords.lat[-1] == 4.75
    assert subset_intermediate_topographies[0].coords.lon[0] == 1.25
    assert subset_intermediate_topographies[0].coords.lon[-1] == 4.75
    # Second topography grid has 0.25 grid spacing
    assert isinstance(subset_intermediate_topographies[1], Topography)
    assert subset_intermediate_topographies[1].shape == (16, 16)
    assert subset_intermediate_topographies[1].coords.lat[0] == 1.125
    assert subset_intermediate_topographies[1].coords.lat[-1] == 4.875
    assert subset_intermediate_topographies[1].coords.lon[0] == 1.125
    assert subset_intermediate_topographies[1].coords.lon[-1] == 4.875
