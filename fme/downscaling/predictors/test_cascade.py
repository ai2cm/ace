import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.downscaling.data import StaticInputs, Topography
from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.predictors.cascade import CascadePredictor


def _latlon_coords_on_ngrid(n: int, edges=(0, 100)):
    start, end = edges
    dx = (end - start) / n
    midpoints = (start + (torch.arange(n) + 0.5) * dx).to(device=get_device())
    return LatLonCoordinates(lat=midpoints, lon=midpoints)


def _get_diffusion_model(coarse_shape, downscale_factor, static_inputs=None):
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
    ).build(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        static_inputs=static_inputs,
    )


@pytest.mark.parametrize("downscale_factors", [[2, 4], [2, 3, 4]])
def test_CascadePredictor_generate(downscale_factors):
    n_times, n_samples_generate, nside_coarse = 3, 2, 4
    grid_bounds = (0, 100)
    models = []
    topography_tensors: list[torch.Tensor | None] = []
    input_n_cells = nside_coarse

    for downscale_factor in downscale_factors:
        fine_n = input_n_cells * downscale_factor
        fine_coords = _latlon_coords_on_ngrid(n=fine_n, edges=grid_bounds)
        topo = Topography(
            data=torch.randn(fine_n, fine_n, device=get_device()),
            coords=fine_coords,
        )
        static_inputs = StaticInputs([topo])
        models.append(
            _get_diffusion_model(
                coarse_shape=(input_n_cells, input_n_cells),
                downscale_factor=downscale_factor,
                static_inputs=static_inputs,
            )
        )
        topography_tensors.append(topo.data)
        input_n_cells *= downscale_factor

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
        topographies=topography_tensors,
    )
    expected_nside = cascade_predictor.downscale_factor * nside_coarse
    assert generated["x"].shape == (
        n_times,
        n_samples_generate,
        expected_nside,
        expected_nside,
    )


def test_CascadePredictor_resolve_topographies():
    nside_coarse = 8
    downscale_factors = [2, 2]
    grid_bounds = (0, 8)
    models = []
    input_n_cells = nside_coarse

    for downscale_factor in downscale_factors:
        fine_n = input_n_cells * downscale_factor
        fine_coords = _latlon_coords_on_ngrid(n=fine_n, edges=grid_bounds)
        topo = Topography(
            data=torch.randn(fine_n, fine_n, device=get_device()),
            coords=fine_coords,
        )
        static_inputs = StaticInputs([topo])
        models.append(
            _get_diffusion_model(
                coarse_shape=(input_n_cells, input_n_cells),
                downscale_factor=downscale_factor,
                static_inputs=static_inputs,
            )
        )
        input_n_cells *= downscale_factor

    cascade_predictor = CascadePredictor(models=models)
    # Coarse grid subset has 1.0 grid spacing and midpoints 1.5 ... 4.5
    coarse_coords = _latlon_coords_on_ngrid(n=4, edges=(1, 5))
    resolved = cascade_predictor._resolve_topographies(coarse_coords=coarse_coords)

    # First topography: 0.5 grid spacing, 8 cells
    assert resolved[0] is not None
    assert resolved[0].shape == (8, 8)
    # Second topography: 0.25 grid spacing, 16 cells
    assert resolved[1] is not None
    assert resolved[1].shape == (16, 16)
