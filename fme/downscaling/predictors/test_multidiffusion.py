import pytest
import torch

from fme.core.device import get_device
from fme.downscaling.data import BatchData
from fme.downscaling.data.patching import HorizontalSlice, PatchAverager, get_patches
from fme.downscaling.models import DiffusionModel
from fme.downscaling.predictors.multidiffusion import (
    MultiDiffusionDenoiser,
    generate_multidiffusion,
)
from fme.downscaling.predictors.serial_denoising import DenoisingMoEPredictor
from fme.downscaling.test_models import (
    _get_diffusion_model,
    make_batch_data,
    make_static_inputs,
)
from fme.downscaling.test_utils import cell_centered_coordinate

DOWNSCALE_FACTOR = 2
MODEL_COARSE_SHAPE = (8, 8)


def _slice(y0, y1, x0, x1) -> HorizontalSlice:
    return HorizontalSlice(y=slice(y0, y1), x=slice(x0, x1))


def _denoiser(net, slices, inputs, shifts, shape) -> MultiDiffusionDenoiser:
    average = PatchAverager(slices, shape, torch.device("cpu"))
    return MultiDiffusionDenoiser(net, average, inputs, shifts)


def test_MultiDiffusionDenoiser_averages_overlapping_patches():
    # two 1x3 patches on a 1x4 domain, overlapping in columns 1-2
    slices = [_slice(0, 1, 0, 3), _slice(0, 1, 1, 4)]
    patch_values = [1.0, 3.0]
    inputs = [torch.full((2, 1, 1, 3), v) for v in patch_values]

    def net(x, x_lr, sigma):
        return x_lr.expand_as(x)

    denoiser = _denoiser(net, slices, inputs, [None, None], (1, 4))
    x = torch.zeros(2, 1, 1, 4)
    denoised = denoiser(x, x.new_empty(2, 0), torch.tensor(1.0))
    expected = torch.tensor([1.0, 2.0, 2.0, 3.0]).expand(2, 1, 1, 4)
    torch.testing.assert_close(denoised, expected)


def _identity_net(x, x_lr, sigma):
    return x


def test_MultiDiffusionDenoiser_base_shift_round_trips():
    # A denoiser returning its input must return the full-domain state
    # unchanged, whatever the per-patch base shifts are.
    slices = [_slice(0, 3, 0, 4), _slice(1, 4, 0, 4)]
    inputs = [torch.zeros(2, 1, 3, 4)] * 2
    shifts: list[torch.Tensor | None] = [torch.randn(2, 1, 3, 4) for _ in slices]
    denoiser = _denoiser(_identity_net, slices, inputs, shifts, (4, 4))
    x = torch.randn(2, 1, 4, 4)
    denoised = denoiser(x, x.new_empty(2, 0), torch.tensor(1.0))
    torch.testing.assert_close(denoised, x)


def test_MultiDiffusionDenoiser_raises_on_state_shape_mismatch():
    slices = [_slice(0, 3, 0, 4), _slice(1, 4, 0, 4)]
    inputs = [torch.zeros(2, 1, 3, 4)] * 2
    denoiser = _denoiser(_identity_net, slices, inputs, [None, None], (4, 4))
    x = torch.randn(2, 1, 4, 5)
    with pytest.raises(ValueError, match="does not match"):
        denoiser(x, x.new_empty(2, 0), torch.tensor(1.0))


def _coarse_batch(coarse_shape: tuple[int, int], batch_size: int = 2) -> BatchData:
    n_lat, n_lon = coarse_shape
    batch = make_batch_data(
        (batch_size, n_lat, n_lon),
        cell_centered_coordinate(0.0, n_lat * DOWNSCALE_FACTOR, n=n_lat),
        cell_centered_coordinate(0.0, n_lon * DOWNSCALE_FACTOR, n=n_lon),
    )
    # vary the data spatially so patch and full-domain interpolation differ
    return BatchData(
        data={"x": torch.randn(batch_size, n_lat, n_lon, device=get_device())},
        time=batch.time,
        latlon_coordinates=batch.latlon_coordinates,
    )


def _model(
    full_coarse_shape: tuple[int, int], predict_residual: bool = True
) -> DiffusionModel:
    static_inputs = make_static_inputs(
        (
            full_coarse_shape[0] * DOWNSCALE_FACTOR,
            full_coarse_shape[1] * DOWNSCALE_FACTOR,
        )
    )
    return _get_diffusion_model(
        coarse_shape=MODEL_COARSE_SHAPE,
        downscale_factor=DOWNSCALE_FACTOR,
        full_fine_coords=static_inputs.coords,
        predict_residual=predict_residual,
        use_fine_topography=True,
        static_inputs=static_inputs,
    )


def _moe(full_coarse_shape: tuple[int, int]) -> DenoisingMoEPredictor:
    return DenoisingMoEPredictor(
        experts=[_model(full_coarse_shape) for _ in range(2)],
        sigma_ranges=[(0.1, 0.5), (0.5, 1.0)],
        num_diffusion_generation_steps=4,
        churn=0.25,
    )


@pytest.mark.parametrize("predict_residual", [True, False])
def test_generate_multidiffusion_single_patch_matches_generate(predict_residual):
    model = _model(MODEL_COARSE_SHAPE, predict_residual=predict_residual)
    batch = _coarse_batch(MODEL_COARSE_SHAPE)
    patches = get_patches(MODEL_COARSE_SHAPE, MODEL_COARSE_SHAPE, overlap=0)
    assert len(patches) == 1

    torch.manual_seed(0)
    expected = model.generate_on_batch_no_target(batch, n_samples=2)
    torch.manual_seed(0)
    generated, _ = generate_multidiffusion(model, batch, patches, n_samples=2)
    torch.testing.assert_close(generated, expected)


@pytest.mark.parametrize("model_type", ["diffusion", "moe"])
@pytest.mark.parametrize(
    "full_coarse_shape",
    [
        pytest.param((12, 12), id="no_partial_patch"),
        pytest.param((11, 13), id="partial_patch"),
    ],
)
def test_generate_multidiffusion_multiple_patches(model_type, full_coarse_shape):
    model: DiffusionModel | DenoisingMoEPredictor
    if model_type == "diffusion":
        model = _model(full_coarse_shape)
    else:
        model = _moe(full_coarse_shape)
    batch = _coarse_batch(full_coarse_shape, batch_size=2)
    patches = get_patches(
        full_coarse_shape, MODEL_COARSE_SHAPE, overlap=4, drop_partial_patches=False
    )
    assert len(patches) > 1
    generated, generated_norm = generate_multidiffusion(
        model, batch, patches, n_samples=3
    )
    fine_shape = tuple(n * DOWNSCALE_FACTOR for n in full_coarse_shape)
    assert generated["x"].shape == (2, 3, *fine_shape)
    assert generated_norm.shape == (2 * 3, 1, *fine_shape)
    assert torch.isfinite(generated["x"]).all()
