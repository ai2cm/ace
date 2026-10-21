import pytest
import torch

from fme.core.device import get_device
from fme.downscaling.data import BatchData, PairedBatchData
from fme.downscaling.data.patching import PatchAverager, _HorizontalSlice, get_patches
from fme.downscaling.models import DiffusionModel
from fme.downscaling.predictors.composite import PatchPredictionConfig, PatchPredictor
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


def _slice(y0, y1, x0, x1) -> _HorizontalSlice:
    return _HorizontalSlice(y=slice(y0, y1), x=slice(x0, x1))


def _denoiser(net, slices, inputs, shifts, shape) -> MultiDiffusionDenoiser:
    # state on CPU, network on the test device
    average = PatchAverager(slices, shape, torch.device("cpu"))
    return MultiDiffusionDenoiser(net, average, inputs, shifts, device=get_device())


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
def test_PatchPredictor_multidiffusion_generate_on_batch_no_target(
    model_type, full_coarse_shape
):
    model: DiffusionModel | DenoisingMoEPredictor
    if model_type == "diffusion":
        model = _model(full_coarse_shape)
    else:
        model = _moe(full_coarse_shape)
    predictor = PatchPredictor(model, coarse_horizontal_overlap=4, multidiffusion=True)
    batch = _coarse_batch(full_coarse_shape, batch_size=2)
    generated = predictor.generate_on_batch_no_target(batch, n_samples=3)
    fine_shape = tuple(n * DOWNSCALE_FACTOR for n in full_coarse_shape)
    assert generated["x"].shape == (2, 3, *fine_shape)
    assert torch.isfinite(generated["x"]).all()


def test_PatchPredictor_multidiffusion_generate_on_batch():
    full_coarse_shape = (12, 12)
    model = _model(full_coarse_shape)
    predictor = PatchPredictor(model, coarse_horizontal_overlap=4, multidiffusion=True)
    coarse = _coarse_batch(full_coarse_shape, batch_size=2)
    fine_shape = (
        full_coarse_shape[0] * DOWNSCALE_FACTOR,
        full_coarse_shape[1] * DOWNSCALE_FACTOR,
    )
    fine = make_batch_data(
        (2, *fine_shape),
        cell_centered_coordinate(0.0, fine_shape[0], n=fine_shape[0]),
        cell_centered_coordinate(0.0, fine_shape[1], n=fine_shape[1]),
    )
    outputs = predictor.generate_on_batch(
        PairedBatchData(fine=fine, coarse=coarse), n_samples=2
    )
    assert outputs.prediction["x"].shape == (2, 2, *fine_shape)
    assert outputs.target["x"].shape == (2, 1, *fine_shape)
    assert outputs.loss.ndim == 0 and torch.isfinite(outputs.loss)
    assert outputs.latent_steps == []


def _multidiffusion_inputs():
    full_coarse_shape = (12, 12)
    model = _model(full_coarse_shape)
    batch = _coarse_batch(full_coarse_shape)
    patches = get_patches(full_coarse_shape, MODEL_COARSE_SHAPE, overlap=4)
    return model, batch, patches


def test_generate_multidiffusion_float32_sampler_close_to_float64():
    # On the same device and seed, the two sampler dtypes draw the same noise,
    # so their samples differ only by rounding.
    model, batch, patches = _multidiffusion_inputs()
    generated = {}
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(0)
        generated[dtype], _ = generate_multidiffusion(
            model, batch, patches, n_samples=2, sampler_dtype=dtype
        )
    torch.testing.assert_close(
        generated[torch.float32], generated[torch.float64], rtol=1e-4, atol=1e-4
    )


def test_generate_multidiffusion_sampler_state_on_cpu():
    model, batch, patches = _multidiffusion_inputs()
    torch.manual_seed(0)
    generated, generated_norm = generate_multidiffusion(
        model, batch, patches, n_samples=2, sampler_state_on_cpu=True
    )
    assert generated_norm.device.type == get_device().type
    assert generated["x"].shape == (2, 2, 24, 24)
    assert torch.isfinite(generated["x"]).all()
    if get_device().type == "cpu":
        torch.manual_seed(0)
        expected, _ = generate_multidiffusion(model, batch, patches, n_samples=2)
        torch.testing.assert_close(generated, expected)


def test_PatchPredictionConfig_multidiffusion_sampler_options_require_multidiffusion():
    for kwargs in (
        {"multidiffusion_sampler_dtype": "float32"},
        {"multidiffusion_sampler_state_on_cpu": True},
    ):
        with pytest.raises(ValueError, match="require multidiffusion=True"):
            PatchPredictionConfig(
                divide_generation=True, composite_prediction=True, **kwargs
            )


def test_PatchPredictionConfig_multidiffusion_requires_divide_generation():
    with pytest.raises(ValueError, match="multidiffusion=True requires"):
        PatchPredictionConfig(multidiffusion=True)
    config = PatchPredictionConfig(
        divide_generation=True, composite_prediction=True, multidiffusion=True
    )
    assert config.needs_patch_predictor
