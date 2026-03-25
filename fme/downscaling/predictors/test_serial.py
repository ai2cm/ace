import pytest
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.downscaling.data import BatchData, BatchedLatLonCoordinates, PairedBatchData
from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.predictors.serial import SerialPredictor


def _get_monotonic_coordinate(size: int, stop: float) -> torch.Tensor:
    bounds = torch.linspace(0, stop, size + 1)
    return (bounds[:-1] + bounds[1:]) / 2


def _make_batch_data(
    shape: tuple[int, int, int],
    lat_values,
    lon_values,
    names: list[str],
) -> BatchData:
    batch_size, lat_size, lon_size = shape
    data = {
        name: torch.ones(batch_size, lat_size, lon_size, device=get_device())
        for name in names
    }
    time = xr.DataArray(range(batch_size), dims=["batch"])
    lat = torch.tensor(lat_values, dtype=torch.float32)
    lon = torch.tensor(lon_values, dtype=torch.float32)
    latlon = BatchedLatLonCoordinates(
        lat=lat.unsqueeze(0).expand(batch_size, -1),
        lon=lon.unsqueeze(0).expand(batch_size, -1),
    )
    return BatchData(data=data, time=time, latlon_coordinates=latlon)


def _build_model(
    coarse_shape,
    downscale_factor,
    in_names,
    out_names,
    high_res_conditioning=None,
):
    """Build a DiffusionModel with given variable names."""
    all_coarse = list(set(in_names) | set(out_names))
    coarse_means = {n: 0.0 for n in all_coarse}
    coarse_stds = {n: 1.0 for n in all_coarse}
    all_fine = list(set(out_names) | set(high_res_conditioning or []))
    fine_means = {n: 0.0 for n in all_fine}
    fine_stds = {n: 1.0 for n in all_fine}

    normalizer = PairedNormalizationConfig(
        fine=NormalizationConfig(means=fine_means, stds=fine_stds),
        coarse=NormalizationConfig(means=coarse_means, stds=coarse_stds),
    )
    config = DiffusionModelConfig(
        module=DiffusionModuleRegistrySelector(
            "unet_diffusion_song", {"model_channels": 4}
        ),
        loss=LossConfig(type="MSE"),
        in_names=in_names,
        out_names=out_names,
        normalization=normalizer,
        p_mean=-1.0,
        p_std=1.0,
        sigma_min=0.1,
        sigma_max=1.0,
        churn=0.5,
        num_diffusion_generation_steps=3,
        predict_residual=False,
        use_fine_topography=False,
        high_res_conditioning=high_res_conditioning,
    )
    return config.build(coarse_shape, downscale_factor)


def _make_paired_batch(
    coarse_shape,
    fine_shape,
    coarse_names,
    fine_names,
    batch_size=2,
):
    lat_c, lon_c = coarse_shape
    lat_f, lon_f = fine_shape
    fine_lat = _get_monotonic_coordinate(lat_f, stop=lat_f)
    fine_lon = _get_monotonic_coordinate(lon_f, stop=lon_f)
    coarse_lat = _get_monotonic_coordinate(lat_c, stop=lat_f)
    coarse_lon = _get_monotonic_coordinate(lon_c, stop=lon_f)
    fine = _make_batch_data((batch_size, lat_f, lon_f), fine_lat, fine_lon, fine_names)
    coarse = _make_batch_data(
        (batch_size, lat_c, lon_c), coarse_lat, coarse_lon, coarse_names
    )
    return PairedBatchData(fine=fine, coarse=coarse)


COARSE_SHAPE = (8, 16)
FINE_SHAPE = (16, 32)
DOWNSCALE_FACTOR = 2


def test_generate_on_batch_no_target():
    batch_size = 2
    first_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["b"],
    )
    second_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["c"],
        high_res_conditioning=["b"],
    )
    predictor = SerialPredictor(first_model, second_model)

    coarse_lat = _get_monotonic_coordinate(COARSE_SHAPE[0], stop=FINE_SHAPE[0])
    coarse_lon = _get_monotonic_coordinate(COARSE_SHAPE[1], stop=FINE_SHAPE[1])
    batch = _make_batch_data(
        (batch_size, *COARSE_SHAPE), coarse_lat, coarse_lon, names=["a"]
    )

    result = predictor.generate_on_batch_no_target(batch, n_samples=2)
    assert "c" in result
    assert result["c"].shape == (batch_size, 2, *FINE_SHAPE)


def test_generate_on_batch_no_target_passthrough_outputs():
    """First model outputs both conditioning and passthrough fields."""
    batch_size = 2
    # First model outputs "b" (conditioning) and "d" (passthrough)
    first_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["b", "d"],
    )
    second_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["c"],
        high_res_conditioning=["b"],
    )
    predictor = SerialPredictor(first_model, second_model)

    coarse_lat = _get_monotonic_coordinate(COARSE_SHAPE[0], stop=FINE_SHAPE[0])
    coarse_lon = _get_monotonic_coordinate(COARSE_SHAPE[1], stop=FINE_SHAPE[1])
    batch = _make_batch_data(
        (batch_size, *COARSE_SHAPE), coarse_lat, coarse_lon, names=["a"]
    )

    n_samples = 3
    result = predictor.generate_on_batch_no_target(batch, n_samples=n_samples)
    assert "b" in result
    assert "c" in result
    assert "d" in result
    assert result["b"].shape == (batch_size, n_samples, *FINE_SHAPE)
    assert result["c"].shape == (batch_size, n_samples, *FINE_SHAPE)
    assert result["d"].shape == (batch_size, n_samples, *FINE_SHAPE)


def test_generate_on_batch():
    batch_size = 2
    first_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["b"],
    )
    second_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["c"],
        high_res_conditioning=["b"],
    )
    predictor = SerialPredictor(first_model, second_model)

    batch = _make_paired_batch(
        COARSE_SHAPE,
        FINE_SHAPE,
        coarse_names=["a", "c"],
        fine_names=["c"],
        batch_size=batch_size,
    )
    result = predictor.generate_on_batch(batch, n_samples=1)
    assert "c" in result.prediction
    assert result.prediction["c"].shape == (batch_size, 1, *FINE_SHAPE)


def test_generate_on_batch_passthrough_outputs():
    """First model outputs "b" (conditioning) and "d" (passthrough).
    Both "c" and "d" should appear in the final prediction."""
    batch_size = 2
    first_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["b", "d"],
    )
    second_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["c"],
        high_res_conditioning=["b"],
    )
    predictor = SerialPredictor(first_model, second_model)

    batch = _make_paired_batch(
        COARSE_SHAPE,
        FINE_SHAPE,
        coarse_names=["a", "c", "d"],
        fine_names=["c", "d"],
        batch_size=batch_size,
    )
    result = predictor.generate_on_batch(batch, n_samples=1)
    assert "c" in result.prediction
    assert "d" in result.prediction
    assert result.prediction["d"].shape == (batch_size, 1, *FINE_SHAPE)
    # target for passthrough field is included from fine data
    assert "d" in result.target
    assert result.target["d"].shape == (batch_size, 1, *FINE_SHAPE)


def test_properties():
    first_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["b"],
    )
    second_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["c"],
        high_res_conditioning=["b"],
    )
    predictor = SerialPredictor(first_model, second_model)

    assert predictor.coarse_shape == COARSE_SHAPE
    assert predictor.downscale_factor == DOWNSCALE_FACTOR
    assert len(predictor.modules) == 2


def test_conditioning_not_subset_of_first_out_names_raises():
    first_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["wrong"],
    )
    second_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["c"],
        high_res_conditioning=["b"],
    )
    with pytest.raises(ValueError, match="not found in"):
        SerialPredictor(first_model, second_model)


def test_overlapping_out_names_raises():
    first_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["b", "c"],
    )
    second_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["c"],
        high_res_conditioning=["b"],
    )
    with pytest.raises(ValueError, match="outputs must not overlap"):
        SerialPredictor(first_model, second_model)


def test_no_high_res_conditioning_raises():
    first_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["b"],
    )
    second_model = _build_model(
        COARSE_SHAPE,
        DOWNSCALE_FACTOR,
        in_names=["a"],
        out_names=["c"],
    )
    with pytest.raises(ValueError, match="high_res_conditioning"):
        SerialPredictor(first_model, second_model)
