import os
from unittest.mock import MagicMock

import pytest
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import OptimizationConfig
from fme.downscaling.data import (
    BatchData,
    BatchedLatLonCoordinates,
    PairedBatchData,
    StaticInput,
    StaticInputs,
)
from fme.downscaling.models import (
    CheckpointModelConfig,
    DiffusionModel,
    DiffusionModelConfig,
    PairedNormalizationConfig,
    _repeat_batch_by_samples,
    _separate_interleaved_samples,
)
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.noise import LogNormalNoiseDistribution
from fme.downscaling.typing_ import FineResCoarseResPair


class LinearDownscaling(torch.nn.Module):
    def __init__(
        self,
        factor: int,
        fine_img_shape: tuple[int, int],
        n_channels_in: int = 1,
        n_channels_out: int | None = None,
    ):
        super().__init__()
        self.img_shape = fine_img_shape
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out or n_channels_in
        height, width = fine_img_shape
        self.linear = torch.nn.Linear(
            ((height * width) // factor**2) * n_channels_in,
            height * width * self.n_channels_out,
            bias=False,
        )
        self._coarse_img_shape = (height // factor, width // factor)

    def forward(self, x):
        x = self.linear(torch.flatten(x, start_dim=1))
        x = x.view(x.shape[0], self.n_channels_out, *self.img_shape)
        return x


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# returns paired batch data mock
def get_mock_batch(shape, topography_scale_factor: int = 1):
    batch = MagicMock()
    batch.data = {"x": torch.ones(*shape, device=get_device())}
    return batch


def get_mock_paired_batch(coarse_shape, fine_shape):
    coarse = get_mock_batch(coarse_shape)
    fine = get_mock_batch(fine_shape)

    return FineResCoarseResPair(fine=fine, coarse=coarse)


def make_batch_data(
    shape: tuple[int, int, int],
    lat_values: list[float],
    lon_values: list[float],
) -> BatchData:
    """Create a BatchData with proper monotonic coordinates."""
    batch_size, lat_size, lon_size = shape
    assert lat_size == len(lat_values)
    assert lon_size == len(lon_values)
    data = {"x": torch.ones(batch_size, lat_size, lon_size, device=get_device())}
    time = xr.DataArray(range(batch_size), dims=["batch"])
    lat = torch.tensor(lat_values, dtype=torch.float32)
    lon = torch.tensor(lon_values, dtype=torch.float32)
    latlon = BatchedLatLonCoordinates(
        lat=lat.unsqueeze(0).expand(batch_size, -1),
        lon=lon.unsqueeze(0).expand(batch_size, -1),
    )
    return BatchData(data=data, time=time, latlon_coordinates=latlon)


def _get_monotonic_coordinate(size: int, stop: float) -> torch.Tensor:
    bounds = torch.linspace(0, stop, size + 1)
    coord = (bounds[:-1] + bounds[1:]) / 2
    return coord


def make_paired_batch_data(
    coarse_shape: tuple[int, int],
    fine_shape: tuple[int, int],
    batch_size: int = 2,
) -> PairedBatchData:
    """
    Create a PairedBatchData with consistent monotonic coordinates.
    """
    lat_c, lon_c = coarse_shape
    lat_f, lon_f = fine_shape
    fine_lat = _get_monotonic_coordinate(lat_f, stop=lat_f)
    fine_lon = _get_monotonic_coordinate(lon_f, stop=lon_f)
    coarse_lat = _get_monotonic_coordinate(lat_c, stop=lat_f)
    coarse_lon = _get_monotonic_coordinate(lon_c, stop=lon_f)
    fine = make_batch_data((batch_size, lat_f, lon_f), fine_lat, fine_lon)
    coarse = make_batch_data((batch_size, lat_c, lon_c), coarse_lat, coarse_lon)
    return PairedBatchData(fine=fine, coarse=coarse)


def make_fine_coords(fine_shape: tuple[int, int]) -> LatLonCoordinates:
    """Create LatLonCoordinates with proper monotonic coordinates for given shape."""
    lat_size, lon_size = fine_shape
    return LatLonCoordinates(
        lat=_get_monotonic_coordinate(lat_size, stop=lat_size),
        lon=_get_monotonic_coordinate(lon_size, stop=lon_size),
    )


def make_static_inputs(fine_shape: tuple[int, int]) -> StaticInputs:
    """Create StaticInputs with proper monotonic coordinates for given shape."""
    coords = make_fine_coords(fine_shape)
    return StaticInputs(
        fields=[StaticInput(torch.ones(*fine_shape, device=get_device()))],
        coords=coords,
    )


def test_module_serialization(tmp_path):
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    static_inputs = make_static_inputs(fine_shape)
    fine_coords = static_inputs.coords
    model = _get_diffusion_model(
        full_fine_coords=fine_coords,
        coarse_shape=coarse_shape,
        downscale_factor=2,
        predict_residual=True,
        use_fine_topography=False,
        static_inputs=static_inputs,
    )
    model_from_state = DiffusionModel.from_state(model.get_state())
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(
            model.module.parameters(), model_from_state.module.parameters()
        )
    )
    assert model_from_state.full_fine_coords is not None
    assert torch.equal(
        model_from_state.full_fine_coords.lat.cpu(), fine_coords.lat.cpu()
    )
    assert torch.equal(
        model_from_state.full_fine_coords.lon.cpu(), fine_coords.lon.cpu()
    )

    torch.save(model.get_state(), tmp_path / "test.ckpt")
    model_from_disk = DiffusionModel.from_state(
        torch.load(tmp_path / "test.ckpt", weights_only=False),
    )
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(
            model.module.parameters(), model_from_disk.module.parameters()
        )
    )
    loaded_static_inputs = model_from_disk.static_inputs
    assert loaded_static_inputs is not None
    assert torch.equal(
        loaded_static_inputs.fields[0].data, static_inputs.fields[0].data
    )
    assert model_from_disk.full_fine_coords is not None
    assert torch.equal(
        model_from_disk.full_fine_coords.lat.cpu(), fine_coords.lat.cpu()
    )
    assert torch.equal(
        model_from_disk.full_fine_coords.lon.cpu(), fine_coords.lon.cpu()
    )


def test_model_raises_when_no_static_fields_but_topography_required():
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    fine_coords = make_fine_coords(fine_shape)
    with pytest.raises(ValueError):
        _get_diffusion_model(
            coarse_shape=coarse_shape,
            downscale_factor=2,
            full_fine_coords=fine_coords,
            use_fine_topography=True,
            static_inputs=StaticInputs(fields=[], coords=fine_coords),
        )


def _get_diffusion_model(
    coarse_shape,
    downscale_factor,
    full_fine_coords: LatLonCoordinates,
    predict_residual=True,
    use_fine_topography=True,
    static_inputs: StaticInputs | None = None,
):
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )
    return DiffusionModelConfig(
        module=DiffusionModuleRegistrySelector(
            "unet_diffusion_song", {"model_channels": 4}
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
        predict_residual=predict_residual,
        use_fine_topography=use_fine_topography,
    ).build(
        coarse_shape,
        downscale_factor,
        full_fine_coords=full_fine_coords,
        static_inputs=static_inputs,
    )


@pytest.mark.parametrize("predict_residual", [True, False])
@pytest.mark.parametrize("use_fine_topography", [True, False])
def test_diffusion_model_train_and_generate(predict_residual, use_fine_topography):
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    batch_size = 2
    fine_coords = make_fine_coords(fine_shape)
    if use_fine_topography:
        static_inputs = make_static_inputs(fine_shape)
        batch = make_paired_batch_data(coarse_shape, fine_shape, batch_size)
    else:
        static_inputs = StaticInputs(fields=[], coords=fine_coords)
        batch = make_paired_batch_data(coarse_shape, fine_shape, batch_size)
    model = _get_diffusion_model(
        full_fine_coords=fine_coords,
        coarse_shape=coarse_shape,
        downscale_factor=2,
        predict_residual=predict_residual,
        use_fine_topography=use_fine_topography,
        static_inputs=static_inputs,
    )

    assert model._get_fine_shape(coarse_shape) == fine_shape

    optimization = OptimizationConfig().build(modules=[model.module], max_epochs=2)
    train_outputs = model.train_on_batch(batch, optimization)
    assert torch.allclose(train_outputs.target["x"], batch.fine.data["x"])

    n_generated_samples = 2
    generated_outputs = [
        model.generate_on_batch(batch) for _ in range(n_generated_samples)
    ]

    for generated_output in generated_outputs:
        assert (
            generated_output.prediction["x"].shape == generated_output.target["x"].shape
        )

    assert torch.all(
        generated_outputs[0].prediction["x"] != generated_outputs[1].prediction["x"]
    )


def test_interleaved_samples_round_trip():
    batch_size = 2
    n_samples = 3

    batch = torch.concat([torch.ones(1, 5), torch.ones(1, 5) * 2], dim=0)
    with_combined_samples = _repeat_batch_by_samples(batch, n_samples)
    with_batch_sample_dims = _separate_interleaved_samples(
        with_combined_samples, n_samples
    )
    assert with_batch_sample_dims.shape == (batch_size, n_samples, 5)
    assert torch.equal(batch, with_batch_sample_dims[:, 0])


def test_normalizer_serialization(tmp_path):
    coarse_shape = (8, 16)

    means = xr.Dataset({"x": 0.0})
    stds = xr.Dataset({"x": 1.0})
    means.to_netcdf(tmp_path / "means.nc")
    stds.to_netcdf(tmp_path / "stds.nc")
    fine_shape = (coarse_shape[0] * 2, coarse_shape[1] * 2)
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=2,
        full_fine_coords=make_fine_coords(fine_shape),
        predict_residual=False,
        use_fine_topography=False,
    )
    torch.save(model.get_state(), tmp_path / "test.ckpt")

    # normalization should be loaded into model config when get_state called,
    # delete netcdfs to check that data is dumped and loaded with checkpoint
    os.remove(tmp_path / "means.nc")
    os.remove(tmp_path / "stds.nc")

    model_from_disk = DiffusionModel.from_state(
        torch.load(tmp_path / "test.ckpt", weights_only=False),
    )

    assert model_from_disk.normalizer.fine.means == {"x": 0}
    assert model_from_disk.normalizer.fine.stds == {"x": 1}
    assert model_from_disk.normalizer.coarse.means == {"x": 0}
    assert model_from_disk.normalizer.coarse.stds == {"x": 1}


def test_loss_weights_scale_channel_losses():
    """Per-variable loss_weights should scale each channel's contribution."""
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    batch_size = 2
    out_names = ["a", "b"]

    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"a": 0.0, "b": 0.0}, stds={"a": 1.0, "b": 1.0}),
        NormalizationConfig(means={"a": 0.0, "b": 0.0}, stds={"a": 1.0, "b": 1.0}),
    )
    fine_coords = make_fine_coords(fine_shape)

    def _build_model(loss_weights):
        return DiffusionModelConfig(
            module=DiffusionModuleRegistrySelector(
                "unet_diffusion_song", {"model_channels": 4}
            ),
            loss=LossConfig(type="MSE"),
            in_names=["a"],
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
            loss_weights=loss_weights,
        ).build(
            coarse_shape,
            2,
            full_fine_coords=fine_coords,
        )

    model_uniform = _build_model({})
    model_weighted = _build_model({"a": 0.0, "b": 2.0})

    # Copy weights so both models produce identical predictions
    model_weighted.module.load_state_dict(model_uniform.module.state_dict())

    def _make_batch():
        coarse_data = {
            "a": torch.randn(batch_size, *coarse_shape, device=get_device()),
            "b": torch.randn(batch_size, *coarse_shape, device=get_device()),
        }
        fine_data = {
            "a": torch.randn(batch_size, *fine_shape, device=get_device()),
            "b": torch.randn(batch_size, *fine_shape, device=get_device()),
        }
        coarse_lat = _get_monotonic_coordinate(coarse_shape[0], stop=fine_shape[0])
        coarse_lon = _get_monotonic_coordinate(coarse_shape[1], stop=fine_shape[1])
        fine_lat = _get_monotonic_coordinate(fine_shape[0], stop=fine_shape[0])
        fine_lon = _get_monotonic_coordinate(fine_shape[1], stop=fine_shape[1])
        time = xr.DataArray(range(batch_size), dims=["batch"])
        coarse_batch = BatchData(
            data=coarse_data,
            time=time,
            latlon_coordinates=BatchedLatLonCoordinates(
                lat=coarse_lat.unsqueeze(0).expand(batch_size, -1),
                lon=coarse_lon.unsqueeze(0).expand(batch_size, -1),
            ),
        )
        fine_batch = BatchData(
            data=fine_data,
            time=time,
            latlon_coordinates=BatchedLatLonCoordinates(
                lat=fine_lat.unsqueeze(0).expand(batch_size, -1),
                lon=fine_lon.unsqueeze(0).expand(batch_size, -1),
            ),
        )
        return PairedBatchData(fine=fine_batch, coarse=coarse_batch)

    torch.manual_seed(0)
    batch = _make_batch()
    optimization = OptimizationConfig().build(
        modules=[model_uniform.module], max_epochs=2
    )
    torch.manual_seed(42)
    out_uniform = model_uniform.train_on_batch(batch, optimization)

    optimization_w = OptimizationConfig().build(
        modules=[model_weighted.module], max_epochs=2
    )
    torch.manual_seed(42)
    out_weighted = model_weighted.train_on_batch(batch, optimization_w)

    assert out_weighted.channel_losses["a"] == pytest.approx(0.0, abs=1e-7)
    assert out_weighted.channel_losses["b"] > out_uniform.channel_losses["b"]


def test_use_fine_topography_raises_when_module_does_not_use_interpolated_input():
    normalization_config = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )
    invalid_selector = DiffusionModuleRegistrySelector(
        "prebuilt",
        {"module": None},
        expects_interpolated_input=False,
    )
    with pytest.raises(ValueError):
        DiffusionModelConfig(  # type: ignore
            invalid_selector,  # type: ignore
            LossConfig(type="MSE"),
            ["x"],
            ["x"],
            normalization_config,
            use_fine_topography=True,
            p_mean=-1.0,
            p_std=1.0,
            sigma_min=0.1,
            sigma_max=1.0,
            churn=0.5,
            num_diffusion_generation_steps=3,
            predict_residual=True,
        )


def test_DiffusionModel_generate_on_batch_no_target():
    fine_shape = (32, 32)
    coarse_shape = (16, 16)
    downscale_factor = 2
    static_inputs = make_static_inputs(fine_shape)
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        full_fine_coords=static_inputs.coords,
        predict_residual=True,
        use_fine_topography=True,
        static_inputs=static_inputs,
    )

    batch_size = 2
    n_generated_samples = 2

    coarse_lat = _get_monotonic_coordinate(coarse_shape[0], stop=fine_shape[0])
    coarse_lon = _get_monotonic_coordinate(coarse_shape[1], stop=fine_shape[1])
    coarse_batch = make_batch_data((batch_size, *coarse_shape), coarse_lat, coarse_lon)

    samples = model.generate_on_batch_no_target(
        coarse_batch,
        n_samples=n_generated_samples,
    )

    assert samples["x"].shape == (
        batch_size,
        n_generated_samples,
        *fine_shape,
    )


def test_DiffusionModel_generate_on_batch_no_target_arbitrary_input_size():
    # The model subsets its own stored static_inputs based on coarse batch
    # coordinates. The stored static_inputs must cover the full fine domain
    # for all tested batch sizes.
    coarse_shape = (16, 16)
    downscale_factor = 2
    # Full fine domain: 64x64 covers inputs for both (8,8) and (32,32) coarse inputs
    # with a downscaling factor of 2
    full_fine_size = 64
    full_fine_shape = (full_fine_size, full_fine_size)
    static_inputs = make_static_inputs(full_fine_shape)
    # need to build with static inputs to get the correct n_in_channels
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        full_fine_coords=static_inputs.coords,
        predict_residual=True,
        use_fine_topography=True,
        static_inputs=static_inputs,
    )
    n_ensemble = 2
    batch_size = 2

    for alternative_input_shape in [(8, 8), (32, 32)]:
        fine_shape = tuple(dim * downscale_factor for dim in alternative_input_shape)
        alt_y, alt_x = alternative_input_shape
        coarse_lat = _get_monotonic_coordinate(alt_y, stop=alt_y * downscale_factor)
        coarse_lon = _get_monotonic_coordinate(alt_x, stop=alt_x * downscale_factor)
        coarse_batch = make_batch_data(
            (batch_size, *alternative_input_shape), coarse_lat, coarse_lon
        )
        samples = model.generate_on_batch_no_target(coarse_batch, n_samples=n_ensemble)

        assert samples["x"].shape == (
            batch_size,
            n_ensemble,
            *fine_shape,
        )


def test_lognorm_noise_backwards_compatibility():
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )

    model_config = DiffusionModelConfig(
        module=DiffusionModuleRegistrySelector(
            "unet_diffusion_song", {"model_channels": 4}
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
        training_noise_distribution=None,
        use_fine_topography=False,
        predict_residual=True,
    )
    assert model_config.noise_distribution == LogNormalNoiseDistribution(
        p_mean=-1.0, p_std=1.0
    )
    model = model_config.build(
        (32, 32),
        2,
        full_fine_coords=make_fine_coords((64, 64)),
        static_inputs=StaticInputs(fields=[], coords=make_fine_coords((64, 64))),
    )
    state = model.get_state()

    # test from_state on checkpoints saved prior to noise distribution classes
    del state["config"]["training_noise_distribution"]
    model_from_state = DiffusionModel.from_state(state)
    assert model_from_state.config.noise_distribution == LogNormalNoiseDistribution(
        p_mean=-1.0, p_std=1.0
    )


def test_noise_config_error():
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )

    with pytest.raises(ValueError):
        DiffusionModelConfig(
            module=DiffusionModuleRegistrySelector(
                "unet_diffusion_song", {"model_channels": 4}
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
            training_noise_distribution=LogNormalNoiseDistribution(-1.0, 1.0),
            use_fine_topography=False,
            predict_residual=True,
        )


def test_get_fine_coords_for_batch():
    # Model trained on full coarse (8x16) / fine (16x32) grid
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    downscale_factor = 2
    static_inputs = make_static_inputs(fine_shape)
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        full_fine_coords=static_inputs.coords,
        use_fine_topography=True,
        static_inputs=static_inputs,
    )

    # Build a batch covering a spatial patch: middle 4 coarse lats and 8 coarse lons.
    full_coarse_lat = _get_monotonic_coordinate(coarse_shape[0], stop=fine_shape[0])
    full_coarse_lon = _get_monotonic_coordinate(coarse_shape[1], stop=fine_shape[1])
    patch_coarse_lat = full_coarse_lat[2:6].tolist()  # [5, 7, 9, 11]
    patch_coarse_lon = full_coarse_lon[4:12].tolist()  # [9, 11, ..., 23]
    batch = make_batch_data((2, 4, 8), patch_coarse_lat, patch_coarse_lon)

    result = model.get_fine_coords_for_batch(batch)

    expected_lat = model.full_fine_coords.lat[4:12]
    expected_lon = model.full_fine_coords.lon[8:24]
    assert torch.allclose(result.lat, expected_lat)
    assert torch.allclose(result.lon, expected_lon)


def test_checkpoint_config_topography_raises():
    with pytest.raises(ValueError):
        CheckpointModelConfig(
            checkpoint_path="/any/path.ckpt",
            fine_topography_path="/topo/path.nc",
        )


def test_checkpoint_model_build_raises_when_checkpoint_has_static_inputs(tmp_path):
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    static_inputs = make_static_inputs(fine_shape)
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=2,
        full_fine_coords=static_inputs.coords,
        predict_residual=True,
        use_fine_topography=True,
        static_inputs=static_inputs,
    )
    checkpoint_path = tmp_path / "test.ckpt"
    torch.save({"model": model.get_state()}, checkpoint_path)

    config = CheckpointModelConfig(
        checkpoint_path=str(checkpoint_path),
        static_inputs={"HGTsfc": "/any/path.nc"},
    )
    with pytest.raises(ValueError):
        config.build()


def test_checkpoint_model_build_with_fine_coordinates_path(tmp_path):
    """Old-format checkpoint (no full_fine_coords key, no coords in static_inputs)
    should load correctly when fine_coordinates_path is provided."""
    coarse_shape = (8, 16)
    fine_shape = (coarse_shape[0] * 2, coarse_shape[1] * 2)
    fine_coords = make_fine_coords(fine_shape)
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=2,
        full_fine_coords=fine_coords,
        use_fine_topography=False,
        static_inputs=StaticInputs(fields=[], coords=fine_coords),
    )
    state = model.get_state()
    # Simulate old checkpoint: no full_fine_coords, no coords in static_inputs
    del state["full_fine_coords"]
    state["static_inputs"] = None

    checkpoint_path = tmp_path / "test.ckpt"
    torch.save({"model": state}, checkpoint_path)

    # Write fine coords to a netCDF file
    coords_path = tmp_path / "coords.nc"
    xr.Dataset(
        {
            "lat": xr.DataArray(fine_coords.lat.numpy(), dims=["lat"]),
            "lon": xr.DataArray(fine_coords.lon.numpy(), dims=["lon"]),
        }
    ).to_netcdf(coords_path)

    config = CheckpointModelConfig(
        checkpoint_path=str(checkpoint_path),
        fine_coordinates_path=str(coords_path),
    )
    loaded_model = config.build()
    assert loaded_model.full_fine_coords is not None
    assert torch.equal(loaded_model.full_fine_coords.lat.cpu(), fine_coords.lat.cpu())
    assert torch.equal(loaded_model.full_fine_coords.lon.cpu(), fine_coords.lon.cpu())


def test_checkpoint_model_build(tmp_path):
    """CheckpointModelConfig loads a modern checkpoint and restores the model."""
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    fine_coords = make_fine_coords(fine_shape)
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=2,
        full_fine_coords=fine_coords,
        use_fine_topography=False,
        static_inputs=StaticInputs(fields=[], coords=fine_coords),
    )
    checkpoint_path = tmp_path / "test.ckpt"
    torch.save({"model": model.get_state()}, checkpoint_path)

    loaded_model = CheckpointModelConfig(checkpoint_path=str(checkpoint_path)).build()
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model.module.parameters(), loaded_model.module.parameters())
    )
    assert torch.equal(loaded_model.full_fine_coords.lat.cpu(), fine_coords.lat.cpu())
    assert torch.equal(loaded_model.full_fine_coords.lon.cpu(), fine_coords.lon.cpu())
    assert (
        not loaded_model.module.training
    ), "Module should be in eval mode after build() to disable dropout"


def test_checkpoint_model_build_old_checkpoint_without_bottleneck_attention(tmp_path):
    """CheckpointModelConfig.build() falls back to bottleneck_attention=False for old
    checkpoints whose module config lacks the bottleneck_attention key.

    Old checkpoints were trained with bottleneck_attention=False (hardcoded), so
    their weights don't have the in0 attention layers.  After the bottleneck_attention
    parameter was added with a default of True, loading would fail with a missing-key
    RuntimeError.  The fix retries with bottleneck_attention=False.
    """
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    fine_coords = make_fine_coords(fine_shape)
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )
    # Build the "old" model explicitly with bottleneck_attention=False
    model = DiffusionModelConfig(
        module=DiffusionModuleRegistrySelector(
            "unet_diffusion_song",
            {"model_channels": 4, "bottleneck_attention": False},
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
        coarse_shape,
        2,
        full_fine_coords=fine_coords,
        static_inputs=StaticInputs(fields=[], coords=fine_coords),
    )
    state = {"model": model.get_state()}
    # Strip bottleneck_attention to simulate a checkpoint saved before the field existed
    del state["model"]["config"]["module"]["config"]["bottleneck_attention"]
    checkpoint_path = tmp_path / "old.ckpt"
    torch.save(state, checkpoint_path)

    loaded_model = CheckpointModelConfig(checkpoint_path=str(checkpoint_path)).build()
    assert not loaded_model.module.training


def test_build_raises_when_static_inputs_coords_mismatch_full_fine_coords():
    """Building with static_inputs whose coords differ from full_fine_coords raises."""
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    fine_coords = make_fine_coords(fine_shape)
    # Same shape but offset values — a different grid covering a different region
    shifted_coords = LatLonCoordinates(
        lat=fine_coords.lat + 10.0,
        lon=fine_coords.lon + 10.0,
    )
    with pytest.raises(ValueError):
        _get_diffusion_model(
            coarse_shape=coarse_shape,
            downscale_factor=2,
            full_fine_coords=fine_coords,
            use_fine_topography=False,
            static_inputs=StaticInputs(fields=[], coords=shifted_coords),
        )


def test_from_state_raises_for_unresolvable_old_checkpoint(tmp_path):
    """DiffusionModel.from_state raises clearly when checkpoint has no fine coords."""
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    fine_coords = make_fine_coords(fine_shape)
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=2,
        full_fine_coords=fine_coords,
        use_fine_topography=False,
        static_inputs=StaticInputs(fields=[], coords=fine_coords),
    )
    state = model.get_state()
    del state["full_fine_coords"]
    state["static_inputs"] = None

    with pytest.raises(ValueError, match="full_fine_coords"):
        DiffusionModel.from_state(state)
