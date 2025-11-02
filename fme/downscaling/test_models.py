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
from fme.downscaling.data import Topography
from fme.downscaling.models import (
    DiffusionModel,
    DiffusionModelConfig,
    PairedNormalizationConfig,
    _repeat_batch_by_samples,
    _separate_interleaved_samples,
)
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
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


def test_module_serialization(tmp_path):
    coarse_shape = (8, 16)
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=2,
        predict_residual=True,
        use_fine_topography=False,
    )
    model_from_state = DiffusionModel.from_state(
        model.get_state(),
    )
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(
            model.module.parameters(), model_from_state.module.parameters()
        )
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


def _get_diffusion_model(
    coarse_shape,
    downscale_factor,
    predict_residual=True,
    use_fine_topography=True,
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
    ).build(coarse_shape, downscale_factor)


@pytest.mark.parametrize("predict_residual", [True, False])
@pytest.mark.parametrize("use_fine_topography", [True, False])
def test_diffusion_model_train_and_generate(predict_residual, use_fine_topography):
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=2,
        predict_residual=predict_residual,
        use_fine_topography=use_fine_topography,
    )

    assert model._get_fine_shape(coarse_shape) == fine_shape

    batch_size = 2

    batch = get_mock_paired_batch(
        [batch_size, *coarse_shape], [batch_size, *fine_shape]
    )
    if use_fine_topography:
        topography = Topography(
            torch.ones(*fine_shape, device=get_device()),
            LatLonCoordinates(
                lat=torch.ones(fine_shape[0]), lon=torch.ones(fine_shape[1])
            ),
        )
    else:
        topography = None
    optimization = OptimizationConfig().build(modules=[model.module], max_epochs=2)
    train_outputs = model.train_on_batch(batch, topography, optimization)
    assert torch.allclose(train_outputs.target["x"], batch.fine.data["x"])

    n_generated_samples = 2
    generated_outputs = [
        model.generate_on_batch(batch, topography) for _ in range(n_generated_samples)
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
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=2,
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


def test_model_error_cases():
    fine_shape = (8, 16)
    coarse_shape = (4, 8)
    upscaling_factor = 2
    batch_size = 3
    normalization_config = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )

    selector: type[DiffusionModuleRegistrySelector]
    model_class = DiffusionModelConfig
    selector = DiffusionModuleRegistrySelector
    extra_kwargs = {
        "p_mean": -1.0,
        "p_std": 1.0,
        "sigma_min": 0.1,
        "sigma_max": 1.0,
        "churn": 0.5,
        "num_diffusion_generation_steps": 3,
        "predict_residual": True,
    }

    # Incompatible on init check
    invalid_selector = selector(
        "prebuilt",
        {"module": None},
        expects_interpolated_input=False,
    )
    with pytest.raises(ValueError):
        model_class(  # type: ignore
            invalid_selector,  # type: ignore
            LossConfig(type="MSE"),
            ["x"],
            ["x"],
            normalization_config,
            use_fine_topography=True,
            **extra_kwargs,  # type: ignore
        )

    # Compatible init, but no topography provided during prediction
    module_selector = selector(
        "prebuilt",
        {"module": DummyModule()},
        expects_interpolated_input=True,
    )
    model = model_class(  # type: ignore
        module_selector,  # type: ignore
        LossConfig(type="MSE"),
        ["x"],
        ["x"],
        normalization_config,
        use_fine_topography=True,
        **extra_kwargs,  # type: ignore
    ).build(
        coarse_shape,
        upscaling_factor,
    )
    batch = get_mock_paired_batch(
        [batch_size, *coarse_shape], [batch_size, *fine_shape]
    )

    # missing fine topography when model requires it
    batch.fine.topography = None
    with pytest.raises(ValueError):
        model.generate_on_batch(batch, topography=None)


def test_DiffusionModel_generate_on_batch_no_target():
    fine_shape = (32, 32)
    coarse_shape = (16, 16)
    downscale_factor = 2
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        predict_residual=True,
        use_fine_topography=True,
    )

    batch_size = 2

    n_generated_samples = 2

    coarse_batch = get_mock_batch(
        [batch_size, *coarse_shape], topography_scale_factor=downscale_factor
    )
    topography = Topography(
        torch.rand(*fine_shape, device=get_device()),
        LatLonCoordinates(lat=torch.ones(fine_shape[0]), lon=torch.ones(fine_shape[1])),
    )
    samples = model.generate_on_batch_no_target(
        coarse_batch,
        topography=topography,
        n_samples=n_generated_samples,
    )

    assert samples["x"].shape == (
        batch_size,
        n_generated_samples,
        *fine_shape,
    )


def test_DiffusionModel_generate_on_batch_no_target_arbitrary_input_size():
    # We currently require an input coarse shape for accounting, but the model
    # can handle arbitrary input sizes

    coarse_shape = (16, 16)
    downscale_factor = 2
    model = _get_diffusion_model(
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        predict_residual=True,
        use_fine_topography=True,
    )
    n_ensemble = 2
    batch_size = 2

    for alternative_input_shape in [(8, 8), (32, 32)]:
        fine_shape = tuple(dim * downscale_factor for dim in alternative_input_shape)
        coarse_batch = get_mock_batch(
            [batch_size, *alternative_input_shape],
            topography_scale_factor=downscale_factor,
        )
        topography = Topography(
            torch.rand(*fine_shape, device=get_device()),
            LatLonCoordinates(torch.ones(fine_shape[0]), torch.ones(fine_shape[1])),
        )
        samples = model.generate_on_batch_no_target(
            coarse_batch, n_samples=n_ensemble, topography=topography
        )

        assert samples["x"].shape == (
            batch_size,
            n_ensemble,
            *fine_shape,
        )
