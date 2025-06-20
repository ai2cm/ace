import os
from unittest.mock import MagicMock

import pytest
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import OptimizationConfig
from fme.downscaling.models import (
    DiffusionModelConfig,
    DownscalingModelConfig,
    Model,
    PairedNormalizationConfig,
    _repeat_batch_by_samples,
    _separate_interleaved_samples,
)
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.typing_ import FineResCoarseResPair


class LinearDownscaling(torch.nn.Module):
    def __init__(
        self,
        factor: int,
        fine_img_shape: tuple[int, int],
        n_channels: int = 1,
    ):
        super().__init__()
        self.img_shape = fine_img_shape
        self.n_channels = n_channels
        height, width = fine_img_shape
        self.linear = torch.nn.Linear(
            ((height * width) // factor**2) * n_channels,
            height * width * n_channels,
            bias=False,
        )
        self._coarse_img_shape = (height // factor, width // factor)

    def forward(self, x):
        if tuple(x.shape[-2:]) != self._coarse_img_shape:
            raise ValueError(
                f"Expected input shape {self._coarse_img_shape}, got {x.shape[-2:]}"
            )
        x = self.linear(torch.flatten(x, start_dim=1))
        x = x.view(x.shape[0], self.n_channels, *self.img_shape)
        return x


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# returns paired batch data mock
def get_mock_batch(shape):
    batch = MagicMock()
    batch.data = {"x": torch.ones(*shape, device=get_device())}

    return batch


def get_mock_paired_batch(coarse_shape, fine_shape):
    coarse = get_mock_batch(coarse_shape)
    fine = get_mock_batch(fine_shape)

    return FineResCoarseResPair(fine=fine, coarse=coarse)


@pytest.mark.parametrize("use_opt", [True, False])
def test_train_and_generate(use_opt):
    fine_shape = (8, 16)
    coarse_shape = (4, 8)
    upscaling_factor = 2
    normalization_config = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )

    batch_size = 3
    module_selector = ModuleRegistrySelector(
        "prebuilt",
        {
            "module": LinearDownscaling(
                factor=upscaling_factor,
                fine_img_shape=fine_shape,
            )
        },
        expects_interpolated_input=False,
    )
    model = DownscalingModelConfig(
        module_selector,
        LossConfig(type="MSE"),
        ["x"],
        ["x"],
        normalization_config,
    ).build(
        coarse_shape,
        upscaling_factor,
    )
    batch = get_mock_paired_batch(
        [batch_size, *coarse_shape], [batch_size, *fine_shape]
    )
    if use_opt:
        optimization = OptimizationConfig().build(modules=[model.module], max_epochs=2)
        outputs = model.train_on_batch(batch, optimization)
    else:
        outputs = model.generate_on_batch(batch)

    assert outputs.prediction.keys() == outputs.target.keys()
    for k in outputs.prediction:
        assert outputs.prediction[k].shape == outputs.target[k].shape


@pytest.mark.parametrize(
    "in_names, out_names",
    [
        pytest.param(["x"], ["x"], id="in_names = out_names"),
        pytest.param(["x", "y"], ["x"], id="in_names > out_names"),
        pytest.param(["x"], ["x", "y"], id="in_names < out_names"),
    ],
)
def test_build_downscaling_model_config_runs(in_names, out_names):
    normalization = PairedNormalizationConfig(
        fine=NormalizationConfig(
            means={n: 0.0 for n in out_names}, stds={n: 1.0 for n in out_names}
        ),
        coarse=NormalizationConfig(
            means={n: 0.0 for n in in_names}, stds={n: 1.0 for n in in_names}
        ),
    )

    img_shape, downscale_factor = (4, 8), 4
    loss = LossConfig(type="L1")
    model_config = DownscalingModelConfig(
        ModuleRegistrySelector(
            "prebuilt",
            {"module": LinearDownscaling(4, (4, 8))},
            expects_interpolated_input=False,
        ),
        loss,
        ["x"],
        ["x"],
        normalization,
    )
    model_config.build(
        img_shape,
        downscale_factor,
    )


def test_serialization(tmp_path):
    fine_shape = (16, 32)
    coarse_shape = (8, 16)
    downscale_factor = 2
    module = LinearDownscaling(factor=2, fine_img_shape=fine_shape)
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )
    model = DownscalingModelConfig(
        ModuleRegistrySelector(
            "prebuilt", {"module": module}, expects_interpolated_input=False
        ),
        LossConfig(type="MSE"),
        ["x"],
        ["x"],
        normalizer,
    ).build(coarse_shape, downscale_factor)

    batch_size = 3
    batch = get_mock_paired_batch(
        [batch_size, *coarse_shape], [batch_size, *fine_shape]
    )
    expected = model.generate_on_batch(batch).prediction["x"]

    model_from_state = Model.from_state(
        model.get_state(),
    )
    torch.testing.assert_close(
        expected,
        model_from_state.generate_on_batch(batch).prediction["x"],
    )

    torch.save(model.get_state(), tmp_path / "test.ckpt")
    model_from_disk = Model.from_state(
        torch.load(tmp_path / "test.ckpt", weights_only=False),
    )
    torch.testing.assert_close(
        expected,
        model_from_disk.generate_on_batch(batch).prediction["x"],
    )


@pytest.mark.parametrize("predict_residual", [True, False])
@pytest.mark.parametrize("use_fine_topography", [True, False])
def test_diffusion_model_train_and_generate(predict_residual, use_fine_topography):
    fine_shape = (16, 32)
    coarse_shape = (8, 16)
    downscale_factor = 2
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )

    model = DiffusionModelConfig(
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

    batch_size = 2
    if use_fine_topography:
        topography = torch.ones(batch_size, *fine_shape, device=get_device())
    else:
        topography = None
    batch = get_mock_paired_batch(
        [batch_size, *coarse_shape], [batch_size, *fine_shape]
    )
    batch.fine.topography = topography
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
    fine_shape = (16, 32)
    coarse_shape = (8, 16)
    downscale_factor = 2
    module = LinearDownscaling(factor=2, fine_img_shape=fine_shape)

    means = xr.Dataset({"x": 0.0})
    stds = xr.Dataset({"x": 1.0})
    means.to_netcdf(tmp_path / "means.nc")
    stds.to_netcdf(tmp_path / "stds.nc")

    normalizer = PairedNormalizationConfig(
        NormalizationConfig(
            global_means_path=tmp_path / "means.nc",
            global_stds_path=tmp_path / "stds.nc",
        ),
        NormalizationConfig(
            global_means_path=tmp_path / "means.nc",
            global_stds_path=tmp_path / "stds.nc",
        ),
    )
    model = DownscalingModelConfig(
        ModuleRegistrySelector(
            "prebuilt", {"module": module}, expects_interpolated_input=False
        ),
        LossConfig(type="MSE"),
        ["x"],
        ["x"],
        normalizer,
    ).build(coarse_shape, downscale_factor)
    torch.save(model.get_state(), tmp_path / "test.ckpt")

    # normalization should be loaded into model config when get_state called,
    # delete netcdfs to check that data is dumped and loaded with checkpoint
    os.remove(tmp_path / "means.nc")
    os.remove(tmp_path / "stds.nc")

    model_from_disk = Model.from_state(
        torch.load(tmp_path / "test.ckpt", weights_only=False),
    )

    assert model_from_disk.normalizer.fine.means == {"x": 0}
    assert model_from_disk.normalizer.fine.stds == {"x": 1}
    assert model_from_disk.normalizer.coarse.means == {"x": 0}
    assert model_from_disk.normalizer.coarse.stds == {"x": 1}


# TODO: it's a pain to write for Downscaling and Diffusion models
#       should find a way to consolidate
@pytest.mark.parametrize("model_config", ["deterministic", "diffusion"])
def test_model_error_cases(model_config):
    fine_shape = (8, 16)
    coarse_shape = (4, 8)
    upscaling_factor = 2
    batch_size = 3
    normalization_config = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )

    model_class: type[DownscalingModelConfig] | type[DiffusionModelConfig]
    selector: type[ModuleRegistrySelector] | type[DiffusionModuleRegistrySelector]
    if model_config == "deterministic":
        model_class = DownscalingModelConfig
        selector = ModuleRegistrySelector
        extra_kwargs = {}
    else:
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
        model.generate_on_batch(batch)


def test_DiffusionModel_generate_on_batch_no_target():
    fine_shape = (32, 32)
    coarse_shape = (16, 16)
    downscale_factor = 2
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )
    n_steps = 3
    model = DiffusionModelConfig(
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
        num_diffusion_generation_steps=n_steps,
        predict_residual=False,
        use_fine_topography=True,
    ).build(coarse_shape, downscale_factor)

    batch_size = 2
    topography = torch.ones(batch_size, *fine_shape, device=get_device())

    n_generated_samples = 2

    coarse_batch = get_mock_batch([batch_size, *coarse_shape])
    samples = model.generate_on_batch_no_target(
        coarse_batch.data,
        fine_topography=topography,
        n_samples=n_generated_samples,
    )

    assert samples["x"].shape == (
        batch_size,
        n_generated_samples,
        *fine_shape,
    )
