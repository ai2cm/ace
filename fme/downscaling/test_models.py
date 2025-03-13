from typing import Optional, Tuple

import pytest
import torch

from fme import get_device
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import OptimizationConfig
from fme.core.typing_ import TensorMapping
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
        fine_img_shape: Tuple[int, int],
        n_channels: int = 1,
        fine_topography: Optional[torch.Tensor] = None,
    ):
        super(LinearDownscaling, self).__init__()
        self.img_shape = fine_img_shape
        self.n_channels = n_channels
        if fine_topography is not None:
            fine_topography = fine_topography.to(get_device())
        self.fine_topography = fine_topography
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
        if self.fine_topography is not None:
            x = x + self.fine_topography  # arbitrary use of fine_topography
        return x


@pytest.mark.parametrize("use_opt", [True, False])
@pytest.mark.parametrize("use_fine_topography", [True, False])
def test_train_and_generate(use_opt, use_fine_topography):
    fine_shape = (8, 16)
    coarse_shape = (4, 8)
    upscaling_factor = 2
    normalization_config = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )

    fine_topography = torch.zeros(1, *fine_shape)

    batch_size = 3
    module_selector = ModuleRegistrySelector(
        "prebuilt",
        {
            "module": LinearDownscaling(
                factor=upscaling_factor,
                fine_img_shape=fine_shape,
                fine_topography=fine_topography if use_fine_topography else None,
            )
        },
    )
    area_weights = FineResCoarseResPair(
        fine=torch.ones(*fine_shape), coarse=torch.ones(*coarse_shape)
    )
    model = DownscalingModelConfig(
        module_selector,
        LossConfig(type="MSE"),
        ["x"],
        ["x"],
        normalization_config,
        use_fine_topography=False,
    ).build(
        coarse_shape,
        upscaling_factor,
        area_weights,
        fine_topography=fine_topography,
    )
    batch: FineResCoarseResPair[TensorMapping] = FineResCoarseResPair(
        fine={"x": torch.ones(batch_size, *fine_shape)},
        coarse={"x": torch.ones(batch_size, *coarse_shape)},
    )
    if use_opt:
        optimization = OptimizationConfig().build(modules=[model.module], max_epochs=2)
        outputs = model.train_on_batch(batch, optimization)
    else:
        outputs = model.generate_on_batch(batch)

    assert outputs.prediction.keys() == outputs.target.keys()
    for k in outputs.prediction:
        assert outputs.prediction[k].shape == outputs.target[k].unsqueeze(1).shape


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
    area_weights = FineResCoarseResPair[torch.Tensor](
        torch.ones(img_shape[0] * downscale_factor, img_shape[1] * downscale_factor),
        torch.ones(img_shape[0], img_shape[1]),
    )
    loss = LossConfig(type="L1")
    model_config = DownscalingModelConfig(
        ModuleRegistrySelector("prebuilt", {"module": LinearDownscaling(4, (4, 8))}),
        loss,
        ["x"],
        ["x"],
        normalization,
        use_fine_topography=False,
    )
    model_config.build(
        img_shape,
        downscale_factor,
        area_weights,
        torch.zeros(*[s * downscale_factor for s in img_shape]),
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
    area_weights = FineResCoarseResPair(
        torch.ones(*fine_shape), torch.ones(*coarse_shape)
    )
    fine_topography = torch.zeros(*fine_shape)
    model = DownscalingModelConfig(
        ModuleRegistrySelector("prebuilt", {"module": module}),
        LossConfig(type="MSE"),
        ["x"],
        ["x"],
        normalizer,
        use_fine_topography=False,
    ).build(coarse_shape, downscale_factor, area_weights, fine_topography)

    batch_size = 3
    batch: FineResCoarseResPair[TensorMapping] = FineResCoarseResPair(
        fine={"x": torch.ones(batch_size, *fine_shape)},
        coarse={"x": torch.ones(batch_size, *coarse_shape)},
    )
    expected = model.generate_on_batch(batch).prediction["x"]

    model_from_state = Model.from_state(
        model.get_state(), area_weights, fine_topography
    )
    torch.testing.assert_close(
        expected,
        model_from_state.generate_on_batch(batch).prediction["x"],
    )

    torch.save(model.get_state(), tmp_path / "test.ckpt")
    model_from_disk = Model.from_state(
        torch.load(tmp_path / "test.ckpt", weights_only=False),
        area_weights,
        fine_topography,
    )
    torch.testing.assert_close(
        expected,
        model_from_disk.generate_on_batch(batch).prediction["x"],
    )


@pytest.mark.parametrize("predict_residual", [True, False])
def test_diffusion_model_train_and_generate(predict_residual):
    fine_shape = (16, 32)
    coarse_shape = (8, 16)
    downscale_factor = 2
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )
    area_weights = FineResCoarseResPair(
        torch.ones(*fine_shape), torch.ones(*coarse_shape)
    )
    fine_topography = torch.zeros(*fine_shape)

    model = DiffusionModelConfig(
        module=DiffusionModuleRegistrySelector(
            "unet_diffusion_song", {"model_channels": 4}
        ),
        loss=LossConfig(type="MSE"),
        in_names=["x"],
        out_names=["x"],
        normalization=normalizer,
        use_fine_topography=False,
        p_mean=-1.0,
        p_std=1.0,
        sigma_min=0.1,
        sigma_max=1.0,
        churn=0.5,
        num_diffusion_generation_steps=3,
        predict_residual=predict_residual,
    ).build(coarse_shape, downscale_factor, area_weights, fine_topography)

    batch_size = 2
    batch: FineResCoarseResPair[TensorMapping] = FineResCoarseResPair(
        {"x": torch.ones(batch_size, *fine_shape)},
        {"x": torch.ones(batch_size, *coarse_shape)},
    )
    optimization = OptimizationConfig().build(modules=[model.module], max_epochs=2)
    train_outputs = model.train_on_batch(batch, optimization)
    assert torch.allclose(train_outputs.target["x"], batch.fine["x"])

    n_generated_samples = 2
    generated_outputs = [
        model.generate_on_batch(batch) for _ in range(n_generated_samples)
    ]

    for generated_output in generated_outputs:
        assert (
            generated_output.prediction["x"].shape
            == generated_output.target["x"].unsqueeze(1).shape
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
        with_combined_samples, batch_size, n_samples
    )
    assert with_batch_sample_dims.shape == (batch_size, n_samples, 5)
    assert torch.equal(batch, with_batch_sample_dims[:, 0])
