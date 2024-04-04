from typing import Tuple

import pytest
import torch

from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.core.typing_ import TensorMapping
from fme.downscaling.models import (
    DownscalingModelConfig,
    Model,
    PairedNormalizationConfig,
)
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.typing_ import HighResLowResPair


class LinearUpscaling(torch.nn.Module):
    def __init__(self, upscaling_factor: int, img_shape: Tuple[int, int]):
        super(LinearUpscaling, self).__init__()
        self.img_shape = img_shape
        height, width = img_shape
        self.linear = torch.nn.Linear(
            height * width // upscaling_factor**2, height * width, bias=False
        )

    def forward(self, x):
        x = self.linear(torch.flatten(x, start_dim=1))
        x = x.view(x.shape[0], 1, *self.img_shape)
        return x


@pytest.mark.parametrize("use_opt", [True, False])
def test_run_on_batch(use_opt):
    highres_shape = (8, 16)
    lowres_shape = (4, 8)
    upscaling_factor = 2
    normalization_config = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )

    batch_size = 3
    module_selector = ModuleRegistrySelector(
        "prebuilt",
        {
            "module": LinearUpscaling(
                upscaling_factor=upscaling_factor, img_shape=highres_shape
            )
        },
    )
    area_weights = HighResLowResPair(
        torch.ones(*highres_shape), torch.ones(*lowres_shape)
    )
    model = DownscalingModelConfig(
        module_selector, LossConfig(type="MSE"), ["x"], ["x"], normalization_config
    ).build(
        lowres_shape,
        upscaling_factor,
        area_weights,
    )
    batch: HighResLowResPair[TensorMapping] = HighResLowResPair(
        {"x": torch.ones(batch_size, *highres_shape)},
        {"x": torch.ones(batch_size, *lowres_shape)},
    )
    if use_opt:
        optimization = OptimizationConfig().build(model.module.parameters(), 2)
    else:
        optimization = NullOptimization()

    outputs = model.run_on_batch(batch, optimization)
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
        highres=NormalizationConfig(
            means={n: 0.0 for n in out_names}, stds={n: 1.0 for n in out_names}
        ),
        lowres=NormalizationConfig(
            means={n: 0.0 for n in in_names}, stds={n: 1.0 for n in in_names}
        ),
    )

    img_shape, upscale_factor = (4, 8), 4
    area_weights = HighResLowResPair[torch.Tensor](
        torch.ones(img_shape[0] * upscale_factor, img_shape[1] * upscale_factor),
        torch.ones(img_shape[0], img_shape[1]),
    )
    loss = LossConfig(type="L1")
    model_config = DownscalingModelConfig(
        ModuleRegistrySelector("prebuilt", {"module": LinearUpscaling(4, (4, 8))}),
        loss,
        ["x"],
        ["x"],
        normalization,
    )
    model_config.build(img_shape, upscale_factor, area_weights)


def test_serialization(tmp_path):
    highres_shape = (16, 32)
    lowres_shape = (8, 16)
    downscale_factor = 2
    module = LinearUpscaling(upscaling_factor=2, img_shape=highres_shape)
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )
    area_weights = HighResLowResPair(
        torch.ones(*highres_shape), torch.ones(*lowres_shape)
    )
    model = DownscalingModelConfig(
        ModuleRegistrySelector("prebuilt", {"module": module}),
        LossConfig(type="MSE"),
        ["x"],
        ["x"],
        normalizer,
    ).build(lowres_shape, downscale_factor, area_weights)

    batch_size = 3
    batch: HighResLowResPair[TensorMapping] = HighResLowResPair(
        highres={"x": torch.ones(batch_size, *highres_shape)},
        lowres={"x": torch.ones(batch_size, *lowres_shape)},
    )
    expected = model.run_on_batch(batch, NullOptimization()).prediction["x"]

    model_from_state = Model.from_state(model.get_state(), area_weights)
    torch.testing.assert_allclose(
        expected,
        model_from_state.run_on_batch(batch, NullOptimization()).prediction["x"],
    )

    torch.save(model.get_state(), tmp_path / "test.ckpt")
    model_from_disk = Model.from_state(torch.load(tmp_path / "test.ckpt"), area_weights)
    torch.testing.assert_allclose(
        expected,
        model_from_disk.run_on_batch(batch, NullOptimization()).prediction["x"],
    )
