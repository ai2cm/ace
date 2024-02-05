from typing import Tuple

import pytest
import torch

from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.downscaling.losses import LossConfig
from fme.downscaling.models import DownscalingModelConfig, Model
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
    highres_shape = (16, 32)  # larger size is need for piq
    upscaling_factor = 2
    module = LinearUpscaling(upscaling_factor=2, img_shape=highres_shape)
    normalizer = HighResLowResPair[StandardNormalizer](
        StandardNormalizer({"x": torch.tensor(0.0)}, {"x": torch.tensor(1.0)}),
        StandardNormalizer({"x": torch.tensor(0.0)}, {"x": torch.tensor(1.0)}),
    )
    batch_size = 3
    model = Model(module, normalizer, torch.nn.MSELoss(), ["x"], ["x"])
    batch = HighResLowResPair(
        {"x": torch.ones(batch_size, highres_shape[0], highres_shape[1])},
        {
            "x": torch.ones(
                batch_size,
                highres_shape[0] // upscaling_factor,
                highres_shape[1] // upscaling_factor,
            )
        },
    )
    if use_opt:
        optimization = OptimizationConfig().build(module.parameters(), 2)
    else:
        optimization = NullOptimization()

    outputs = model.run_on_batch(batch, optimization)
    assert outputs.prediction.keys() == outputs.target.keys()
    for k in outputs.prediction:
        assert outputs.prediction[k].shape == outputs.target[k].shape


def test_build_downscaling_model_config_runs():
    normalizer = HighResLowResPair[NormalizationConfig](
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )

    model_config = DownscalingModelConfig(
        ModuleRegistrySelector("prebuilt", {"module": LinearUpscaling(4, (4, 8))}),
        LossConfig("L1Loss"),
        ["x"],
        ["x"],
        normalizer,
    )

    img_shape, upscale_factor = (4, 8), 4
    model_config.build(img_shape, upscale_factor)
