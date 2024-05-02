import numpy as np
import pytest
import torch

from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.device import get_device
from fme.core.normalizer import FromStateNormalizer
from fme.core.stepper import SingleModuleStepperConfig


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((8, 16)),
    ],
)
def test_sfno_init(shape):
    num_layers = 2
    sfno_config_data = {
        "type": "SphericalFourierNeuralOperatorNet",
        "config": {
            "num_layers": num_layers,
            "embed_dim": 3,
            "scale_factor": 1,
        },
    }
    stepper_config_data = {
        "builder": sfno_config_data,
        "in_names": ["x"],
        "out_names": ["x"],
        "normalization": FromStateNormalizer(
            state={
                "means": {"x": np.random.randn(1)},
                "stds": {"x": np.random.randn(1)},
            }
        ),
    }
    area = torch.ones((1, 16, 32)).to(get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7)).to(
        get_device()
    )
    stepper_config = SingleModuleStepperConfig.from_state(stepper_config_data)
    stepper = stepper_config.get_stepper(
        img_shape=shape,
        area=area,
        sigma_coordinates=sigma_coordinates,
    )
    assert len(stepper.module.module.blocks) == num_layers
