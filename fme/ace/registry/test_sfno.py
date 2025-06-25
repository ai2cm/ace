import dataclasses
import datetime

import numpy as np
import pytest
import torch

from fme.ace.stepper import SingleModuleStepperConfig
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.normalizer import NormalizationConfig

TIMESTEP = datetime.timedelta(hours=6)


@pytest.mark.parametrize("shape", [pytest.param((8, 16))])
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
        "normalization": dataclasses.asdict(
            NormalizationConfig(
                means={"x": float(np.random.randn(1).item())},
                stds={"x": float(np.random.randn(1).item())},
            )
        ),
    }
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(shape[0]), lon=torch.zeros(shape[1])
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    ).to(get_device())
    stepper_config = SingleModuleStepperConfig.from_state(stepper_config_data)
    stepper = stepper_config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=horizontal_coordinate,
            vertical_coordinate=vertical_coordinate,
            timestep=TIMESTEP,
        ),
    )
    assert len(stepper.modules) == 1
    assert len(stepper.modules[0].module.blocks) == num_layers
