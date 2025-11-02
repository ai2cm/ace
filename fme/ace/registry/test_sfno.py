import dataclasses
import datetime

import numpy as np
import pytest
import torch

from fme.ace.stepper.single_module import StepperConfig
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector

TIMESTEP = datetime.timedelta(hours=6)


@pytest.mark.parametrize("shape", [pytest.param((8, 16))])
def test_sfno_init(shape):
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(shape[0]), lon=torch.zeros(shape[1])
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    ).to(get_device())
    num_layers = 2
    stepper_config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="SphericalFourierNeuralOperatorNet",
                        config={
                            "num_layers": num_layers,
                            "embed_dim": 3,
                            "scale_factor": 1,
                        },
                    ),
                    in_names=["x"],
                    out_names=["x"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={"x": float(np.random.randn(1).item())},
                            stds={"x": float(np.random.randn(1).item())},
                        ),
                    ),
                ),
            ),
        ),
    )
    stepper = stepper_config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=horizontal_coordinate,
            vertical_coordinate=vertical_coordinate,
            timestep=TIMESTEP,
        ),
    )
    assert len(stepper.modules) == 1
    assert len(stepper.modules[0].module.blocks) == num_layers
