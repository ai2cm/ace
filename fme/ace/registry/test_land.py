import dataclasses
import datetime

import numpy as np
import pytest
import torch

from fme.ace.stepper.single_module import StepperConfig
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import StepLossConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector

TIMESTEP = datetime.timedelta(hours=6)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((8, 16)),
    ],
)
def test_landnet_init(shape):
    hidden_dims = [64, 64]
    network_type = "MLP"
    use_positional_embedding = False

    area = torch.ones((1, 16, 32)).to(get_device())
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    ).to(get_device())
    stepper_config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="LandNet",
                        config={
                            "hidden_dims": hidden_dims,
                            "network_type": network_type,
                            "use_positional_embedding": use_positional_embedding,
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
        loss=StepLossConfig(type="MSE", weights={"temperature": 1.0}),
    )
    stepper = stepper_config.get_stepper(
        dataset_info=DatasetInfo(
            img_shape=shape,
            gridded_operations=LatLonOperations(area),
            vertical_coordinate=vertical_coordinate,
            timestep=TIMESTEP,
        ),
    )

    assert len(stepper.modules) == 1
    assert len(stepper.modules[0].module.layers) == len(hidden_dims) + 1
