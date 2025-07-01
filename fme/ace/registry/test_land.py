import dataclasses
import datetime

import numpy as np
import pytest
import torch

from fme.ace.stepper import SingleModuleStepperConfig
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.normalizer import NormalizationConfig

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

    landnet_config_data = {
        "type": "LandNet",
        "config": {
            "hidden_dims": hidden_dims,
            "network_type": network_type,
            "use_positional_embedding": use_positional_embedding,
        },
    }
    stepper_config_data = {
        "builder": landnet_config_data,
        "in_names": ["x"],
        "out_names": ["x"],
        "normalization": dataclasses.asdict(
            NormalizationConfig(
                means={"x": float(np.random.randn(1).item())},
                stds={"x": float(np.random.randn(1).item())},
            )
        ),
    }
    area = torch.ones((1, 16, 32)).to(get_device())
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    ).to(get_device())
    stepper_config = SingleModuleStepperConfig.from_state(stepper_config_data)
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
