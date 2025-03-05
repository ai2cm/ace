import datetime
from typing import List, Tuple

import pytest
import torch

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.gridded_ops import LatLonOperations
from fme.core.normalizer import NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.step import StepConfigABC
from fme.core.typing_ import TensorDict

from .radiation import SeparateRadiationStepConfig

CONFIG_CASES = [
    pytest.param(
        SeparateRadiationStepConfig(
            builder=ModuleSelector(
                type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
            ),
            radiation_builder=ModuleSelector(
                type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
            ),
            in_names=["a", "b", "e"],
            out_names=["b", "c"],
            radiation_in_names=["d", "e"],
            radiation_out_names=["e", "f"],
            normalization=NormalizationConfig(
                means={"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0, "f": 0.0},
                stds={"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0, "f": 1.0},
            ),
        ),
        id="separate_radiation",
    ),
]
TIMESTEP = datetime.timedelta(hours=6)


def get_tensor_dict(
    names: List[str], img_shape: Tuple[int, ...], n_samples: int
) -> TensorDict:
    data_dict = {}
    device = fme.get_device()
    for name in names:
        data_dict[name] = torch.rand(
            n_samples,
            *img_shape,
            device=device,
        )
    return data_dict


@pytest.mark.parametrize("config", CONFIG_CASES)
def test_reloaded_step_gives_same_prediction(config: StepConfigABC):
    device = fme.get_device()
    torch.manual_seed(0)

    img_shape = (5, 5)
    n_samples = 5
    area = torch.ones(img_shape, device=device)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    step = config.get_step(
        img_shape=img_shape,
        gridded_operations=LatLonOperations(area),
        vertical_coordinate=vertical_coordinate,
        timestep=TIMESTEP,
    )
    new_step = step.__class__.from_state(step.get_state())
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_forcing_data = get_tensor_dict(
        step.next_step_forcing_names, img_shape, n_samples
    )
    first_result = step.step(input_data, next_step_forcing_data)
    second_result = new_step.step(input_data, next_step_forcing_data)
    assert set(first_result.keys()) == set(step.output_names)
    assert set(second_result.keys()) == set(step.output_names)
    for k in first_result:
        torch.testing.assert_close(first_result[k], second_result[k])
