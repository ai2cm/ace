import dataclasses
import datetime
from typing import List, Tuple

import dacite
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
                type="SphericalFourierNeuralOperatorNet",
                config={
                    "scale_factor": 1,
                    "embed_dim": 4,
                    "num_layers": 2,
                },
            ),
            radiation_builder=ModuleSelector(
                type="SphericalFourierNeuralOperatorNet",
                config={
                    "scale_factor": 1,
                    "embed_dim": 4,
                    "num_layers": 2,
                },
            ),
            main_prognostic_names=["prog_a", "prog_b"],
            shared_forcing_names=["forcing_shared"],
            radiation_only_forcing_names=["forcing_rad"],
            radiation_diagnostic_names=["diagnostic_rad"],
            main_diagnostic_names=["diagnostic_main"],
            normalization=NormalizationConfig(
                means={
                    name: 0.0
                    for name in [
                        "prog_a",
                        "prog_b",
                        "forcing_shared",
                        "forcing_rad",
                        "diagnostic_rad",
                        "diagnostic_main",
                    ]
                },
                stds={
                    name: 1.0
                    for name in [
                        "prog_a",
                        "prog_b",
                        "forcing_shared",
                        "forcing_rad",
                        "diagnostic_rad",
                        "diagnostic_main",
                    ]
                },
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


def get_step(config: StepConfigABC, img_shape: Tuple[int, int]):
    device = fme.get_device()
    area = torch.ones(img_shape, device=device)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    return config.get_step(
        img_shape, LatLonOperations(area), vertical_coordinate, TIMESTEP
    )


@pytest.mark.parametrize("config", CONFIG_CASES)
def test_reloaded_step_gives_same_prediction(config: StepConfigABC):
    torch.manual_seed(0)
    img_shape = (5, 5)
    n_samples = 5
    step = get_step(config, img_shape)
    new_step = step.__class__.from_state(step.get_state())
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    first_result = step.step(input_data, next_step_input_data)
    second_result = new_step.step(input_data, next_step_input_data)
    assert set(first_result.keys()) == set(step.output_names)
    assert set(second_result.keys()) == set(step.output_names)
    for k in first_result:
        torch.testing.assert_close(first_result[k], second_result[k])


@pytest.mark.parametrize("config", CONFIG_CASES)
def test_next_step_forcing_names_is_forcing(config: StepConfigABC):
    data = dataclasses.asdict(config)
    img_shape = (5, 5)
    step = get_step(config, img_shape)
    forcing_names = set(step.input_names).difference(step.output_names)
    data["next_step_forcing_names"] = [list(forcing_names)[0]]
    dacite.from_dict(config.__class__, data, config=dacite.Config(strict=True))


@pytest.mark.parametrize("config", CONFIG_CASES)
def test_next_step_forcing_names_is_prognostic(config: StepConfigABC):
    data = dataclasses.asdict(config)
    img_shape = (5, 5)
    step = get_step(config, img_shape)
    prognostic_names = set(step.output_names).intersection(step.input_names)
    name = list(prognostic_names)[0]
    data["next_step_forcing_names"] = [name]
    with pytest.raises(ValueError) as err:
        dacite.from_dict(config.__class__, data, config=dacite.Config(strict=True))
    assert "next_step_forcing_name" in str(err.value)
    assert name in str(err.value)


@pytest.mark.parametrize("config", CONFIG_CASES)
def test_next_step_forcing_names_is_diagnostic(config: StepConfigABC):
    data = dataclasses.asdict(config)
    img_shape = (5, 5)
    step = get_step(config, img_shape)
    diagnostic_names = set(step.output_names).difference(step.input_names)
    name = list(diagnostic_names)[0]
    data["next_step_forcing_names"] = [name]
    with pytest.raises(ValueError) as err:
        dacite.from_dict(config.__class__, data, config=dacite.Config(strict=True))
    assert "next_step_forcing_name" in str(err.value)
    assert name in str(err.value)
