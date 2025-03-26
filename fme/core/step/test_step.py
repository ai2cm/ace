import dataclasses
import datetime
import unittest
import unittest.mock
from typing import List, Tuple

import dacite
import pytest
import torch

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import LatLonOperations
from fme.core.multi_call import MultiCallConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.multi_call import MultiCallStepConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepABC, StepSelector
from fme.core.typing_ import TensorDict

from .radiation import SeparateRadiationStepConfig

SEPARATE_RADIATION_CONFIG = SeparateRadiationStepConfig(
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
    normalization=NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
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
)

SELECTOR_CONFIG_CASES = [
    pytest.param(
        StepSelector(
            type="separate_radiation",
            config=dataclasses.asdict(SEPARATE_RADIATION_CONFIG),
        ),
        id="separate_radiation",
    ),
    pytest.param(
        StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="SphericalFourierNeuralOperatorNet",
                        config={
                            "scale_factor": 1,
                            "embed_dim": 4,
                            "num_layers": 2,
                        },
                    ),
                    in_names=["forcing_shared", "forcing_rad"],
                    out_names=["diagnostic_main", "diagnostic_rad"],
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={
                                "forcing_shared": 0.0,
                                "forcing_rad": 0.0,
                                "diagnostic_main": 0.0,
                                "diagnostic_rad": 0.0,
                            },
                            stds={
                                "forcing_shared": 1.0,
                                "forcing_rad": 1.0,
                                "diagnostic_main": 1.0,
                                "diagnostic_rad": 1.0,
                            },
                        ),
                    ),
                ),
            ),
        ),
        id="single_module",
    ),
    pytest.param(
        StepSelector(
            type="multi_call",
            config=dataclasses.asdict(
                MultiCallStepConfig(
                    wrapped_step=StepSelector(
                        type="separate_radiation",
                        config=dataclasses.asdict(SEPARATE_RADIATION_CONFIG),
                    ),
                    config=MultiCallConfig(
                        forcing_name="forcing_rad",
                        forcing_multipliers={"double": 2.0},
                        output_names=["diagnostic_rad"],
                    ),
                ),
            ),
        ),
        id="multi_call_separate_radiation",
    ),
]

HAS_NEXT_STEP_FORCING_NAME_CASES = [
    pytest.param(
        StepSelector(
            type="separate_radiation",
            config=dataclasses.asdict(SEPARATE_RADIATION_CONFIG),
        ),
        id="multi_call_separate_radiation",
    ),
]

HAS_NEXT_STEP_FORCING_NAME_CASES = [
    pytest.param(
        StepSelector(
            type="separate_radiation",
            config=dataclasses.asdict(SEPARATE_RADIATION_CONFIG),
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


def get_step(selector: StepSelector, img_shape: Tuple[int, int]) -> StepABC:
    device = fme.get_device()
    area = torch.ones(img_shape, device=device)
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    dataset_info = DatasetInfo(
        img_shape=img_shape,
        gridded_operations=LatLonOperations(area),
        vertical_coordinate=vertical_coordinate,
        timestep=TIMESTEP,
    )
    return selector.get_step(dataset_info)


@pytest.mark.parametrize("config", HAS_NEXT_STEP_FORCING_NAME_CASES)
def test_next_step_forcing_names_is_forcing(config: StepSelector):
    data = dataclasses.asdict(config)
    img_shape = (5, 5)
    step = get_step(config, img_shape)
    forcing_names = set(step.input_names).difference(step.output_names)
    data["config"]["next_step_forcing_names"] = [list(forcing_names)[0]]
    dacite.from_dict(config.__class__, data, config=dacite.Config(strict=True))


@pytest.mark.parametrize("config", HAS_NEXT_STEP_FORCING_NAME_CASES)
def test_next_step_forcing_names_is_prognostic(config: StepSelector):
    data = dataclasses.asdict(config)
    img_shape = (5, 5)
    step = get_step(config, img_shape)
    prognostic_names = set(step.output_names).intersection(step.input_names)
    name = list(prognostic_names)[0]
    data["config"]["next_step_forcing_names"] = [name]
    with pytest.raises(ValueError) as err:
        dacite.from_dict(config.__class__, data, config=dacite.Config(strict=True))
    assert "next_step_forcing_name" in str(err.value)
    assert name in str(err.value)


@pytest.mark.parametrize("config", HAS_NEXT_STEP_FORCING_NAME_CASES)
def test_next_step_forcing_names_is_diagnostic(config: StepSelector):
    data = dataclasses.asdict(config)
    img_shape = (5, 5)
    step = get_step(config, img_shape)
    diagnostic_names = set(step.output_names).difference(step.input_names)
    name = list(diagnostic_names)[0]
    data["config"]["next_step_forcing_names"] = [name]
    with pytest.raises(ValueError) as err:
        dacite.from_dict(config.__class__, data, config=dacite.Config(strict=True))
    assert "next_step_forcing_name" in str(err.value)
    assert name in str(err.value)


@pytest.mark.parametrize("config", SELECTOR_CONFIG_CASES)
def test_step_applies_wrapper(config: StepSelector):
    torch.manual_seed(0)
    img_shape = (5, 5)
    n_samples = 5
    step = get_step(config, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    multi_calls = 1
    if isinstance(config._step_config_instance, MultiCallStepConfig):
        if config._step_config_instance.config is not None:
            multi_calls += len(config._step_config_instance.config.forcing_multipliers)

    wrapper = unittest.mock.MagicMock(side_effect=lambda x: x)
    step.step(input_data, next_step_input_data, wrapper=wrapper)
    assert wrapper.call_count == multi_calls * len(step.modules)
    for module in step.modules:
        wrapper.assert_any_call(module)
