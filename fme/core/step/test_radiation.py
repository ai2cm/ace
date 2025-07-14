import datetime
from typing import Literal

import pytest
import torch

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.typing_ import TensorDict

from .radiation import SeparateRadiationStepConfig

MODULE_TYPES = Literal["AddOne", "SphericalFourierNeuralOperatorNet"]
IMAGE_SHAPE = (4, 2)
MAIN_PROGNOSTIC_NAMES = ["prog_a", "prog_b"]
SHARED_FORCING_NAMES = ["forcing_shared"]
RADIATION_FORCING_NAMES = ["forcing_rad"]
MAIN_DIAGNOSTIC_NAMES = ["diagnostic_main"]
RADIATION_DIAGNOSTIC_NAMES = ["diagnostic_rad"]


class AddOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def get_tensor_dict(
    names: list[str], img_shape: tuple[int, int], n_samples: int
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


def get_network_and_loss_normalization_config(
    names: list[str],
    norm_mean: float = 0.0,
) -> NetworkAndLossNormalizationConfig:
    return NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={name: norm_mean for name in names},
            stds={name: 1.0 for name in names},
        ),
    )


def get_separate_radiation_config(
    module_name: MODULE_TYPES = "SphericalFourierNeuralOperatorNet",
    norm_mean: float = 0.0,
    step_config_kwargs: dict | None = None,
) -> SeparateRadiationStepConfig:
    if step_config_kwargs is None:
        step_config_kwargs = {}

    names = (
        MAIN_PROGNOSTIC_NAMES
        + SHARED_FORCING_NAMES
        + RADIATION_FORCING_NAMES
        + RADIATION_DIAGNOSTIC_NAMES
        + MAIN_DIAGNOSTIC_NAMES
    )
    normalization = get_network_and_loss_normalization_config(
        names=names,
        norm_mean=norm_mean,
    )
    if module_name == "SphericalFourierNeuralOperatorNet":
        config = {
            "scale_factor": 1,
            "embed_dim": 4,
            "num_layers": 2,
        }
        builder = ModuleSelector(type=module_name, config=config)
        radiation_builder = ModuleSelector(type=module_name, config=config)
    elif module_name == "AddOne":
        config = {"module": AddOne()}
        builder = ModuleSelector(type="prebuilt", config=config)
        radiation_builder = ModuleSelector(type="prebuilt", config=config)
    else:
        raise ValueError(f"module_name {module_name!r} is not supported for this test.")

    return SeparateRadiationStepConfig(
        builder=builder,
        radiation_builder=radiation_builder,
        main_prognostic_names=MAIN_PROGNOSTIC_NAMES,
        shared_forcing_names=SHARED_FORCING_NAMES,
        radiation_only_forcing_names=RADIATION_FORCING_NAMES,
        radiation_diagnostic_names=RADIATION_DIAGNOSTIC_NAMES,
        main_diagnostic_names=MAIN_DIAGNOSTIC_NAMES,
        normalization=normalization,
        **step_config_kwargs,
    )


def get_separate_radiation_step(
    module_name: MODULE_TYPES = "SphericalFourierNeuralOperatorNet",
    norm_mean: float = 0.0,
    step_config_kwargs: dict | None = None,
):
    config = get_separate_radiation_config(module_name, norm_mean, step_config_kwargs)
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(IMAGE_SHAPE[0]), lon=torch.zeros(IMAGE_SHAPE[1])
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=datetime.timedelta(hours=6),
    )
    return config.get_step(dataset_info, lambda x: None)


@pytest.mark.parametrize("detach_radiation", [True, False])
def test_detach_radiation(detach_radiation: bool):
    step_config_kwargs = {"detach_radiation": detach_radiation}
    step = get_separate_radiation_step(step_config_kwargs=step_config_kwargs)
    input_data = get_tensor_dict(
        names=step.input_names,
        img_shape=IMAGE_SHAPE,
        n_samples=1,
    )
    input_data["forcing_rad"].requires_grad = True
    output_data = step.step(input_data, input_data)
    for name, value in output_data.items():
        assert value.requires_grad, f"{name} should require grad"
    grad = torch.autograd.grad(
        outputs=output_data["diagnostic_rad"].sum(),
        inputs=input_data["forcing_rad"],
        allow_unused=True,
    )[0]
    assert grad is not None
    # have to call again as torch.autograd.grad frees the graph
    output_data = step.step(input_data, input_data)
    grad = torch.autograd.grad(
        outputs=output_data["diagnostic_main"].sum(),
        inputs=input_data["forcing_rad"],
        allow_unused=True,
    )[0]
    if detach_radiation:
        assert grad is None
    else:
        assert grad is not None


@pytest.mark.parametrize("residual_prediction", [False, True])
def test_residual_prediction(residual_prediction: bool):
    norm_mean = 2.0
    step_config_kwargs = {"residual_prediction": residual_prediction}
    step = get_separate_radiation_step(
        module_name="AddOne",
        norm_mean=norm_mean,
        step_config_kwargs=step_config_kwargs,
    )
    input_data = get_tensor_dict(
        names=step.input_names,
        img_shape=IMAGE_SHAPE,
        n_samples=1,
    )
    output = step.step(input_data, {})

    for name in MAIN_PROGNOSTIC_NAMES:
        if residual_prediction:
            expected_a_output = 2 * input_data[name] + 1 - norm_mean
        else:
            expected_a_output = input_data[name] + 1
        torch.testing.assert_close(output[name], expected_a_output)

    assert not set(SHARED_FORCING_NAMES).intersection(set(output))
    assert not set(RADIATION_FORCING_NAMES).intersection(set(output))
