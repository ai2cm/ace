import pathlib
from unittest.mock import Mock

import pytest
import torch

import fme
from fme.ace.stepper.parameter_init import ParameterInitializationConfig
from fme.core.coordinates import NullVerticalCoordinate
from fme.core.optimization import OptimizationConfig
from fme.core.registry.module import ModuleSelector

from .data_loading.data_typing import CoupledVerticalCoordinate
from .test_stepper import (
    CoupledDatasetInfoBuilder,
    get_stepper_and_batch,
    get_stepper_config,
)

DEVICE = fme.get_device()


class AddBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x + self.bias


def test_stepper_gradient_accumulation_integration():
    ocean_in_names = ["o_prog", "o_sfc_temp", "o_mask", "a_diag1"]
    ocean_out_names = ["o_prog", "o_sfc_temp", "o_diag1", "o_diag2"]
    atmos_in_names = ["a_prog1", "a_prog2", "a_sfc_temp", "ocean_frac", "o_prog"]
    atmos_out_names = ["a_prog1", "a_prog2", "a_sfc_temp", "a_diag1", "a_diag2"]

    coupler, coupled_data = get_stepper_and_batch(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_in_names,
        atmosphere_out_names=atmos_out_names,
        n_forward_times_ocean=2,
        n_forward_times_atmosphere=4,
        n_samples=3,
        sst_name_in_ocean_data="o_sfc_temp",
        sfc_temp_name_in_atmosphere_data="a_sfc_temp",
        ocean_fraction_name="ocean_frac",
        ocean_builder=ModuleSelector(type="prebuilt", config={"module": AddBias()}),
        atmosphere_builder=ModuleSelector(
            type="prebuilt", config={"module": AddBias()}
        ),
    )

    assert len(coupler.atmosphere.modules) == 1
    assert len(coupler.ocean.modules) == 1

    atmos_module = coupler.atmosphere.modules[0].module
    ocean_module = coupler.ocean.modules[0].module
    atmos_module.mock_caller = Mock()
    ocean_module.mock_caller = Mock()

    def atmos_hook_v1(module, grad_input, grad_output):
        module.mock_caller()
        assert len(grad_input) == 1
        if module.mock_caller.call_count == 4:
            # 4th backprop is 1st forward step
            assert grad_input[0] is None
        else:
            assert grad_input[0].shape == (3, 5, 5, 5)
        assert len(grad_output) == 1
        assert grad_output[0].shape == (3, 5, 5, 5)

    def ocean_hook_v1(module, grad_input, grad_output):
        module.mock_caller()
        assert len(grad_input) == 1
        # ocean always has atmos inputs that require grad
        assert grad_input[0].shape == (3, 4, 5, 5)
        assert len(grad_output) == 1
        assert grad_output[0].shape == (3, 4, 5, 5)

    atmos_handle = atmos_module.register_full_backward_hook(atmos_hook_v1)
    ocean_handle = ocean_module.register_full_backward_hook(ocean_hook_v1)

    # without gradient accumulation, atmos steps not detached
    optim = OptimizationConfig(use_gradient_accumulation=False).build(
        coupler.modules, 1
    )
    _ = coupler.train_on_batch(
        data=coupled_data.data,
        optimization=optim,
    )

    # 4 atmos steps, 2 ocean steps
    assert atmos_module.mock_caller.call_count == 4
    assert ocean_module.mock_caller.call_count == 2

    atmos_handle.remove()
    ocean_handle.remove()

    atmos_module.mock_caller.reset_mock()
    ocean_module.mock_caller.reset_mock()

    def atmos_hook_v2(module, grad_input, grad_output):
        module.mock_caller()

    def ocean_hook_v2(module, grad_input, grad_output):
        module.mock_caller()
        assert len(grad_input) == 1
        # ocean never has inputs that require grad
        assert grad_input[0] is None
        assert len(grad_output) == 1
        assert grad_output[0].shape == (3, 4, 5, 5)

    _ = atmos_module.register_full_backward_hook(atmos_hook_v2)
    _ = ocean_module.register_full_backward_hook(ocean_hook_v2)

    # with gradient accumulation, atmos steps detached
    optim = OptimizationConfig(use_gradient_accumulation=True).build(coupler.modules, 1)

    _ = coupler.train_on_batch(
        data=coupled_data.data,
        optimization=optim,
    )

    assert atmos_module.mock_caller.call_count == 4
    assert ocean_module.mock_caller.call_count == 2


@pytest.mark.parametrize("from_coupled_stepper_state", [True, False])
def test_stepper_parameter_init_integration(
    tmp_path: pathlib.Path, from_coupled_stepper_state: bool
):
    ocean_in_names = ["o_prog", "o_sfc_temp", "o_mask", "a_diag1"]
    ocean_out_names = ["o_prog", "o_sfc_temp", "o_diag1", "o_diag2"]
    atmos_in_names = ["a_prog1", "a_prog2", "a_sfc_temp", "ocean_frac", "o_prog"]
    atmos_out_names = ["a_prog1", "a_prog2", "a_sfc_temp", "a_diag1", "a_diag2"]
    vcoord = CoupledVerticalCoordinate(
        ocean=NullVerticalCoordinate(),
        atmosphere=NullVerticalCoordinate(),
    )
    dataset_info = CoupledDatasetInfoBuilder(vcoord=vcoord).dataset_info
    config = get_stepper_config(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_in_names,
        atmosphere_out_names=atmos_out_names,
        sst_name_in_ocean_data="o_sfc_temp",
        sfc_temp_name_in_atmosphere_data="a_sfc_temp",
        ocean_fraction_name="ocean_frac",
        ocean_timedelta="2D",
        atmosphere_timedelta="1D",
        ocean_builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
        ),
        atmosphere_builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
        ),
    )
    if from_coupled_stepper_state:
        # mimic CoupledStepper pretraining
        pretrained_coupled_stepper = config.get_stepper(dataset_info)
        ckpt_path = str(tmp_path / "ckpt.pt")
        ocean_path = None
        atmos_path = None
        torch.save({"stepper": pretrained_coupled_stepper.get_state()}, ckpt_path)
        ocean_state = pretrained_coupled_stepper.ocean.modules.state_dict()
        atmos_state = pretrained_coupled_stepper.atmosphere.modules.state_dict()
    else:
        # mimic separate Stepper pretraining runs
        ocean_stepper = config.ocean.stepper.get_stepper(dataset_info.ocean)
        atmos_stepper = config.atmosphere.stepper.get_stepper(dataset_info.atmosphere)
        ckpt_path = None
        ocean_path = str(tmp_path / "ocean.pt")
        atmos_path = str(tmp_path / "atmos.pt")
        torch.save({"stepper": ocean_stepper.get_state()}, ocean_path)
        torch.save({"stepper": atmos_stepper.get_state()}, atmos_path)
        ocean_state = ocean_stepper.modules.state_dict()
        atmos_state = atmos_stepper.modules.state_dict()
    config = get_stepper_config(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_in_names,
        atmosphere_out_names=atmos_out_names,
        sst_name_in_ocean_data="o_sfc_temp",
        sfc_temp_name_in_atmosphere_data="a_sfc_temp",
        ocean_fraction_name="ocean_frac",
        ocean_timedelta="2D",
        atmosphere_timedelta="1D",
        ocean_builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
        ),
        atmosphere_builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
        ),
        ocean_parameter_init=ParameterInitializationConfig(weights_path=ocean_path),
        atmosphere_parameter_init=ParameterInitializationConfig(
            weights_path=atmos_path
        ),
        checkpoint_path=ckpt_path,
    )
    coupled_stepper = config.get_stepper(dataset_info)
    coupled_ocean_state = coupled_stepper.ocean.modules.state_dict()
    coupled_atmos_state = coupled_stepper.atmosphere.modules.state_dict()
    for name, param in ocean_state.items():
        torch.testing.assert_close(param, coupled_ocean_state[name])
    for name, param in atmos_state.items():
        torch.testing.assert_close(param, coupled_atmos_state[name])
