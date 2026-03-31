import pathlib
from unittest.mock import Mock

import pytest
import torch

from fme.ace.stepper.parameter_init import ParameterInitializationConfig
from fme.core.coordinates import NullVerticalCoordinate
from fme.core.loss import StepLossConfig
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.scheduler import SchedulerConfig
from fme.coupled.data_loading.batch_data import CoupledPrognosticState
from fme.coupled.loss import LossContributionsConfig

from .data_loading.data_typing import CoupledVerticalCoordinate
from .stepper import (
    ComponentTrainingConfig,
    CoupledParameterInitConfig,
    CoupledTrainStepperConfig,
)
from .test_stepper import (
    CoupledDatasetInfoBuilder,
    get_stepper_config,
    get_train_stepper_and_batch,
)


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

    train_stepper, coupled_data, _, _ = get_train_stepper_and_batch(
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
    coupler = train_stepper._stepper

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
    _ = train_stepper.train_on_batch(
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

    _ = train_stepper.train_on_batch(
        data=coupled_data.data,
        optimization=optim,
    )

    assert atmos_module.mock_caller.call_count == 4
    assert ocean_module.mock_caller.call_count == 2


@pytest.mark.parametrize("from_coupled_stepper_state", [True, False])
def test_stepper_parameter_init_integration(
    tmp_path: pathlib.Path, from_coupled_stepper_state: bool, very_fast_only: bool
):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

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
    )
    train_stepper_config = CoupledTrainStepperConfig(
        ocean=ComponentTrainingConfig(
            loss=StepLossConfig(type="MSE"),
            parameter_init=ParameterInitializationConfig(weights_path=ocean_path),
        ),
        atmosphere=ComponentTrainingConfig(
            loss=StepLossConfig(type="MSE"),
            parameter_init=ParameterInitializationConfig(weights_path=atmos_path),
        ),
        parameter_init=CoupledParameterInitConfig(checkpoint_path=ckpt_path),
    )
    coupled_train_stepper = train_stepper_config.get_train_stepper(config, dataset_info)
    coupled_ocean_state = coupled_train_stepper.ocean.modules.state_dict()
    coupled_atmos_state = coupled_train_stepper.atmosphere.modules.state_dict()
    for name, param in ocean_state.items():
        torch.testing.assert_close(param, coupled_ocean_state[name])
    for name, param in atmos_state.items():
        torch.testing.assert_close(param, coupled_atmos_state[name])


class _LearnableAddOne(torch.nn.Module):
    """AddOne with a learnable parameter so outputs track the graph."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale + 1


class _LearnableTimesTwo(torch.nn.Module):
    """TimesTwo with a learnable parameter so outputs track the graph."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return x * self.scale


def _build_optimization(parameters, use_gradient_accumulation=False):
    return Optimization(
        parameters=list(parameters),
        optimizer_type="Adam",
        lr=1e-3,
        max_epochs=1,
        scheduler=SchedulerConfig(),
        enable_automatic_mixed_precision=False,
        kwargs={},
        use_gradient_accumulation=use_gradient_accumulation,
    )


def _get_initial_condition(train_stepper, data):
    stepper = train_stepper._stepper
    return CoupledPrognosticState(
        atmosphere_data=data.atmosphere_data.get_start(
            stepper.atmosphere.prognostic_names, stepper.n_ic_timesteps
        ),
        ocean_data=data.ocean_data.get_start(
            stepper.ocean.prognostic_names, stepper.n_ic_timesteps
        ),
    )


def _build_train_stepper_and_data(atmos_n_steps):
    train_stepper_config = CoupledTrainStepperConfig(
        ocean=ComponentTrainingConfig(loss=StepLossConfig(type="MSE")),
        atmosphere=ComponentTrainingConfig(
            loss=StepLossConfig(type="MSE"),
            loss_contributions=LossContributionsConfig(n_steps=atmos_n_steps),
        ),
    )
    # 2 ocean steps, 4 atmos steps (inner_steps=2)
    return get_train_stepper_and_batch(
        train_stepper_config=train_stepper_config,
        ocean_in_names=["sst", "mask_0"],
        ocean_out_names=["sst"],
        atmosphere_in_names=["surface_temperature", "ocean_fraction"],
        atmosphere_out_names=["surface_temperature"],
        n_forward_times_ocean=2,
        n_forward_times_atmosphere=4,
        n_samples=1,
        atmosphere_builder=ModuleSelector(
            type="prebuilt", config={"module": _LearnableAddOne()}
        ),
        ocean_builder=ModuleSelector(
            type="prebuilt", config={"module": _LearnableTimesTwo()}
        ),
    )


@pytest.mark.parametrize("atmos_n_steps", [1, 2])
def test_unoptimized_steps_detached(atmos_n_steps):
    """Steps beyond loss_contributions.n_steps should not require grad
    (produced under torch.no_grad) while optimized steps should."""
    train_stepper, coupled_data, _, _ = _build_train_stepper_and_data(atmos_n_steps)
    data = coupled_data.data

    ic = _get_initial_condition(train_stepper, data)
    generator = train_stepper._stepper.get_prediction_generator(
        ic,
        data,
        NullOptimization(),
        step_is_optimized=train_stepper._loss.step_is_optimized,
    )
    for step in generator:
        has_grad = any(v.requires_grad for v in step.data.values())
        if step.realm == "atmosphere":
            if step.step < atmos_n_steps:
                assert has_grad, f"atmosphere step {step.step} should require grad"
            else:
                assert (
                    not has_grad
                ), f"atmosphere step {step.step} should not require grad"
        else:
            assert has_grad, f"ocean step {step.step} should require grad"


@pytest.mark.parametrize("atmos_n_steps", [1, 2])
def test_unoptimized_steps_detached_with_gradient_accumulation(atmos_n_steps):
    """Optimized steps should require grad even when gradient accumulation
    detaches tensors between steps."""
    train_stepper, coupled_data, _, _ = _build_train_stepper_and_data(atmos_n_steps)
    data = coupled_data.data

    optimization = _build_optimization(
        train_stepper.modules.parameters(), use_gradient_accumulation=True
    )
    optimization.set_mode(train_stepper.modules)

    ic = _get_initial_condition(train_stepper, data)
    with optimization.autocast():
        generator = train_stepper._stepper.get_prediction_generator(
            ic,
            data,
            optimization,
            step_is_optimized=train_stepper._loss.step_is_optimized,
        )
        for step in generator:
            has_grad = any(v.requires_grad for v in step.data.values())
            if step.realm == "atmosphere":
                if step.step < atmos_n_steps:
                    assert has_grad, (
                        f"atmosphere step {step.step} should require grad "
                        "even with gradient accumulation"
                    )
                else:
                    assert (
                        not has_grad
                    ), f"atmosphere step {step.step} should not require grad"
            else:
                assert has_grad, (
                    f"ocean step {step.step} should require grad "
                    "even with gradient accumulation"
                )


@pytest.mark.parametrize("atmos_n_steps", [1, 2])
def test_unoptimized_steps_loss_metrics(atmos_n_steps):
    """Only optimized steps should produce loss metrics."""
    train_stepper, coupled_data, _, _ = _build_train_stepper_and_data(atmos_n_steps)
    result = train_stepper.train_on_batch(
        data=coupled_data.data,
        optimization=NullOptimization(),
    )
    for i in range(atmos_n_steps):
        assert f"loss/atmosphere_step_{i}" in result.atmosphere.metrics
    for i in range(atmos_n_steps, 4):
        assert f"loss/atmosphere_step_{i}" not in result.atmosphere.metrics
    assert "loss/ocean_step_0" in result.ocean.metrics
    assert "loss/ocean_step_1" in result.ocean.metrics
