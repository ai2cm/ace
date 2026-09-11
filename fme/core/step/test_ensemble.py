import contextlib
import dataclasses
import itertools
import unittest.mock

import pytest
import torch
from torch import nn

from fme.core.corrector.output import CorrectorDiagnostics
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.ensemble import EnsembleStep, EnsembleStepConfig
from fme.core.step.output import StepOutput
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.stepper_state import StepperState
from fme.core.testing import get_dataset_info, trivial_network_and_loss_normalization

from .test_step_registry import MockStep, MockStepConfig

IMG_SHAPE = (4, 8)
NAMES = ["a", "b"]


class PlusOne(nn.Module):
    def forward(self, x):
        return x + 1


class TimesTwo(nn.Module):
    def forward(self, x):
        return 2 * x


def get_single_module_selector(
    module: nn.Module, residual_prediction: bool = False
) -> StepSelector:
    """A step over the given module with identity normalization, so the step's
    output is the module's output (plus the input, if residual)."""
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(type="prebuilt", config={"module": module}),
                in_names=NAMES,
                out_names=NAMES,
                normalization=trivial_network_and_loss_normalization(NAMES),
                residual_prediction=residual_prediction,
            )
        ),
    )


def get_ensemble_step(
    members: list[StepSelector], weights: list[float]
) -> EnsembleStep:
    config = EnsembleStepConfig(members=members, weights=weights)
    return config.get_step(get_dataset_info(img_shape=IMG_SHAPE), lambda _: None)


def run_step(step: EnsembleStep, input_data: dict[str, torch.Tensor]) -> StepOutput:
    return step.step(
        args=StepArgs(input=input_data, next_step_input_data={}, labels=None)
    )


def get_mock_selector(out_names: list[str]) -> StepSelector:
    return StepSelector(type="mock", config={"in_names": ["a"], "out_names": out_names})


@contextlib.contextmanager
def mock_step_config_properties():
    """Give MockStepConfig the properties the ensemble config compares."""
    with (
        unittest.mock.patch.object(
            MockStepConfig, "n_ic_timesteps", new_callable=unittest.mock.PropertyMock
        ) as n_ic_timesteps,
        unittest.mock.patch.object(
            MockStepConfig,
            "next_step_input_names",
            new_callable=unittest.mock.PropertyMock,
        ) as next_step_input_names,
    ):
        n_ic_timesteps.return_value = 1
        next_step_input_names.return_value = frozenset()
        yield


def test_ensemble_output_is_weighted_sum_of_members():
    step = get_ensemble_step(
        members=[
            get_single_module_selector(PlusOne()),
            get_single_module_selector(TimesTwo()),
        ],
        weights=[0.25, 0.75],
    )
    input_data = {name: torch.rand(2, *IMG_SHAPE) for name in NAMES}
    output = run_step(step, input_data).output
    assert set(output) == set(NAMES)
    for name in NAMES:
        x = input_data[name]
        torch.testing.assert_close(output[name], 0.25 * (x + 1) + 0.75 * (2 * x))


def test_ensemble_weights_are_not_normalized():
    step = get_ensemble_step(
        members=[
            get_single_module_selector(PlusOne()),
            get_single_module_selector(PlusOne()),
        ],
        weights=[1.0, 1.0],
    )
    input_data = {name: torch.rand(2, *IMG_SHAPE) for name in NAMES}
    output = run_step(step, input_data).output
    for name in NAMES:
        torch.testing.assert_close(output[name], 2 * (input_data[name] + 1))


def test_ensemble_mixes_residual_and_non_residual_members():
    step = get_ensemble_step(
        members=[
            get_single_module_selector(PlusOne(), residual_prediction=False),
            get_single_module_selector(PlusOne(), residual_prediction=True),
        ],
        weights=[0.5, 0.5],
    )
    input_data = {name: torch.rand(2, *IMG_SHAPE) for name in NAMES}
    output = run_step(step, input_data).output
    for name in NAMES:
        x = input_data[name]
        # non-residual member predicts x + 1, residual member predicts x + (x + 1)
        torch.testing.assert_close(output[name], 0.5 * (x + 1) + 0.5 * (2 * x + 1))


def test_ensemble_step_delegates_to_members():
    plus_one = PlusOne()
    times_two = TimesTwo()
    step = get_ensemble_step(
        members=[
            get_single_module_selector(plus_one),
            get_single_module_selector(times_two),
        ],
        weights=[0.5, 0.5],
    )
    assert step.input_names == frozenset(NAMES)
    assert step.output_names == frozenset(NAMES)
    assert step.n_ic_timesteps == 1
    assert len(step.modules) == 2
    assert step.get_state().keys() == {"members"}
    assert len(step.get_state()["members"]) == 2
    step.eval()
    assert all(not member._training for member in step._members)
    step.train()
    assert all(member._training for member in step._members)


def test_ensemble_load_state_requires_matching_member_count():
    step = get_ensemble_step(
        members=[
            get_single_module_selector(PlusOne()),
            get_single_module_selector(TimesTwo()),
        ],
        weights=[0.5, 0.5],
    )
    state = step.get_state()
    step.load_state(state)  # round trip is accepted
    with pytest.raises(ValueError, match="members"):
        step.load_state({"members": state["members"][:1]})


def test_ensemble_config_rejects_weight_count_mismatch():
    with pytest.raises(ValueError, match="weights"):
        EnsembleStepConfig(
            members=[
                get_single_module_selector(PlusOne()),
                get_single_module_selector(TimesTwo()),
            ],
            weights=[1.0],
        )


def test_ensemble_config_rejects_no_members():
    with pytest.raises(ValueError, match="at least one member"):
        EnsembleStepConfig(members=[], weights=[])


def test_ensemble_config_rejects_disagreeing_members():
    with mock_step_config_properties():
        with pytest.raises(ValueError, match="member 1 .* output_names"):
            EnsembleStepConfig(
                members=[get_mock_selector(["a"]), get_mock_selector(["b"])],
                weights=[0.5, 0.5],
            )


def test_ensemble_state_comes_from_first_member_and_diagnostics_are_dropped():
    output_a = {"a": torch.ones(1, 2, 2)}
    output_b = {"a": torch.zeros(1, 2, 2)}
    state_a = StepperState()
    state_b = StepperState()
    member_outputs = itertools.cycle(
        [
            StepOutput(
                output=output_a,
                stepper_state=state_a,
                corrector_diagnostics=CorrectorDiagnostics(
                    delta={"a": torch.ones(1, 2, 2)}
                ),
            ),
            StepOutput(output=output_b, stepper_state=state_b),
        ]
    )

    def _step(args: StepArgs, wrapper=lambda x: x):
        return next(member_outputs)

    with (
        mock_step_config_properties(),
        unittest.mock.patch.object(MockStep, "step", side_effect=_step),
    ):
        step = get_ensemble_step(
            [get_mock_selector(["a"]), get_mock_selector(["a"])], weights=[0.5, 0.5]
        )
        result = run_step(step, {"a": torch.rand(1, 2, 2)})
    torch.testing.assert_close(result.output["a"], 0.5 * torch.ones(1, 2, 2))
    assert result.stepper_state is state_a
    assert result.corrector_diagnostics.delta == {}


def test_ensemble_step_rejects_members_with_different_output_variables():
    member_outputs = itertools.cycle(
        [
            StepOutput(output={"a": torch.ones(1, 2, 2)}),
            StepOutput(output={"a": torch.ones(1, 2, 2), "extra": torch.ones(1, 2, 2)}),
        ]
    )

    def _step(args: StepArgs, wrapper=lambda x: x):
        return next(member_outputs)

    with (
        mock_step_config_properties(),
        unittest.mock.patch.object(MockStep, "step", side_effect=_step),
    ):
        step = get_ensemble_step(
            [get_mock_selector(["a"]), get_mock_selector(["a"])], weights=[0.5, 0.5]
        )
        with pytest.raises(ValueError, match="member 1 returned variables"):
            run_step(step, {"a": torch.rand(1, 2, 2)})
