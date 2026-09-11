import contextlib
import dataclasses
import itertools
import unittest.mock

import pytest
import torch
from torch import nn

from fme.core.corrector.output import CorrectorDiagnostics
from fme.core.ocean import OceanConfig
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
    module: nn.Module,
    residual_prediction: bool = False,
    in_names: list[str] = NAMES,
    out_names: list[str] = NAMES,
    **kwargs,
) -> StepSelector:
    """A step over the given module with identity normalization, so the step's
    output is the module's output (plus the input, if residual)."""
    all_names = sorted(set(in_names).union(out_names))
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(type="prebuilt", config={"module": module}),
                in_names=in_names,
                out_names=out_names,
                normalization=trivial_network_and_loss_normalization(all_names),
                residual_prediction=residual_prediction,
                **kwargs,
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


@pytest.mark.parametrize(
    "member_kwargs, disagreement",
    [
        pytest.param(dict(in_names=["a", "b", "c"]), "input_names", id="input"),
        pytest.param(dict(out_names=["a"]), "output_names", id="output"),
        pytest.param(
            dict(prescribed_prognostic_names=["a"]),
            "next_step_input_names",
            id="next_step_input",
        ),
        pytest.param(
            dict(in_names=["a", "b", "f"], next_step_forcing_names=["f"]),
            "next_step_forcing_names",
            id="next_step_forcing",
        ),
        pytest.param(
            dict(
                in_names=["a", "b", "mask"],
                ocean=OceanConfig("a", "mask", interpolate=True),
            ),
            "ocean",
            id="ocean",
        ),
    ],
)
def test_ensemble_config_names_the_disagreeing_property(member_kwargs, disagreement):
    """Member 1 is built with member_kwargs; member 0 gets only what is needed
    to make the named property the first one that differs."""
    first_kwargs: dict = {}
    if disagreement not in ("input_names", "output_names"):
        first_kwargs = {
            k: v for k, v in member_kwargs.items() if k in ("in_names", "out_names")
        }
    if disagreement == "ocean":
        first_kwargs["ocean"] = OceanConfig("a", "mask")
    with pytest.raises(ValueError, match=f"member 1 .* {disagreement}"):
        EnsembleStepConfig(
            members=[
                get_single_module_selector(PlusOne(), **first_kwargs),
                get_single_module_selector(PlusOne(), **member_kwargs),
            ],
            weights=[0.5, 0.5],
        )


def test_ensemble_step_forwards_wrapper_epoch_and_regularizer_loss():
    wrappers_seen = []

    def _step(args: StepArgs, wrapper=lambda x: x):
        wrappers_seen.append(wrapper)
        return StepOutput(output={"a": torch.ones(1, 2, 2)})

    def _wrapper(module: nn.Module) -> nn.Module:
        return module

    with (
        mock_step_config_properties(),
        unittest.mock.patch.object(MockStep, "step", side_effect=_step),
        unittest.mock.patch.object(MockStep, "set_epoch") as set_epoch,
        unittest.mock.patch.object(
            MockStep,
            "get_regularizer_loss",
            side_effect=[torch.tensor(1.0), torch.tensor(2.0)],
        ),
    ):
        step = get_ensemble_step(
            [get_mock_selector(["a"]), get_mock_selector(["a"])], weights=[0.5, 0.5]
        )
        step.step(
            args=StepArgs(
                input={"a": torch.rand(1, 2, 2)}, next_step_input_data={}, labels=None
            ),
            wrapper=_wrapper,
        )
        step.set_epoch(3)
        torch.testing.assert_close(step.get_regularizer_loss(), torch.tensor(3.0))
    assert wrappers_seen == [_wrapper, _wrapper]
    assert set_epoch.call_args_list == [unittest.mock.call(3), unittest.mock.call(3)]


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
