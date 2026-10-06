import pytest
import torch

from fme.core.corrector.output import CorrectorDiagnostics
from fme.core.step.output import StepOutput
from fme.core.step.step_diagnostics import StepDiagnostics


def _step_output(delta_value: float | None) -> StepOutput:
    output = {"a": torch.zeros(2, 4, 5)}
    if delta_value is None:
        return StepOutput(output=output)
    return StepOutput(
        output=output,
        corrector_diagnostics=CorrectorDiagnostics(
            delta={"a": torch.full((2, 4, 5), delta_value)}
        ),
    )


def test_stack_diagnostics_stacks_forward_step_aligned():
    outputs = [_step_output(float(i)) for i in range(3)]
    stacked = StepOutput.stack_diagnostics(outputs)
    assert isinstance(stacked, StepDiagnostics)
    assert set(stacked.delta) == {"a"}
    assert stacked.delta["a"].shape == (2, 3, 4, 5)
    for step in range(3):
        torch.testing.assert_close(
            stacked.delta["a"][:, step],
            torch.full((2, 4, 5), float(step)),
        )


def test_stack_diagnostics_returns_none_without_deltas():
    outputs = [_step_output(None) for _ in range(3)]
    assert StepOutput.stack_diagnostics(outputs) is None
    assert StepOutput.stack_diagnostics([]) is None


def test_stack_diagnostics_raises_on_inconsistent_names():
    outputs = [_step_output(1.0), _step_output(None)]
    with pytest.raises(ValueError, match="inconsistent"):
        StepOutput.stack_diagnostics(outputs)
