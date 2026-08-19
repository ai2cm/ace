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


def _step_output_diagnosis_only(pre_diag_value: float) -> StepOutput:
    """StepOutput with only pre_diagnosis_fields (no delta)."""
    return StepOutput(
        output={"a": torch.zeros(2, 4, 5)},
        corrector_diagnostics=CorrectorDiagnostics(
            pre_diagnosis_fields={"a": torch.full((2, 4, 5), pre_diag_value)},
        ),
    )


def test_stack_diagnostics_diagnosis_only():
    """Stacking outputs with only pre_diagnosis_fields (no delta) produces a
    StepDiagnostics with pre_diagnosis_fields and a non-None sample_dim_size."""
    outputs = [_step_output_diagnosis_only(float(i)) for i in range(3)]
    stacked = StepOutput.stack_diagnostics(outputs)
    assert isinstance(stacked, StepDiagnostics)
    assert stacked.delta == {}
    assert set(stacked.pre_diagnosis_fields) == {"a"}
    assert stacked.pre_diagnosis_fields["a"].shape == (2, 3, 4, 5)
    assert stacked.sample_dim_size() == 2
    for step in range(3):
        torch.testing.assert_close(
            stacked.pre_diagnosis_fields["a"][:, step],
            torch.full((2, 4, 5), float(step)),
        )
