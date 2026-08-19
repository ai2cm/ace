import dataclasses
from collections.abc import Sequence

import torch

from fme.core.corrector.output import CorrectorDiagnostics
from fme.core.step.step_diagnostics import StepDiagnostics
from fme.core.stepper_state import StepperState
from fme.core.typing_ import TensorDict


@dataclasses.dataclass
class StepOutput:
    """One step's denormalized output plus its corrector diagnostics and state.

    Parameters:
        output: The denormalized data at the next time step.
        stepper_state: Per-sample state to thread into the next step call, or
            None if no state is carried.
        corrector_diagnostics: The corrector's per-variable correction ``delta``
            for this step. Empty when no corrector ran or none modified anything.
    """

    output: TensorDict
    stepper_state: StepperState | None = None
    corrector_diagnostics: CorrectorDiagnostics = dataclasses.field(
        default_factory=CorrectorDiagnostics
    )

    @classmethod
    def stack_diagnostics(
        cls, outputs: Sequence["StepOutput"]
    ) -> StepDiagnostics | None:
        """Stack per-step corrector diagnostics into a time series.

        Stacks each output's correction ``delta`` and
        ``pre_diagnosis_fields`` along a new time dim (dim 1), aligned with
        the forward steps the outputs correspond to. How the per-step
        diagnostics compose into a series stays encapsulated here; callers
        attach the returned container and consumers read it through
        ``StepDiagnostics.to_datasets``.

        Args:
            outputs: One ``StepOutput`` per forward step, in step order.

        Returns:
            The stacked diagnostics, or None when no output carries a delta
            or pre-diagnosis entry (no corrector ran, or none modified
            anything).
        """
        delta_keys: set[str] = set()
        pre_diag_keys: set[str] = set()
        for output in outputs:
            delta_keys.update(output.corrector_diagnostics.delta.keys())
            pre_diag_keys.update(
                output.corrector_diagnostics.pre_diagnosis_fields.keys()
            )
        if not delta_keys and not pre_diag_keys:
            return None
        for output in outputs:
            if delta_keys and output.corrector_diagnostics.delta.keys() != delta_keys:
                raise ValueError(
                    "Cannot stack corrector diagnostics with inconsistent "
                    f"delta variable names across steps: expected "
                    f"{sorted(delta_keys)}, got "
                    f"{sorted(output.corrector_diagnostics.delta.keys())}."
                )
            if (
                pre_diag_keys
                and output.corrector_diagnostics.pre_diagnosis_fields.keys()
                != pre_diag_keys
            ):
                raise ValueError(
                    "Cannot stack corrector diagnostics with inconsistent "
                    f"pre_diagnosis_fields variable names across steps: "
                    f"expected {sorted(pre_diag_keys)}, got "
                    f"{sorted(output.corrector_diagnostics.pre_diagnosis_fields.keys())}."
                )
        delta = {
            k: torch.stack(
                [output.corrector_diagnostics.delta[k] for output in outputs],
                dim=1,
            )
            for k in sorted(delta_keys)
        }
        pre_diagnosis_fields = {
            k: torch.stack(
                [
                    output.corrector_diagnostics.pre_diagnosis_fields[k]
                    for output in outputs
                ],
                dim=1,
            )
            for k in sorted(pre_diag_keys)
        }
        return StepDiagnostics(
            delta=delta,
            pre_diagnosis_fields=pre_diagnosis_fields,
        )
