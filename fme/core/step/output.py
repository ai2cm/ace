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

        Stacks each output's correction ``delta`` along a new time dim (dim 1),
        aligned with the forward steps the outputs correspond to. How the
        per-step diagnostics compose into a series stays encapsulated here;
        callers attach the returned container and consumers read it through
        ``StepDiagnostics.to_dataset``.

        Args:
            outputs: One ``StepOutput`` per forward step, in step order.

        Returns:
            The stacked diagnostics, or None when no output carries a delta
            (no corrector ran, or none modified anything).
        """
        keys: set[str] = set()
        for output in outputs:
            keys.update(output.corrector_diagnostics.delta.keys())
        if not keys:
            return None
        for output in outputs:
            if output.corrector_diagnostics.delta.keys() != keys:
                raise ValueError(
                    "Cannot stack corrector diagnostics with inconsistent "
                    f"variable names across steps: expected {sorted(keys)}, "
                    f"got {sorted(output.corrector_diagnostics.delta.keys())}."
                )
        return StepDiagnostics(
            delta={
                k: torch.stack(
                    [output.corrector_diagnostics.delta[k] for output in outputs],
                    dim=1,
                )
                for k in sorted(keys)
            }
        )
