import dataclasses

from fme.core.corrector.output import CorrectorDiagnostics
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
