"""State carried by the Stepper across predict calls.

The stepper threads this object through ``predict_generator``: each step may
read and update sub-state owned by its components (today only the corrector,
in the future possibly others). The terminal state is attached to the returned
``PrognosticState`` so it propagates from one ``predict`` call to the next
during inference.

The Stepper does not inspect the contents of sub-states; they are opaque
payloads owned by their respective components.
"""

import dataclasses

from fme.core.corrector.state import CorrectorState


@dataclasses.dataclass
class StepperState:
    """Per-sample state carried by the Stepper across predict calls.

    Parameters:
        corrector_state: State owned by the corrector. ``None`` when the
            corrector has not seeded any state yet.
    """

    corrector_state: CorrectorState | None = None

    def to_device(self) -> "StepperState":
        return StepperState(
            corrector_state=(
                None
                if self.corrector_state is None
                else self.corrector_state.to_device()
            ),
        )

    def to_cpu(self) -> "StepperState":
        return StepperState(
            corrector_state=(
                None if self.corrector_state is None else self.corrector_state.to_cpu()
            ),
        )

    def pin_memory(self) -> "StepperState":
        if self.corrector_state is not None:
            self.corrector_state.pin_memory()
        return self

    def broadcast_ensemble(self, n_ensemble: int) -> "StepperState":
        return StepperState(
            corrector_state=(
                None
                if self.corrector_state is None
                else self.corrector_state.broadcast_ensemble(n_ensemble)
            ),
        )

    def sample_dim_size(self) -> int | None:
        """Return the leading (sample) dim of any non-None sub-state, or None."""
        if self.corrector_state is not None:
            return self.corrector_state.sample_dim_size()
        return None
