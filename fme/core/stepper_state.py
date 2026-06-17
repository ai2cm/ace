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
from collections.abc import Sequence

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

    @classmethod
    def cat(cls, states: "Sequence[StepperState]") -> "StepperState":
        """Concatenate stepper states along the sample dimension.

        Inputs must agree on which sub-states are present (all ``None`` or all
        not-``None``); a present sub-state is concatenated along the sample dim.
        """
        corrector_states = [s.corrector_state for s in states]
        present = [c for c in corrector_states if c is not None]
        if not present:
            return cls()
        if len(present) != len(corrector_states):
            raise ValueError(
                "Cannot cat StepperState with inconsistent corrector_state presence."
            )
        return cls(corrector_state=CorrectorState.cat(present))

    def split(self, sample_sizes: "Sequence[int]") -> "list[StepperState]":
        """Split along the sample dimension into the given sample sizes."""
        if self.corrector_state is None:
            return [StepperState() for _ in sample_sizes]
        return [
            StepperState(corrector_state=c)
            for c in self.corrector_state.split(sample_sizes)
        ]
