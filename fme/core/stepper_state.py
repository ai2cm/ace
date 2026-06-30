"""State carried by the Stepper across predict calls.

The stepper threads this object through ``predict_generator``: each step may
read and update sub-state owned by its components (the corrector, and a
seedable random source). The terminal state is attached to the returned
``PrognosticState`` so it propagates from one ``predict`` call to the next
during inference.

The Stepper does not inspect the contents of sub-states; they are opaque
payloads owned by their respective components.
"""

import dataclasses

from fme.core.corrector.state import CorrectorState
from fme.core.random_state import RandomState


@dataclasses.dataclass
class StepperState:
    """Per-sample state carried by the Stepper across predict calls.

    Parameters:
        corrector_state: State owned by the corrector. ``None`` when the
            corrector has not seeded any state yet.
        random_state: Seedable random source driving stochastic modules (e.g.
            ``NoiseConditionedSFNO``). ``None`` when the rollout is not seeded,
            in which case stochastic modules fall back to the global torch RNG.
    """

    corrector_state: CorrectorState | None = None
    random_state: RandomState | None = None

    def to_device(self) -> "StepperState":
        return StepperState(
            corrector_state=(
                None
                if self.corrector_state is None
                else self.corrector_state.to_device()
            ),
            random_state=(
                None if self.random_state is None else self.random_state.to_device()
            ),
        )

    def to_cpu(self) -> "StepperState":
        return StepperState(
            corrector_state=(
                None if self.corrector_state is None else self.corrector_state.to_cpu()
            ),
            random_state=(
                None if self.random_state is None else self.random_state.to_cpu()
            ),
        )

    def pin_memory(self) -> "StepperState":
        if self.corrector_state is not None:
            self.corrector_state.pin_memory()
        if self.random_state is not None:
            self.random_state.pin_memory()
        return self

    def broadcast_ensemble(self, n_ensemble: int) -> "StepperState":
        return StepperState(
            corrector_state=(
                None
                if self.corrector_state is None
                else self.corrector_state.broadcast_ensemble(n_ensemble)
            ),
            random_state=(
                None
                if self.random_state is None
                else self.random_state.broadcast_ensemble(n_ensemble)
            ),
        )

    def sample_dim_size(self) -> int | None:
        """Return the leading (sample) dim of any sub-state that has one, or None.

        The random state is shared across the batch and has no per-sample
        dimension, so only the corrector state can constrain the sample size.
        """
        if self.corrector_state is not None:
            return self.corrector_state.sample_dim_size()
        return None
