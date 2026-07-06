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

import torch

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

    def to_state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize present sub-states for a restart sidecar.

        Each present sub-state is delegated to and its keys namespaced
        (e.g. ``"corrector_state.global_dry_air_mass"``). A ``<name>.present``
        marker records that a sub-state was set even when it serializes to no
        fields (an empty ``CorrectorState``), so ``from_state_dict`` restores a
        ``None`` sub-state as ``None`` and a present-but-empty one as empty. The
        stepper only knows the sub-state names; each sub-state owns its fields.
        """
        result: dict[str, torch.Tensor] = {}
        for name in ("corrector_state", "random_state"):
            sub_state = getattr(self, name)
            if sub_state is None:
                continue
            result[f"{name}.present"] = torch.tensor(True)
            for key, value in sub_state.to_state_dict().items():
                result[f"{name}.{key}"] = value
        return result

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "StepperState":
        """Rebuild from ``to_state_dict``; a sub-state absent from the serialized
        state (no ``<name>.present`` marker) is restored as ``None``.
        """
        corrector_state: CorrectorState | None = None
        random_state: RandomState | None = None
        if "corrector_state.present" in state:
            corrector_state = CorrectorState.from_state_dict(
                _sub_state_dict(state, "corrector_state")
            )
        if "random_state.present" in state:
            random_state = RandomState.from_state_dict(
                _sub_state_dict(state, "random_state")
            )
        return cls(corrector_state=corrector_state, random_state=random_state)


def _sub_state_dict(
    state: dict[str, torch.Tensor], name: str
) -> dict[str, torch.Tensor]:
    """Extract a sub-state's fields from a namespaced ``StepperState`` state dict,
    stripping the ``<name>.`` prefix and dropping the ``<name>.present`` marker.
    """
    prefix = f"{name}."
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix) and key != f"{prefix}present"
    }
