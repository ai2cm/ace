"""State carried by the Stepper across predict calls.

The stepper threads this object through ``predict_generator``: each step may
read and update sub-state owned by its components (the corrector, and a
seedable random source). The terminal state is attached to the returned
``PrognosticState`` so it propagates from one ``predict`` call to the next
during inference.

The Stepper does not inspect the contents of sub-states; they are opaque
payloads owned by their respective components.
"""

from __future__ import annotations

import dataclasses

import torch

from fme.core.corrector.state import CorrectorState
from fme.core.random_state import RandomState


class UngatheredStateDictError(Exception):
    """The state dict does not contain gathered per-rank random state."""


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

    def select_sample_slice(self, sample_slice: slice) -> "StepperState":
        """Select a contiguous range of samples."""
        return StepperState(
            corrector_state=(
                None
                if self.corrector_state is None
                else self.corrector_state.select_sample_slice(sample_slice)
            ),
            random_state=self.random_state,
        )

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
        """Serialize present sub-states for a restart stepper state file.

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

    @staticmethod
    def per_sample_state_keys() -> set[str]:
        """Namespaced ``to_state_dict`` keys whose tensors carry a leading
        per-sample dimension, delegated to each sub-state's own declaration.

        Lets a serializer mark those variables per-sample explicitly (so they are
        subselected along the sample axis with the prognostic variables) instead
        of inferring per-sample-ness from a tensor length that happens to match
        the sample count.
        """
        keys: set[str] = set()
        for name, sub_state_class in (
            ("corrector_state", CorrectorState),
            ("random_state", RandomState),
        ):
            keys.update(
                f"{name}.{key}" for key in sub_state_class.per_sample_state_keys()
            )
        return keys


class GatheredStepperState:
    """Stepper state after a data-parallel gather.

    Stores the per-rank ``StepperState`` objects directly. Serialization
    is fully per-rank: each rank's ``StepperState`` is delegated to
    ``StepperState.to_state_dict``/``from_state_dict`` under a
    ``rank_<i>.`` namespace, so new sub-states added to ``StepperState``
    are automatically included without changes here.
    """

    def __init__(self, *, states: list[StepperState]):
        self._states = list(states)

    @property
    def n_ranks(self) -> int:
        return len(self._states)

    def to_cpu(self) -> GatheredStepperState:
        return GatheredStepperState(
            states=[s.to_cpu() for s in self._states]
        )

    def get_for_rank(self, rank: int) -> StepperState:
        """Return the ``StepperState`` for a single data-parallel rank."""
        return self._states[rank]

    def scatter_random_state(self, rank: int) -> StepperState:
        """Return a ``StepperState`` with the corrector gathered from
        all ranks and only the specified rank's random state.

        The restart file stores data for all samples together, so the
        corrector must match the full sample count; this method
        concatenates the per-rank corrector shards for that purpose.
        """
        correctors = [s.corrector_state for s in self._states]
        if correctors[0] is not None:
            gathered_corrector: CorrectorState | None = CorrectorState.concat(
                correctors  # type: ignore[arg-type]
            )
        else:
            gathered_corrector = None
        return StepperState(
            corrector_state=gathered_corrector,
            random_state=self._states[rank].random_state,
        )

    def to_state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize for a restart file.

        Each rank's ``StepperState`` is serialized under a
        ``rank_<i>.`` namespace, with an ``n_ranks`` marker so the
        reader knows how many to expect.
        """
        result: dict[str, torch.Tensor] = {
            "n_ranks": torch.tensor(len(self._states))
        }
        for i, state in enumerate(self._states):
            for key, value in state.to_state_dict().items():
                result[f"rank_{i}.{key}"] = value
        return result

    @classmethod
    def from_state_dict(
        cls, state: dict[str, torch.Tensor]
    ) -> GatheredStepperState:
        """Rebuild from ``to_state_dict``.

        Raises:
            UngatheredStateDictError: If the state dict does not contain
                an ``n_ranks`` marker, indicating it is a plain
                ``StepperState`` dict rather than a gathered one.
        """
        if "n_ranks" not in state:
            raise UngatheredStateDictError(
                "State dict has no n_ranks marker; "
                "this is a plain StepperState dict, not a gathered one."
            )
        n_ranks = int(state["n_ranks"].item())
        states = [
            StepperState.from_state_dict(_sub_state_dict(state, f"rank_{i}"))
            for i in range(n_ranks)
        ]
        return cls(states=states)

    def per_sample_state_keys(self) -> set[str]:
        """Keys whose tensors carry a leading per-sample dimension.

        Per-rank state variables are NOT per-sample in the gathered
        sense: each rank's corrector carries that rank's sample shard,
        not the full gathered sample count, so none of them should share
        the ``sample`` dim with the prognostic data.
        """
        return set()


def _sub_state_dict(
    state: dict[str, torch.Tensor], name: str
) -> dict[str, torch.Tensor]:
    """Extract a sub-state's fields from a namespaced state dict,
    stripping the ``<name>.`` prefix and dropping the ``<name>.present`` marker.
    """
    prefix = f"{name}."
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix) and key != f"{prefix}present"
    }
