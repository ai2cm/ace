"""State carried by a Corrector across step calls.

The corrector owns the contents of this dataclass. The stepper treats it as an
opaque payload threaded along with the prognostic state from one prediction
window to the next.
"""

import dataclasses
from collections.abc import Sequence

import torch

from fme.core.device import get_device


@dataclasses.dataclass
class CorrectorState:
    """Per-sample state owned by the Corrector.

    All fields are optional and per-sample (no time dim). ``None`` means
    the field has not been seeded yet.

    Parameters:
        global_dry_air_mass: Reference global-mean dry-air mass (expressed
            as the area-weighted mean of the dry-air surface pressure)
            captured from the IC, used to pin the global dry-air mass of
            subsequent steps. Shape ``(n_samples, 1, 1)``.
    """

    global_dry_air_mass: torch.Tensor | None = None

    def to_device(self) -> "CorrectorState":
        device = get_device()
        return CorrectorState(
            global_dry_air_mass=(
                None
                if self.global_dry_air_mass is None
                else self.global_dry_air_mass.to(device)
            ),
        )

    def to_cpu(self) -> "CorrectorState":
        return CorrectorState(
            global_dry_air_mass=(
                None
                if self.global_dry_air_mass is None
                else self.global_dry_air_mass.cpu()
            ),
        )

    def pin_memory(self) -> "CorrectorState":
        if self.global_dry_air_mass is not None:
            self.global_dry_air_mass = self.global_dry_air_mass.pin_memory()
        return self

    def broadcast_ensemble(self, n_ensemble: int) -> "CorrectorState":
        if self.global_dry_air_mass is None:
            return CorrectorState()
        return CorrectorState(
            global_dry_air_mass=torch.repeat_interleave(
                self.global_dry_air_mass, n_ensemble, dim=0
            ),
        )

    def sample_dim_size(self) -> int | None:
        """Return the leading (sample) dim of any non-None field, or None if empty."""
        if self.global_dry_air_mass is not None:
            return self.global_dry_air_mass.shape[0]
        return None

    @classmethod
    def cat(cls, states: "Sequence[CorrectorState]") -> "CorrectorState":
        """Concatenate corrector states along the sample dimension.

        Inputs must agree on which fields are present (all ``None`` or all
        not-``None`` for each field); a present field is concatenated along
        dim 0.
        """
        masses = [s.global_dry_air_mass for s in states]
        present = [m for m in masses if m is not None]
        if not present:
            return cls()
        if len(present) != len(masses):
            raise ValueError(
                "Cannot cat CorrectorState with inconsistent global_dry_air_mass "
                "presence."
            )
        return cls(global_dry_air_mass=torch.cat(present, dim=0))

    def split(self, sample_sizes: "Sequence[int]") -> "list[CorrectorState]":
        """Split along the sample dimension into the given sample sizes."""
        if self.global_dry_air_mass is None:
            return [CorrectorState() for _ in sample_sizes]
        pieces = torch.split(self.global_dry_air_mass, list(sample_sizes), dim=0)
        return [CorrectorState(global_dry_air_mass=p) for p in pieces]
