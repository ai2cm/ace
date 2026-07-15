"""State carried by a Corrector across step calls.

The corrector owns the contents of this dataclass. The stepper treats it as an
opaque payload threaded along with the prognostic state from one prediction
window to the next.
"""

import dataclasses

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

    def to_state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize the set fields for a restart. An unseeded (all-None) field
        is omitted, so a round-trip of an empty ``CorrectorState`` stays empty.
        """
        if self.global_dry_air_mass is None:
            return {}
        return {"global_dry_air_mass": self.global_dry_air_mass}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "CorrectorState":
        """Rebuild from a serialized state; absent fields stay ``None``."""
        return cls(global_dry_air_mass=state.get("global_dry_air_mass"))

    @staticmethod
    def per_sample_state_keys() -> set[str]:
        """``to_state_dict`` keys whose tensors carry a leading per-sample
        dimension. All corrector fields are per-sample (see class docstring), so
        a serializer can mark them explicitly rather than inferring it from
        shape.
        """
        return {"global_dry_air_mass"}
