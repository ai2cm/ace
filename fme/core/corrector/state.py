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
        global_mean_surface_pressure: Reference global-mean surface pressure
            captured from the first ``input`` the corrector sees, used to pin
            the global mean of subsequent steps. Shape ``(n_samples, 1, 1)``.
    """

    global_mean_surface_pressure: torch.Tensor | None = None

    def to_device(self) -> "CorrectorState":
        device = get_device()
        return CorrectorState(
            global_mean_surface_pressure=(
                None
                if self.global_mean_surface_pressure is None
                else self.global_mean_surface_pressure.to(device)
            ),
        )

    def to_cpu(self) -> "CorrectorState":
        return CorrectorState(
            global_mean_surface_pressure=(
                None
                if self.global_mean_surface_pressure is None
                else self.global_mean_surface_pressure.cpu()
            ),
        )

    def pin_memory(self) -> "CorrectorState":
        if self.global_mean_surface_pressure is not None:
            self.global_mean_surface_pressure = (
                self.global_mean_surface_pressure.pin_memory()
            )
        return self

    def broadcast_ensemble(self, n_ensemble: int) -> "CorrectorState":
        if self.global_mean_surface_pressure is None:
            return CorrectorState()
        return CorrectorState(
            global_mean_surface_pressure=torch.repeat_interleave(
                self.global_mean_surface_pressure, n_ensemble, dim=0
            ),
        )

    def sample_dim_size(self) -> int | None:
        """Return the leading (sample) dim of any non-None field, or None if empty."""
        if self.global_mean_surface_pressure is not None:
            return self.global_mean_surface_pressure.shape[0]
        return None
