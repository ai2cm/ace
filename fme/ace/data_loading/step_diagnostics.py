"""Opaque, time-aware carriage for per-step private diagnostics.

A ``StepDiagnostics`` carries diagnostic *input* data produced inside the
``Stepper`` at the corrector boundary (today the pre-correction ``uncorrected``
series) so that downstream consumers — the correction-metrics aggregator and the
correction netCDF writer — can reconstruct what the corrector did without
re-running the model.

It follows the ``StepperState`` encapsulation pattern: ``BatchData`` and
``PairedData`` carry it as a single opaque field, forward it through every
structure-preserving method via the protocol defined here, and never inspect its
contents. Unlike ``StepperState`` (a terminal per-sample state), this payload is
a per-timestep diagnostic *series* aligned with the prediction it describes, so
it is **time-aware**: time-touching container operations slice or pad the series
alongside ``data``.

To add another piece of per-step private diagnostic data in the future, add a
field here and extend the protocol methods; the threading on ``BatchData`` and
``PairedData`` and the shared round-trip tests then cover it automatically.
"""

import dataclasses

import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.tensors import repeat_interleave_batch_dim
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class StepDiagnostics:
    """Per-timestep private diagnostics carried alongside a prediction series.

    Parameters:
        uncorrected: Pre-correction values of exactly the corrector-modified
            variables, as a sparse time series aligned (sample, time, ...) with
            the prediction ``data``. An empty mapping when no corrector modified
            any variable, so consumers need no ``None`` checks once they hold the
            container.
    """

    uncorrected: TensorMapping

    def to_device(self) -> "StepDiagnostics":
        device = get_device()
        return StepDiagnostics(
            uncorrected={k: v.to(device) for k, v in self.uncorrected.items()},
        )

    def to_cpu(self) -> "StepDiagnostics":
        return StepDiagnostics(
            uncorrected={k: v.cpu() for k, v in self.uncorrected.items()},
        )

    def pin_memory(self) -> "StepDiagnostics":
        self.uncorrected = {
            name: tensor.pin_memory() for name, tensor in self.uncorrected.items()
        }
        return self

    def broadcast_ensemble(self, n_ensemble: int) -> "StepDiagnostics":
        return StepDiagnostics(
            uncorrected=repeat_interleave_batch_dim(self.uncorrected, n_ensemble),
        )

    def sample_dim_size(self) -> int | None:
        """Return the leading (sample) dim of the series, or None when empty."""
        for tensor in self.uncorrected.values():
            return tensor.shape[0]
        return None

    def select_time_slice(self, time_slice: slice) -> "StepDiagnostics":
        """Slice the carried series along the time dim to stay aligned with data."""
        return StepDiagnostics(
            uncorrected={k: v[:, time_slice] for k, v in self.uncorrected.items()},
        )

    def prepend_time(self, n_steps: int) -> "StepDiagnostics":
        """Pad ``n_steps`` NaN-filled timesteps at the front of the series.

        Mirrors ``BatchData.prepend`` adding initial-condition timesteps to
        ``data``: the carried series has no initial-condition entry, so the
        prepended slots are NaN placeholders that keep the series aligned with
        ``data`` (and are stripped again by ``remove_initial_condition``).
        """
        padded: TensorDict = {}
        for name, tensor in self.uncorrected.items():
            pad_shape = (tensor.shape[0], n_steps, *tensor.shape[2:])
            pad = torch.full(
                pad_shape, float("nan"), dtype=tensor.dtype, device=tensor.device
            )
            padded[name] = torch.cat([pad, tensor], dim=1)
        return StepDiagnostics(uncorrected=padded)

    def scatter_spatial(self, global_img_shape: tuple[int, int]) -> "StepDiagnostics":
        """Slice the carried series to the local spatial chunk."""
        dist = Distributed.get_instance()
        return StepDiagnostics(
            uncorrected=dist.scatter_spatial(dict(self.uncorrected), global_img_shape),
        )


def get_uncorrected(step_diagnostics: StepDiagnostics | None) -> TensorMapping:
    """Read the pre-correction series out of a possibly-absent container.

    The single accessor the correction-metrics aggregator and the correction
    netCDF writer use to reach into the carriage. Returns an empty mapping when
    no diagnostics were carried (no corrector ran, or a non-prediction
    ``BatchData``), so those consumers stay silent without inspecting the
    container themselves.
    """
    if step_diagnostics is None:
        return {}
    return step_diagnostics.uncorrected
