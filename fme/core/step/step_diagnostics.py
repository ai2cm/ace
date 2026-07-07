"""Per-sample step diagnostics carried on prediction data.

``StepDiagnostics`` is an opaque container attached by ``Stepper.predict`` to
the prediction ``BatchData``. Like ``StepperState``, the structure-preserving
operations (device moves, ensemble broadcast, pin-memory) apply without
inspecting its contents. Data consumers have two sanctioned read surfaces:
``to_dataset`` for serialization (a self-describing, detached CPU export for
the step-diagnostics writer), and the ``delta`` field directly for in-memory
consumers that need the tensors on device (e.g. inference metrics).
"""

import dataclasses

import xarray as xr

from fme.core.device import get_device
from fme.core.tensors import repeat_interleave_batch_dim
from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class StepDiagnostics:
    """Per-sample diagnostic series aligned with prediction data.

    Parameters:
        delta: The per-step correction ``corrected - network_output`` for each
            corrector-modified variable, shaped ``(sample, time, ...)`` and
            aligned with the prediction data's forward steps. May be empty;
            every operation is a safe no-op on an empty mapping.
            The tensors carry the stepper's output masking (NaN off-mask) and
            are the on-device read surface for in-memory consumers; use
            ``to_dataset`` when exporting for writing.
    """

    delta: TensorMapping

    def to_device(self) -> "StepDiagnostics":
        device = get_device()
        return StepDiagnostics(delta={k: v.to(device) for k, v in self.delta.items()})

    def to_cpu(self) -> "StepDiagnostics":
        return StepDiagnostics(delta={k: v.cpu() for k, v in self.delta.items()})

    def pin_memory(self) -> "StepDiagnostics":
        self.delta = {k: v.pin_memory() for k, v in self.delta.items()}
        return self

    def broadcast_ensemble(self, n_ensemble: int) -> "StepDiagnostics":
        return StepDiagnostics(
            delta=repeat_interleave_batch_dim(self.delta, n_ensemble)
        )

    def sample_dim_size(self) -> int | None:
        """Return the leading (sample) dim of the delta tensors, or None when
        the delta is empty.
        """
        for tensor in self.delta.values():
            return tensor.shape[0]
        return None

    def to_dataset(self, time: xr.DataArray) -> xr.Dataset:
        """Export the diagnostics for writing, with the given time coordinate.

        Args:
            time: The valid-time coordinate of the prediction data this
                diagnostics series is aligned with, with dims
                ``(sample, time)``.

        Returns:
            A dataset with one variable per delta entry, dims
            ``(sample, time, ...)``, and the given times as a ``valid_time``
            coordinate.
        """
        data_vars = {}
        for name, tensor in self.delta.items():
            array = tensor.detach().cpu().numpy()
            dims = ["sample", "time"] + [f"dim_{i}" for i in range(array.ndim - 2)]
            data_vars[name] = xr.DataArray(array, dims=dims)
        ds = xr.Dataset(data_vars)
        return ds.assign_coords(valid_time=(("sample", "time"), time.values))
