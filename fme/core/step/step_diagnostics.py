"""Per-sample step diagnostics carried on prediction data.

``StepDiagnostics`` is an opaque container attached by ``Stepper.predict`` to
the prediction ``BatchData``. Like ``StepperState``, the structure-preserving
operations (device moves, ensemble broadcast, pin-memory) apply without
inspecting its contents. Data consumers have two sanctioned read surfaces:
``to_datasets`` for serialization (self-describing, detached CPU exports for
the step-diagnostics writer, keyed by dataset name), and the ``delta`` /
``pre_diagnosis_fields`` fields directly for in-memory consumers that need the
tensors on device (e.g. inference metrics).
"""

import dataclasses
from collections.abc import Mapping

import xarray as xr

from fme.core.device import get_device
from fme.core.tensors import repeat_interleave_batch_dim
from fme.core.typing_ import TensorMapping

# Dataset name for the corrector's per-step correction deltas; determines the
# output netCDF filename, i.e. step_diagnostics/correction_deltas.nc.
CORRECTION_DELTAS = "correction_deltas"

# Dataset name for pre-diagnosis field snapshots; determines the output netCDF
# filename, i.e. step_diagnostics/pre_diagnosis_fields.nc.
PRE_DIAGNOSIS_FIELDS = "pre_diagnosis_fields"


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
            ``to_datasets`` when exporting for writing.
        pre_diagnosis_fields: The pre-diagnosis field values (raw network
            output before the corrector diagnosed a field), shaped
            ``(sample, time, ...)`` and aligned with the prediction data's
            forward steps. Empty when no corrector ran or none diagnosed
            anything.
    """

    delta: TensorMapping
    pre_diagnosis_fields: TensorMapping = dataclasses.field(default_factory=dict)

    def to_device(self) -> "StepDiagnostics":
        device = get_device()
        return StepDiagnostics(
            delta={k: v.to(device) for k, v in self.delta.items()},
            pre_diagnosis_fields={
                k: v.to(device) for k, v in self.pre_diagnosis_fields.items()
            },
        )

    def to_cpu(self) -> "StepDiagnostics":
        return StepDiagnostics(
            delta={k: v.cpu() for k, v in self.delta.items()},
            pre_diagnosis_fields={
                k: v.cpu() for k, v in self.pre_diagnosis_fields.items()
            },
        )

    def pin_memory(self) -> "StepDiagnostics":
        self.delta = {k: v.pin_memory() for k, v in self.delta.items()}
        self.pre_diagnosis_fields = {
            k: v.pin_memory() for k, v in self.pre_diagnosis_fields.items()
        }
        return self

    def broadcast_ensemble(self, n_ensemble: int) -> "StepDiagnostics":
        return StepDiagnostics(
            delta=repeat_interleave_batch_dim(self.delta, n_ensemble),
            pre_diagnosis_fields=repeat_interleave_batch_dim(
                self.pre_diagnosis_fields, n_ensemble
            ),
        )

    def sample_dim_size(self) -> int | None:
        """Return the leading (sample) dim of any non-empty field, or None when
        both mappings are empty.
        """
        for tensor in self.delta.values():
            return tensor.shape[0]
        for tensor in self.pre_diagnosis_fields.values():
            return tensor.shape[0]
        return None

    def to_datasets(self, time: xr.DataArray) -> Mapping[str, xr.Dataset]:
        """Export the diagnostics as named datasets for writing, with the
        given time coordinate.

        Args:
            time: The valid-time coordinate of the prediction data this
                diagnostics series is aligned with, with dims
                ``(sample, time)``.

        Returns:
            A mapping of dataset name to dataset. The correction deltas appear
            under ``CORRECTION_DELTAS`` and the pre-diagnosis field snapshots
            under ``PRE_DIAGNOSIS_FIELDS``, each with one variable per entry,
            dims ``(sample, time, ...)``, and the given times as a
            ``valid_time`` coordinate; a dataset is empty when its mapping is
            empty.
        """

        def _to_dataset(mapping: TensorMapping) -> xr.Dataset:
            data_vars = {}
            for name, tensor in mapping.items():
                array = tensor.detach().cpu().numpy()
                dims = ["sample", "time"] + [f"dim_{i}" for i in range(array.ndim - 2)]
                data_vars[name] = xr.DataArray(array, dims=dims)
            ds = xr.Dataset(data_vars)
            return ds.assign_coords(valid_time=(("sample", "time"), time.values))

        result: dict[str, xr.Dataset] = {
            CORRECTION_DELTAS: _to_dataset(self.delta),
        }
        if self.pre_diagnosis_fields:
            result[PRE_DIAGNOSIS_FIELDS] = _to_dataset(self.pre_diagnosis_fields)
        return result
