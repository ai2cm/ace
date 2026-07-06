"""Embed the full round-trippable state of a ``PrognosticState``/``BatchData``
into the restart netCDF, and read it back.

``restart.nc`` historically stored only the prognostic ``data`` and ``time`` -
a lossy projection of ``BatchData`` that silently drops ``stepper_state``,
``labels``, and ``data_mask``. This module carries those extra fields inside the
same netCDF so a resumed rollout continues exactly where it left off, while the
file stays a normal, inspectable xarray ``Dataset`` (the prognostic variables
keep their plain names).

Encoding conventions:

* Extra fields live under the reserved prefix ``_fme_state__`` so they cannot
  collide with prognostic variable names.
* A dataset attribute ``_fme_restart_schema_version`` is the presence marker: a
  reader without it (a legacy ``restart.nc`` or any user-supplied IC dataset)
  behaves exactly as before (no stepper state, labels from config).
* Only fields that are actually present are written, so an unseeded,
  corrector-free, label-less run produces a restart netCDF byte-identical to the
  data+time-only file written before this feature.

Scope: ``n_ensemble`` and ``horizontal_dims`` are intentionally NOT serialized.
They are structural/derived on load - the restart's sample dimension already
carries the broadcast ensemble, and storing ``n_ensemble`` would risk
reintroducing the double-broadcast bug. Only the round-trippable payload
(``stepper_state``, ``labels``, ``data_mask``) is embedded.
"""

import dataclasses
import json
from typing import Any

import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.core.labels import BatchLabels
from fme.core.stepper_state import StepperState

RESERVED_PREFIX = "_fme_state__"
SCHEMA_ATTR = "_fme_restart_schema_version"
SCHEMA_VERSION = 1
DTYPE_ATTR = "_fme_dtype"
SAMPLE_DIM = "sample"

_STEPPER_PREFIX = f"{RESERVED_PREFIX}stepper__"
_LABELS_VALUES_VAR = f"{RESERVED_PREFIX}labels_values"
_LABELS_NAMES_ATTR = f"{RESERVED_PREFIX}labels_names"
_LABEL_INDEX_DIM = f"{RESERVED_PREFIX}label_index"
_DATA_MASK_PREFIX = f"{RESERVED_PREFIX}data_mask__"

# netCDF4 has no native bool; bool tensors are stored as uint8 and cast back on
# read using the recorded dtype attribute.
_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    )
}


@dataclasses.dataclass
class RestartExtras:
    """The extra ``BatchData`` fields recovered from a full-state restart netCDF.

    Each is ``None`` when the restart did not carry it.
    """

    stepper_state: StepperState | None = None
    labels: BatchLabels | None = None
    data_mask: dict[str, torch.Tensor] | None = None


def _to_storage_array(tensor: torch.Tensor) -> tuple[np.ndarray, str]:
    """Return a netCDF-storable array plus the original torch dtype name.

    Bool is stored as uint8 (netCDF4 has no bool); the recorded dtype lets the
    reader restore the exact original dtype.
    """
    cpu = tensor.detach().cpu()
    dtype_name = str(cpu.dtype)
    if cpu.dtype == torch.bool:
        return cpu.to(torch.uint8).numpy(), dtype_name
    return cpu.numpy(), dtype_name


def _stepper_var_dims(var_name: str, array: np.ndarray, n_samples: int) -> list[str]:
    """Dim names for a reserved stepper variable.

    A leading axis sized like the batch's sample dimension is named ``sample`` so
    ``start_indices`` subselection (``ds.isel(sample=...)``) subsets it in step
    with the prognostic variables; every other axis gets a variable-private
    reserved dim so distinct variables never share (and thus constrain) a dim.
    The corrector's ``global_dry_air_mass`` (n_samples, 1, 1) is the per-sample
    case; the generator state (a flat uint8 vector) is not.
    """
    dims: list[str] = []
    for axis, size in enumerate(array.shape):
        if axis == 0 and size == n_samples:
            dims.append(SAMPLE_DIM)
        else:
            dims.append(f"{var_name}_d{axis}")
    return dims


def encode_restart_extras(
    batch: BatchData,
) -> tuple[dict[str, xr.DataArray], dict[str, Any]]:
    """Build the reserved data variables and dataset attributes that embed the
    round-trippable extras of ``batch`` (stepper_state, labels, data_mask).

    Returns ``({}, {})`` when there is nothing extra to store, so the restart
    netCDF stays byte-identical to a data+time-only file.
    """
    data_arrays: dict[str, xr.DataArray] = {}
    attrs: dict[str, Any] = {}
    n_samples = batch.time.shape[0]

    if batch.stepper_state is not None:
        # to_state_dict()/from_state_dict() are the tensor intermediate; the
        # stepper stays opaque - this layer only maps its namespaced keys to
        # reserved netCDF variables and never inspects sub-state fields.
        state_dict = batch.stepper_state.to_cpu().to_state_dict()
        for key, tensor in state_dict.items():
            var_name = f"{_STEPPER_PREFIX}{key}"
            array, dtype_name = _to_storage_array(tensor)
            dims = _stepper_var_dims(var_name, array, n_samples)
            data_arrays[var_name] = xr.DataArray(
                array, dims=dims, attrs={DTYPE_ATTR: dtype_name}
            )

    if batch.labels is not None:
        array, dtype_name = _to_storage_array(batch.labels.tensor)
        data_arrays[_LABELS_VALUES_VAR] = xr.DataArray(
            array, dims=[SAMPLE_DIM, _LABEL_INDEX_DIM], attrs={DTYPE_ATTR: dtype_name}
        )
        attrs[_LABELS_NAMES_ATTR] = json.dumps(list(batch.labels.names))

    if batch.data_mask is not None:
        for name, mask in batch.data_mask.items():
            array, dtype_name = _to_storage_array(mask)
            data_arrays[f"{_DATA_MASK_PREFIX}{name}"] = xr.DataArray(
                array, dims=[SAMPLE_DIM], attrs={DTYPE_ATTR: dtype_name}
            )

    if data_arrays:
        attrs[SCHEMA_ATTR] = SCHEMA_VERSION
    return data_arrays, attrs


def has_restart_extras(ds: xr.Dataset) -> bool:
    """Whether ``ds`` carries embedded full-state extras (the schema marker)."""
    return SCHEMA_ATTR in ds.attrs


def _restore_tensor(da: xr.DataArray) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(da.values))
    dtype_name = da.attrs.get(DTYPE_ATTR)
    if dtype_name is not None:
        tensor = tensor.to(_DTYPE_BY_NAME[str(dtype_name)])
    return tensor


def decode_restart_extras(ds: xr.Dataset) -> RestartExtras:
    """Reconstruct the embedded extras from a full-state restart ``Dataset``.

    Assumes ``has_restart_extras(ds)`` is True. Any subselection on the sample
    dimension must already have been applied to ``ds`` so the recovered
    per-sample tensors line up with the prognostic variables.
    """
    state_dict: dict[str, torch.Tensor] = {}
    data_mask: dict[str, torch.Tensor] = {}
    for name in ds.data_vars:
        name_str = str(name)
        if name_str.startswith(_STEPPER_PREFIX):
            key = name_str[len(_STEPPER_PREFIX) :]
            state_dict[key] = _restore_tensor(ds[name])
        elif name_str.startswith(_DATA_MASK_PREFIX):
            var = name_str[len(_DATA_MASK_PREFIX) :]
            data_mask[var] = _restore_tensor(ds[name])

    stepper_state = StepperState.from_state_dict(state_dict) if state_dict else None

    labels: BatchLabels | None = None
    if _LABELS_VALUES_VAR in ds:
        names = json.loads(ds.attrs[_LABELS_NAMES_ATTR])
        labels = BatchLabels(_restore_tensor(ds[_LABELS_VALUES_VAR]), names=list(names))

    return RestartExtras(
        stepper_state=stepper_state,
        labels=labels,
        data_mask=data_mask or None,
    )
