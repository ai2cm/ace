import dataclasses
import json
import warnings
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, TypeVar

import cftime
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import default_collate

from fme.core.dataset.dataset import DatasetItem
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.labels import BatchLabels, LabelEncoding
from fme.core.random_state import RandomState
from fme.core.stepper_state import StepperState
from fme.core.tensors import repeat_interleave_batch_dim, unfold_ensemble_dim
from fme.core.typing_ import EnsembleTensorDict, TensorDict, TensorMapping

SelfType = TypeVar("SelfType", bound="BatchData")

# ``BatchData`` serializes to an xarray ``Dataset`` in which the prognostic
# ``data`` variables keep their plain names (so the file stays a normal,
# inspectable netCDF), and the round-trippable extras that a plain data+time
# file would drop (``stepper_state``, ``labels``, ``data_mask``) are embedded
# under this reserved prefix. The prefix cannot collide with a prognostic
# variable name. A dataset attribute records the schema version and doubles as
# the presence marker: a dataset without it carries no embedded state.
_RESERVED_PREFIX = "_fme_state__"
_SCHEMA_ATTR = "_fme_schema_version"
_SCHEMA_VERSION = 1
_DTYPE_ATTR = "_fme_dtype"
_SAMPLE_DIM = "sample"
_TIME_DIM = "time"

_STEPPER_PREFIX = f"{_RESERVED_PREFIX}stepper__"
_LABELS_VALUES_VAR = f"{_RESERVED_PREFIX}labels_values"
_LABELS_NAMES_ATTR = f"{_RESERVED_PREFIX}labels_names"
_LABEL_INDEX_DIM = f"{_RESERVED_PREFIX}label_index"
_DATA_MASK_PREFIX = f"{_RESERVED_PREFIX}data_mask__"

# netCDF4 has no native bool; bool tensors are stored as uint8 and cast back to
# their recorded dtype on read.
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


def _to_storage_array(tensor: torch.Tensor) -> tuple[np.ndarray, str]:
    """Return a netCDF-storable array plus the original torch dtype name.

    Bool is stored as uint8 (netCDF4 has no bool); the recorded dtype lets
    ``from_xarray_dataset`` restore the exact original dtype.
    """
    cpu = tensor.detach().cpu()
    dtype_name = str(cpu.dtype)
    if cpu.dtype == torch.bool:
        return cpu.to(torch.uint8).numpy(), dtype_name
    return cpu.numpy(), dtype_name


def _restore_tensor(da: xr.DataArray) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(da.values))
    dtype_name = da.attrs.get(_DTYPE_ATTR)
    if dtype_name is not None:
        tensor = tensor.to(_DTYPE_BY_NAME[str(dtype_name)])
    return tensor


def _reserved_var_dims(var_name: str, ndim: int, per_sample: bool) -> list[str]:
    """Dim names for a reserved-state variable.

    A ``per_sample`` variable (declared via the sub-state's
    ``per_sample_state_keys``, e.g. the corrector's ``global_dry_air_mass``)
    carries the shared ``sample`` dim on axis 0 so ``start_indices`` subselection
    (``ds.isel(sample=...)``) subsets it in step with the prognostic variables.
    Per-sample-ness is explicit, never inferred from a length matching the sample
    count: a variable that is not per-sample (the generator state) always uses
    private dims regardless of its length. Every non-sample axis gets a
    variable-private reserved dim so distinct variables never share (and thus
    constrain) a dim.
    """
    dims: list[str] = []
    for axis in range(ndim):
        if per_sample and axis == 0:
            dims.append(_SAMPLE_DIM)
        else:
            dims.append(f"{var_name}_d{axis}")
    return dims


def _check_device(data: TensorMapping, device: torch.device):
    for v in data.values():
        if v.device != device:
            raise ValueError(f"data must be on {device}")


def _collate_with_masking(
    sample_data: Sequence[TensorDict],
    sample_missing_names: Sequence[frozenset[str] | None],
) -> tuple[TensorDict, TensorDict | None]:
    """Collate samples that all share the same keys, building a mask from
    per-sample missing-variable metadata.

    Each sample dict is expected to contain all variables (NaN-filled for
    missing ones), so ``default_collate`` can stack them without Python
    loops over variables and samples.

    Args:
        sample_data: Per-sample dictionaries of tensors (all with the same keys).
        sample_missing_names: Per-sample frozensets naming which variables are
            NaN-filled placeholders, or None if all variables are present.

    Returns:
        A tuple of (batch_data, data_mask) where batch_data has all variables
        stacked along dim 0 and data_mask has per-variable boolean tensors of
        shape [n_samples] indicating presence, or None if all variables are
        present in all samples.
    """
    batch_data: TensorDict = default_collate(sample_data)

    has_any_missing = any(m is not None and len(m) > 0 for m in sample_missing_names)
    if not has_any_missing:
        return batch_data, None

    data_mask: TensorDict = {}
    any_masked = False

    for name in batch_data:
        present = torch.tensor(
            [name not in (m or frozenset()) for m in sample_missing_names]
        )
        data_mask[name] = present
        if not present.all():
            any_masked = True

    if not any_masked:
        return batch_data, None
    return batch_data, data_mask


class PrognosticState:
    """
    Thin typing wrapper around BatchData to indicate that the data is a prognostic
    state, such as an initial condition or final state when evolving forward in time.
    """

    def __init__(self, data: "BatchData"):
        """
        Initialize the state.

        Args:
            data: The data to initialize the state with.
        """
        self._data = data

    def to_device(self) -> "PrognosticState":
        return PrognosticState(self._data.to_device())

    def with_random_state(self, random_state: RandomState) -> "PrognosticState":
        """Return a copy with a seeded RandomState attached to its stepper_state.

        Used to seed stochastic inference: the random_state threads through the
        rollout via the stepper_state so the noise sequence is reproducible. Any
        other sub-state already present on the stepper_state is preserved.
        """
        stepper_state = self._data.stepper_state or StepperState()
        return PrognosticState(
            dataclasses.replace(
                self._data,
                stepper_state=dataclasses.replace(
                    stepper_state, random_state=random_state
                ),
            )
        )

    def as_batch_data(self) -> "BatchData":
        return self._data


@dataclasses.dataclass
class BatchData:
    """A container for the data and time coordinates of a batch.

    Parameters:
        data: Data for each variable in each sample of shape (sample, time, ...),
            concatenated along samples to make a batch. To be used directly in training,
            validation, and inference.
        time: An array representing time coordinates for each sample in the batch,
            concatenated along samples to make a batch. To be used in writing out
            inference predictions with time coordinates, not directly in ML.
        labels: Labels for each sample in the batch.
        horizontal_dims: Horizontal dimensions of the data. Used for writing to
            netCDF files.
        epoch: The epoch number for the batch data.
        n_ensemble: The number of ensemble members represented in the batch data.
            This is a suggestion for the purpose of computing ensemble metrics.
            For example, an ensemble is something you would want to compute CRPS
            or ensemble mean RMSE over.
        data_mask: Per-variable boolean tensors of shape ``[n_samples]`` where
            True means the variable is present for that sample. None when all
            variables are present in all samples.
        stepper_state: Opaque per-sample state owned by Stepper components
            (today only the corrector). ``None`` when no state has been
            seeded; the data loader never sets this — it is populated only
            by ``Stepper.predict`` to thread state across prediction windows.
    """

    data: TensorMapping
    time: xr.DataArray
    labels: BatchLabels | None = None
    horizontal_dims: list[str] = dataclasses.field(
        default_factory=lambda: ["lat", "lon"]
    )
    epoch: int | None = None
    n_ensemble: int = 1
    data_mask: TensorMapping | None = None
    stepper_state: StepperState | None = None

    @classmethod
    def new_for_testing(
        cls,
        names: Iterable[str],
        n_samples: int = 2,
        n_timesteps: int = 10,
        t_initial: cftime.datetime = cftime.datetime(2020, 1, 1),
        freq="6h",
        increment_times: bool = False,
        calendar="julian",
        img_shape: tuple[int, ...] = (9, 18),
        horizontal_dims: list[str] = ["lat", "lon"],
        epoch: int | None = 0,
        labels: BatchLabels | None = None,
        device: torch.device | None = None,
        data_mask: TensorMapping | None = None,
    ) -> "BatchData":
        """
        Create a new batch data object for testing.

        Args:
            names: The names of the variables to create.
            n_samples: The number of samples to create.
            n_timesteps: The number of timesteps to create.
            t_initial: The initial time.
            freq: The frequency of the time steps.
            increment_times: Whether to increment the initial time for each sample
                when creating the time coordinate.
            calendar: The calendar of the time steps.
            img_shape: The shape of the horizontal dimensions of the data.
            horizontal_dims: The horizontal dimensions of the data.
            epoch: The epoch number for the batch data.
            labels: The labels of the data.
            device: The device to create the data on. By default, the device is
                determined by the global device specified by get_device().
            data_mask: Boolean tensors of shape ``[n_samples]`` keyed by
                variable name, where True means the variable is present
                for that sample.  None when all variables are present.
        """
        if device is None:
            device = get_device()
        time = xr.DataArray(
            data=xr.date_range(
                start=t_initial,
                periods=n_timesteps,
                freq=freq,
                calendar=calendar,
                use_cftime=True,
            ),
            dims=["time"],
        ).drop_vars(["time"])
        if increment_times:
            sample_times = xr.concat(
                [time + pd.to_timedelta(freq) * i for i in range(n_samples)],
                dim="sample",
            )
        else:
            sample_times = xr.concat([time] * n_samples, dim="sample")
        return BatchData(
            data={
                k: torch.randn(n_samples, n_timesteps, *img_shape).to(device)
                for k in names
            },
            time=sample_times,
            labels=labels,
            horizontal_dims=horizontal_dims,
            epoch=epoch,
            data_mask=data_mask,
        )

    @property
    def dims(self) -> list[str]:
        return ["sample", "time"] + self.horizontal_dims

    @property
    def n_timesteps(self) -> int:
        return self.time["time"].values.size

    @property
    def ensemble_data(self) -> EnsembleTensorDict:
        """
        Add an explicit ensemble dimension to a data tensor dict.

        Returns:
            The tensor dict with an explicit ensemble dimension.
        """
        return unfold_ensemble_dim(TensorDict(self.data), n_ensemble=self.n_ensemble)

    def to_device(self) -> "BatchData":
        device = get_device()
        return self.__class__(
            data={k: v.to(device) for k, v in self.data.items()},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
            labels=self.labels.to(device) if self.labels is not None else None,
            n_ensemble=self.n_ensemble,
            data_mask=(
                {k: v.to(device) for k, v in self.data_mask.items()}
                if self.data_mask is not None
                else None
            ),
            stepper_state=(
                self.stepper_state.to_device()
                if self.stepper_state is not None
                else None
            ),
        )

    def scatter_spatial(self, global_img_shape: tuple[int, int]) -> "BatchData":
        """Slice data tensors to the local spatial chunk."""
        dist = Distributed.get_instance()
        return self.__class__(
            data=dist.scatter_spatial(dict(self.data), global_img_shape),
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
            labels=self.labels,
            n_ensemble=self.n_ensemble,
            data_mask=self.data_mask,
            stepper_state=self.stepper_state,
        )

    def to_cpu(self) -> "BatchData":
        return self.__class__(
            data={k: v.cpu() for k, v in self.data.items()},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
            labels=self.labels.to("cpu") if self.labels is not None else None,
            n_ensemble=self.n_ensemble,
            data_mask=(
                {k: v.cpu() for k, v in self.data_mask.items()}
                if self.data_mask is not None
                else None
            ),
            stepper_state=(
                self.stepper_state.to_cpu() if self.stepper_state is not None else None
            ),
        )

    @classmethod
    def _get_kwargs(cls, horizontal_dims: list[str] | None) -> dict[str, Any]:
        if horizontal_dims is None:
            kwargs = {}
        else:
            kwargs = {"horizontal_dims": horizontal_dims}
        return kwargs

    @classmethod
    def new_on_cpu(
        cls,
        data: TensorMapping,
        time: xr.DataArray,
        epoch: int | None = None,
        labels: BatchLabels | None = None,
        horizontal_dims: list[str] | None = None,
        n_ensemble: int = 1,
        data_mask: TensorMapping | None = None,
        stepper_state: StepperState | None = None,
    ) -> "BatchData":
        _check_device(data, torch.device("cpu"))
        if labels is not None:
            if labels.tensor.device != torch.device("cpu"):
                raise ValueError(f"labels must be on cpu, got {labels.tensor.device}")
        kwargs = cls._get_kwargs(horizontal_dims)
        if isinstance(labels, list):
            warnings.warn(
                "Passing labels as a list is deprecated, and they will be ignored. "
                "Please pass a BatchLabels object "
                "instead, or None to indicate no label information.",
                DeprecationWarning,
            )
            labels = None
        return BatchData(
            data=data,
            time=time,
            labels=labels,
            epoch=epoch,
            n_ensemble=n_ensemble,
            data_mask=data_mask,
            stepper_state=stepper_state,
            **kwargs,
        )

    @classmethod
    def new_on_device(
        cls,
        data: TensorMapping,
        time: xr.DataArray,
        epoch: int | None = None,
        labels: BatchLabels | None = None,
        horizontal_dims: list[str] | None = None,
        n_ensemble: int = 1,
        data_mask: TensorMapping | None = None,
        stepper_state: StepperState | None = None,
    ) -> "BatchData":
        """
        Move the data to the current global device specified by get_device().
        """
        _check_device(data, get_device())
        kwargs = cls._get_kwargs(horizontal_dims)
        if isinstance(labels, list):
            warnings.warn(
                "Passing labels as a list is deprecated, and they will be ignored. "
                "Please pass a BatchLabels object "
                "instead, or None to indicate no label information.",
                DeprecationWarning,
            )
            labels = None
        return BatchData(
            data=data,
            time=time,
            epoch=epoch,
            labels=labels,
            n_ensemble=n_ensemble,
            data_mask=data_mask,
            stepper_state=stepper_state,
            **kwargs,
        )

    @staticmethod
    def dataset_has_embedded_state(ds: xr.Dataset) -> bool:
        """Whether ``ds`` carries embedded ``BatchData`` state.

        True iff it was produced by ``to_xarray_dataset`` with extras present
        (the schema-version marker attribute). Used by the restart reader to
        decide between ``from_xarray_dataset`` and the lenient plain reader.
        """
        return _SCHEMA_ATTR in ds.attrs

    def validate_initial_condition(
        self,
        prognostic_names: Collection[str],
        labels: list[str] | None,
        n_ensemble: int,
    ) -> None:
        """Check that config values are consistent with this loaded full-state
        initial condition.

        A full-state restart already had its prognostic names, labels, and
        ensemble broadcast applied before it was saved, so on load we validate
        rather than re-derive: the config must agree with what was saved, and a
        mismatch is an error (never a silent re-application).

        Raises:
            ValueError: if any ``prognostic_names`` is absent from ``data``; if
                ``labels`` is given but does not match the loaded labels (including
                the loaded-None case); or if ``n_ensemble`` differs from the
                loaded state (a full-state restart cannot be re-broadcast - its
                sample dimension already carries the ensemble).
        """
        missing = [name for name in prognostic_names if name not in self.data]
        if missing:
            raise ValueError(
                f"Loaded initial condition is missing prognostic variables "
                f"{missing}. Present variables: {sorted(self.data)}."
            )
        if labels is not None:
            if self.labels is None:
                raise ValueError(
                    f"Config provided labels {labels} but the loaded initial "
                    "condition carries none."
                )
            if self.labels.names != list(labels):
                raise ValueError(
                    f"Config labels {list(labels)} do not match the loaded initial "
                    f"condition's labels {self.labels.names}."
                )
        if n_ensemble != self.n_ensemble:
            raise ValueError(
                f"Requested n_ensemble={n_ensemble} but the loaded full-state "
                f"initial condition represents n_ensemble={self.n_ensemble}. A "
                "full-state restart cannot be re-broadcast: its sample dimension "
                "already carries the ensemble."
            )

    def to_xarray_dataset(self) -> xr.Dataset:
        """Serialize this ``BatchData`` to a single xarray ``Dataset``.

        The prognostic ``data`` variables keep their plain names and ``time`` is
        written alongside, so the file stays a normal, inspectable netCDF; a
        single-timestep batch has its length-1 time dimension squeezed (the
        restart presentation). The round-trippable extras a plain data+time file
        would drop - ``stepper_state``, ``labels``, ``data_mask`` - are embedded
        under reserved ``_fme_state__`` variables with a schema-version marker
        attribute, but only when present, so a batch carrying none of them
        serializes to a plain data+time dataset. Inverse: ``from_xarray_dataset``.

        Structural fields (``n_ensemble``, ``horizontal_dims``, ``epoch``) are
        not serialized: ``horizontal_dims`` is recovered from the variable dims
        on load, and ``n_ensemble`` is intentionally left to be re-derived (the
        sample dimension already carries any broadcast ensemble).
        """
        if self.time.sizes[_TIME_DIM] == 1:
            time_dim = self.dims.index(_TIME_DIM)
            dims_to_write = self.dims[:time_dim] + self.dims[time_dim + 1 :]

            def present(x: torch.Tensor) -> torch.Tensor:
                return x.squeeze(dim=time_dim)

            time_array = self.time.isel(time=0)
        else:
            dims_to_write = self.dims

            def present(x: torch.Tensor) -> torch.Tensor:
                return x

            time_array = self.time

        data_arrays: dict[str, xr.DataArray] = {}
        for name, tensor in self.data.items():
            data_arrays[name] = xr.DataArray(
                present(tensor).detach().cpu().numpy(), dims=dims_to_write
            )
        data_arrays[_TIME_DIM] = time_array

        extra_arrays, attrs = self._encode_reserved_state()
        data_arrays.update(extra_arrays)
        ds = xr.Dataset(data_arrays)
        ds.attrs.update(attrs)
        return ds

    @classmethod
    def from_xarray_dataset(cls, ds: xr.Dataset) -> "BatchData":
        """Reconstruct a ``BatchData`` from ``to_xarray_dataset``'s output.

        Strict inverse of ``to_xarray_dataset``: recovers the prognostic data,
        time, and any embedded ``stepper_state``/``labels``/``data_mask``,
        restoring exact tensor dtypes. Only guaranteed for datasets produced by
        ``to_xarray_dataset``; it is NOT intended to interpret arbitrary external
        IC datasets or legacy plain restart files (those go through the lenient
        reader in ``get_initial_condition``). ``horizontal_dims`` is recovered
        from the prognostic variables' dims.
        """
        time_da = ds[_TIME_DIM]
        squeezed = list(time_da.dims) == [_SAMPLE_DIM]
        if squeezed:
            time = xr.DataArray(time_da.values[:, None], dims=[_SAMPLE_DIM, _TIME_DIM])
        else:
            time = xr.DataArray(time_da.values, dims=[_SAMPLE_DIM, _TIME_DIM])

        data: dict[str, torch.Tensor] = {}
        horizontal_dims: list[str] | None = None
        for name in ds.data_vars:
            name_str = str(name)
            if name_str == _TIME_DIM or name_str.startswith(_RESERVED_PREFIX):
                continue
            da = ds[name]
            tensor = torch.as_tensor(np.asarray(da.values))
            if squeezed:
                tensor = tensor.unsqueeze(1)
            data[name_str] = tensor
            if horizontal_dims is None:
                horizontal_dims = [
                    str(d) for d in da.dims if d not in (_SAMPLE_DIM, _TIME_DIM)
                ]

        stepper_state, labels, data_mask = cls._decode_reserved_state(ds)
        return cls.new_on_cpu(
            data=data,
            time=time,
            labels=labels,
            data_mask=data_mask,
            stepper_state=stepper_state,
            horizontal_dims=horizontal_dims,
        )

    def _encode_reserved_state(
        self,
    ) -> tuple[dict[str, xr.DataArray], dict[str, Any]]:
        """Reserved data variables and dataset attributes embedding the
        round-trippable extras (stepper_state, labels, data_mask), or ``({}, {})``
        when none are present.
        """
        data_arrays: dict[str, xr.DataArray] = {}
        attrs: dict[str, Any] = {}

        if self.stepper_state is not None:
            # to_state_dict()/from_state_dict() are the tensor intermediate; the
            # stepper stays opaque - this layer only maps its namespaced keys to
            # reserved variables and never inspects sub-state fields. Per-sample
            # variables are marked explicitly by the stepper's declaration, not
            # inferred from a length matching the sample count.
            state_dict = self.stepper_state.to_cpu().to_state_dict()
            per_sample_keys = self.stepper_state.per_sample_state_keys()
            for key, tensor in state_dict.items():
                var_name = f"{_STEPPER_PREFIX}{key}"
                array, dtype_name = _to_storage_array(tensor)
                data_arrays[var_name] = xr.DataArray(
                    array,
                    dims=_reserved_var_dims(
                        var_name, array.ndim, per_sample=key in per_sample_keys
                    ),
                    attrs={_DTYPE_ATTR: dtype_name},
                )

        if self.labels is not None:
            array, dtype_name = _to_storage_array(self.labels.tensor)
            data_arrays[_LABELS_VALUES_VAR] = xr.DataArray(
                array,
                dims=[_SAMPLE_DIM, _LABEL_INDEX_DIM],
                attrs={_DTYPE_ATTR: dtype_name},
            )
            attrs[_LABELS_NAMES_ATTR] = json.dumps(list(self.labels.names))

        if self.data_mask is not None:
            for name, mask in self.data_mask.items():
                array, dtype_name = _to_storage_array(mask)
                data_arrays[f"{_DATA_MASK_PREFIX}{name}"] = xr.DataArray(
                    array, dims=[_SAMPLE_DIM], attrs={_DTYPE_ATTR: dtype_name}
                )

        if data_arrays:
            attrs[_SCHEMA_ATTR] = _SCHEMA_VERSION
        return data_arrays, attrs

    @classmethod
    def _decode_reserved_state(
        cls, ds: xr.Dataset
    ) -> tuple[StepperState | None, BatchLabels | None, dict[str, torch.Tensor] | None]:
        state_dict: dict[str, torch.Tensor] = {}
        data_mask: dict[str, torch.Tensor] = {}
        for name in ds.data_vars:
            name_str = str(name)
            if name_str.startswith(_STEPPER_PREFIX):
                state_dict[name_str[len(_STEPPER_PREFIX) :]] = _restore_tensor(ds[name])
            elif name_str.startswith(_DATA_MASK_PREFIX):
                data_mask[name_str[len(_DATA_MASK_PREFIX) :]] = _restore_tensor(
                    ds[name]
                )

        stepper_state = StepperState.from_state_dict(state_dict) if state_dict else None
        labels: BatchLabels | None = None
        if _LABELS_VALUES_VAR in ds:
            names = json.loads(ds.attrs[_LABELS_NAMES_ATTR])
            labels = BatchLabels(
                _restore_tensor(ds[_LABELS_VALUES_VAR]), names=list(names)
            )
        return stepper_state, labels, (data_mask or None)

    def __post_init__(self):
        if len(self.time.shape) != 2:
            raise ValueError(
                "Expected time to have shape (n_samples, n_times), got shape "
                f"{self.time.shape}."
            )
        for k, v in self.data.items():
            if v.shape[:2] != self.time.shape[:2]:
                raise ValueError(
                    f"Data for variable {k} has shape {v.shape}, expected shape "
                    f"(n_samples, n_times) for time but got shape "
                    f"{self.time.shape}."
                )
        if (
            self.labels is not None
            and self.labels.tensor.shape[0] != self.time.shape[0]
        ):
            raise ValueError(
                "Labels tensor first dimension must match number of samples in "
                f"time. Got labels shape {self.labels.tensor.shape} and time shape "
                f"{self.time.shape}."
            )
        if self.data_mask is not None:
            n_samples = self.time.shape[0]
            for k, v in self.data_mask.items():
                if v.shape != (n_samples,):
                    raise ValueError(
                        f"data_mask for variable {k} has shape {v.shape}, "
                        f"expected ({n_samples},)."
                    )
        if self.stepper_state is not None:
            ss_n = self.stepper_state.sample_dim_size()
            if ss_n is not None and ss_n != self.time.shape[0]:
                raise ValueError(
                    f"stepper_state leading dim {ss_n} does not match the number "
                    f"of samples in time ({self.time.shape[0]})."
                )

    @classmethod
    def from_sample_tuples(
        cls,
        samples: Sequence[DatasetItem],
        sample_dim_name: str = "sample",
        horizontal_dims: list[str] | None = None,
        label_encoding: LabelEncoding | None = None,
        allow_missing_variables: bool = False,
    ) -> "BatchData":
        (
            sample_data,
            sample_times,
            sample_labels,
            sample_epochs,
            sample_missing_names,
        ) = zip(*samples)
        if not all(epoch == sample_epochs[0] for epoch in sample_epochs):
            raise ValueError("All samples must have the same epoch.")
        if allow_missing_variables:
            batch_data, data_mask = _collate_with_masking(
                sample_data, sample_missing_names
            )
        else:
            batch_data = default_collate(sample_data)
            data_mask = None
        batch_time = xr.concat(sample_times, dim=sample_dim_name)
        if label_encoding is None:
            if sample_labels[0] is not None:
                raise ValueError("label_encoding must be provided if labels are used.")
            labels = None
        else:
            labels = label_encoding.encode(list(sample_labels), device="cpu")
        return BatchData.new_on_cpu(
            data=batch_data,
            time=batch_time,
            labels=labels,
            horizontal_dims=horizontal_dims,
            epoch=sample_epochs[0],
            data_mask=data_mask,
        )

    def compute_derived_variables(
        self: SelfType,
        derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
        forcing_data: SelfType,
    ) -> SelfType:
        """
        Compute derived variables from the data and forcing data.

        The forcing data must have the same time coordinate as the batch data.

        Args:
            derive_func: A function that takes the data and forcing data and returns a
                dictionary of derived variables.
            forcing_data: The forcing data to compute derived variables from.
        """
        if not np.all(forcing_data.time.values == self.time.values):
            raise ValueError(
                "Forcing data must have the same time coordinate as the batch data."
            )
        derived_data = derive_func(self.data, forcing_data.data)
        return self.__class__(
            data={**self.data, **derived_data},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
            labels=self.labels,
            n_ensemble=self.n_ensemble,
            data_mask=self.data_mask,
            stepper_state=self.stepper_state,
        )

    def remove_initial_condition(self: SelfType, n_ic_timesteps: int) -> SelfType:
        """
        Remove the initial condition timesteps from the data.
        """
        if n_ic_timesteps == 0:
            raise RuntimeError("No initial condition timesteps to remove.")
        return self.__class__(
            {k: v[:, n_ic_timesteps:] for k, v in self.data.items()},
            time=self.time.isel(time=slice(n_ic_timesteps, None)),
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
            labels=self.labels,
            n_ensemble=self.n_ensemble,
            data_mask=self.data_mask,
            stepper_state=self.stepper_state,
        )

    def subset_names(self: SelfType, names: Collection[str]) -> SelfType:
        """
        Subset the data to only include the given names.
        """
        if self.data_mask is not None:
            data_mask = {k: v for k, v in self.data_mask.items() if k in names}
        else:
            data_mask = None
        return self.__class__(
            {k: v for k, v in self.data.items() if k in names},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
            labels=self.labels,
            n_ensemble=self.n_ensemble,
            data_mask=data_mask,
            stepper_state=self.stepper_state,
        )

    def get_start(
        self: SelfType, prognostic_names: Collection[str], n_ic_timesteps: int
    ) -> PrognosticState:
        """
        Get the initial condition state.
        """
        return PrognosticState(
            self.subset_names(prognostic_names).select_time_slice(
                slice(0, n_ic_timesteps)
            )
        )

    def get_end(
        self: SelfType, prognostic_names: Collection[str], n_ic_timesteps: int
    ) -> PrognosticState:
        """
        Get the final state which can be used as a new initial condition.
        """
        return PrognosticState(
            self.subset_names(prognostic_names).select_time_slice(
                slice(-n_ic_timesteps, None)
            )
        )

    def select_time_slice(self: SelfType, time_slice: slice) -> SelfType:
        """
        Select a window of data from the batch.
        """
        return self.__class__(
            {k: v[:, time_slice] for k, v in self.data.items()},
            time=self.time[:, time_slice],
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
            labels=self.labels,
            n_ensemble=self.n_ensemble,
            data_mask=self.data_mask,
            stepper_state=self.stepper_state,
        )

    def prepend(self: SelfType, initial_condition: PrognosticState) -> SelfType:
        """
        Prepend the initial condition to the data.
        """
        initial_batch_data = initial_condition.as_batch_data()
        filled_data = {**initial_batch_data.data}
        example_tensor = list(initial_batch_data.data.values())[0]
        state_data_device = list(self.data.values())[0].device
        for k in self.data:
            if k not in filled_data:
                filled_data[k] = torch.full_like(example_tensor, fill_value=np.nan)
        return self.__class__(
            data={
                k: torch.cat([filled_data[k].to(state_data_device), v], dim=1)
                for k, v in self.data.items()
            },
            time=xr.concat([initial_batch_data.time, self.time], dim="time"),
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
            labels=self.labels,
            n_ensemble=self.n_ensemble,
            data_mask=self.data_mask,
            stepper_state=self.stepper_state,
        )

    def broadcast_ensemble(self: SelfType, n_ensemble: int) -> SelfType:
        """
        Broadcast a singleton ensemble to a new BatchData obj with n_ensemble members
        per ensemble.
        """
        if self.n_ensemble != 1:
            raise ValueError(
                "Can only broadcast singleton ensembles, but this BatchData has "
                f"n_ensemble={self.n_ensemble} and cannot be broadcast."
            )
        data = repeat_interleave_batch_dim(self.data, n_ensemble)
        # Repeat-interleave the time coordinate to match the block ordering of
        # ``repeat_interleave_batch_dim`` (and of the labels/data_mask below):
        # sample s lands at positions [s * n_ensemble, (s + 1) * n_ensemble). A
        # plain ``xr.concat([time] * n_ensemble)`` would instead tile the samples
        # ([s0, s1, ..., s0, s1, ...]), misaligning data and time whenever the
        # samples carry distinct times (e.g. inference ICs at different start
        # dates with n_ensemble_per_ic > 1).
        n_samples = self.time.sizes["sample"]
        time = self.time.isel(sample=np.repeat(np.arange(n_samples), n_ensemble))
        if self.labels is None:
            labels = None
        else:
            labels = BatchLabels(
                torch.repeat_interleave(self.labels.tensor, n_ensemble, dim=0),
                self.labels.names,
            )
        if self.data_mask is None:
            data_mask = None
        else:
            data_mask = {
                k: torch.repeat_interleave(v, n_ensemble, dim=0)
                for k, v in self.data_mask.items()
            }
        # Keep tensors on the same device as input. Do not move to get_device() here:
        # from a DataLoader worker (e.g. InferenceDataset with n_ensemble > 1), data
        # must stay on CPU so the loader can pin_memory() and transfer to GPU later.
        return self.__class__(
            data=data,
            time=time,
            horizontal_dims=self.horizontal_dims,
            labels=labels,
            epoch=self.epoch,
            n_ensemble=n_ensemble,
            data_mask=data_mask,
            stepper_state=(
                self.stepper_state.broadcast_ensemble(n_ensemble)
                if self.stepper_state is not None
                else None
            ),
        )

    def pin_memory(self: SelfType) -> SelfType:
        """Used by torch.utils.data.DataLoader when pin_memory=True to page-lock
        tensors in CPU memory, resulting in faster transfers from CPU to GPU.

        See https://docs.pytorch.org/docs/stable/data.html#memory-pinning

        """
        self.data = {name: tensor.pin_memory() for name, tensor in self.data.items()}
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        if self.data_mask is not None:
            self.data_mask = {
                name: tensor.pin_memory() for name, tensor in self.data_mask.items()
            }
        if self.stepper_state is not None:
            self.stepper_state.pin_memory()
        return self


@dataclasses.dataclass
class PairedData:
    """A container for the data and time coordinate of a batch, with paired
    prediction and target data.
    """

    prediction: TensorMapping
    reference: TensorMapping
    time: xr.DataArray
    labels: BatchLabels | None = None
    n_ensemble: int = 1

    @property
    def forcing(self) -> TensorMapping:
        return {k: v for k, v in self.reference.items() if k not in self.prediction}

    @property
    def target(self) -> TensorMapping:
        return {k: v for k, v in self.reference.items() if k in self.prediction}

    def broadcast_ensemble(self, n_ensemble: int) -> "PairedData":
        """
        Add an explicit ensemble dimension to a PairedData tensors.

        Returns:
            The tensor dict with an explicit ensemble dimension.
        """
        if self.n_ensemble != 1:
            raise ValueError(
                "Can only broadcast singleton ensembles, but this PairedData has "
                f"n_ensemble={self.n_ensemble} and cannot be broadcast."
            )
        prediction = repeat_interleave_batch_dim(self.prediction, n_ensemble)
        reference = repeat_interleave_batch_dim(self.reference, n_ensemble)
        # Match the block ordering of repeat_interleave_batch_dim (see
        # BatchData.broadcast_ensemble): repeat-interleave time per sample rather
        # than tiling, so data and time stay aligned for distinct-time samples.
        n_samples = self.time.sizes["sample"]
        time = self.time.isel(sample=np.repeat(np.arange(n_samples), n_ensemble))
        if self.labels is None:
            labels = None
        else:
            labels = BatchLabels(
                torch.repeat_interleave(self.labels.tensor, n_ensemble, dim=0),
                self.labels.names,
            )
        return PairedData(
            prediction={k: v.to(get_device()) for k, v in prediction.items()},
            reference={k: v.to(get_device()) for k, v in reference.items()},
            time=time,
            labels=labels,
            n_ensemble=n_ensemble,
        )

    def as_ensemble_tensor_dicts(
        self, n_ensemble: int
    ) -> tuple[EnsembleTensorDict, EnsembleTensorDict]:
        """
        Unfold the batch dimension into an explicit ensemble dimension.

        Returns:
            (unfolded_reference, unfolded_prediction) for use with ensemble
            aggregators. Each has shape (n_batch, n_ensemble, ...).
        """
        unfolded_reference = unfold_ensemble_dim(
            TensorDict(self.reference), n_ensemble=n_ensemble
        )
        unfolded_prediction = unfold_ensemble_dim(
            TensorDict(self.prediction), n_ensemble=n_ensemble
        )
        return (unfolded_reference, unfolded_prediction)

    @classmethod
    def from_batch_data(
        cls,
        prediction: BatchData,
        reference: BatchData,
    ) -> "PairedData":
        if not np.all(prediction.time.values == reference.time.values):
            raise ValueError("Prediction and target time coordinate must be the same.")
        return PairedData(
            prediction=prediction.data,
            reference=reference.data,
            labels=prediction.labels,
            time=prediction.time,
            n_ensemble=prediction.n_ensemble,
        )

    @classmethod
    def new_on_device(
        cls,
        prediction: TensorMapping,
        reference: TensorMapping,
        time: xr.DataArray,
        labels: BatchLabels | None = None,
        n_ensemble: int = 1,
    ) -> "PairedData":
        device = get_device()
        _check_device(prediction, device)
        _check_device(reference, device)
        return PairedData(
            prediction=prediction,
            reference=reference,
            labels=labels,
            time=time,
            n_ensemble=n_ensemble,
        )

    @classmethod
    def new_on_cpu(
        cls,
        prediction: TensorMapping,
        reference: TensorMapping,
        time: xr.DataArray,
        labels: BatchLabels | None = None,
        n_ensemble: int = 1,
    ) -> "PairedData":
        _check_device(prediction, torch.device("cpu"))
        _check_device(reference, torch.device("cpu"))
        return PairedData(
            prediction=prediction,
            reference=reference,
            labels=labels,
            time=time,
            n_ensemble=n_ensemble,
        )
