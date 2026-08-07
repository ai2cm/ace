import asyncio
import dataclasses
import datetime
import functools
from collections.abc import Hashable, MutableMapping, Sequence
from typing import Literal

import numpy as np
import torch
import xarray as xr
import zarr

from fme.core.coordinates import (
    HEALPixCoordinates,
    HorizontalCoordinates,
    LatLonCoordinates,
)

SLICE_NONE = slice(None)

ZARRS_CODEC_PIPELINE = "zarrs.ZarrsCodecPipeline"


def accumulate_labels(labels: list[set[str] | None]) -> set[str] | None:
    """
    Accumulate labels from multiple sources.
    """
    if any(label is None for label in labels):
        if not all(label is None for label in labels):
            raise ValueError(
                "All datasets must have labels listed, "
                "or no labels must be listed. Provide an explicit empty list "
                "for datasets with no labels."
            )
        return None
    all_labels: set[str] = set()
    for label in labels:
        if label is None:
            raise RuntimeError("Check above should prevent this case.")
        all_labels.update(label)
    return all_labels


def _get_indexers(
    variable: xr.Variable, dims: Sequence[Hashable]
) -> tuple[slice | None, ...]:
    """Returns a tuple of indexers for the dimensions provided.

    Indexers select all data from dimensions that exist in the variable, and
    create new axes for dimensions that do not exist. The returned tuple will
    have the same length as the provided sequence of dimensions.

    Inspired by similar code in xarray:
    https://github.com/pydata/xarray/blob/1d43672574332615f225089d69f95a9f8d81d912/xarray/core/computation.py#L681-L688
    """
    indexers: list[slice | None] = []
    for dim in dims:
        if dim in variable.dims:
            indexers.append(SLICE_NONE)
        else:
            indexers.append(None)
    return tuple(indexers)


def as_alignable_tensor(
    variable: xr.Variable,
    dims: Sequence[Hashable],
) -> torch.tensor:
    """Load data from variable as a tensor with a singleton axis for each of the
    given dims that the variable does not have.

    The result is not yet broadcast to a particular shape, so it does not depend
    on the length of the time dimension and can be reused across samples.
    """
    arr = variable.values
    indexers = _get_indexers(variable, dims)
    return torch.as_tensor(arr[indexers])


def as_broadcasted_tensor(
    variable: xr.Variable,
    dims: Sequence[Hashable],
    shape: Sequence[int],
) -> torch.tensor:
    """Load data from variable and broadcast to tensor with the given shape."""
    return torch.broadcast_to(as_alignable_tensor(variable, dims), shape)


def _broadcast_array_to_tensor(
    array: np.ndarray,
    dims: Sequence[Hashable],
    shape: Sequence[int],
) -> torch.tensor:
    """Convert from numpy array to tensor and broadcast to the given shape.

    Note: if number of dimensions is different between input array and
    desired shape, it is assumed that array only has time dimension and will be
    broadcasted over all spatial dimensions.
    """
    tensor = torch.as_tensor(array)
    if len(array.shape) != len(shape):
        if array.shape[0] != shape[0]:
            raise ValueError("Must have matching time dimension")
        if len(array.shape) != 1:
            raise ValueError("If broadcasting, array must be 1D")
        # insert new singleton dimensions for missing spatial dimensions
        n_spatial_dims = len(dims) - 1
        tensor = tensor[(...,) + (None,) * n_spatial_dims]
    return torch.broadcast_to(tensor, shape)


def _get_array_selection(
    name: str, shape: tuple[int, ...], selection: tuple[slice | int, ...]
) -> tuple[slice | int, ...] | slice | int:
    """Resolve the selection to apply to an array with the given shape."""
    if len(shape) != len(selection):
        if len(shape) != 1:
            raise ValueError(
                f"Index selection slices/indices were provided as {selection} "
                f"but array {name} has {len(shape)} dimensions. "
            )
        # 1d scalar arrays are sliced along time dim
        return selection[0]
    return selection


# Opening a group and getting an array from it each read the corresponding
# zarr.json, so without caching every sample re-reads one metadata document per
# requested variable. The cached objects hold metadata and a store reference,
# and the set of keys is fixed by the configured datasets (files x variables),
# so these do not grow without bound at runtime.
@functools.cache
def _open_async_group(path: str):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(zarr.api.asynchronous.open(store=path))


@functools.cache
def _get_async_array(path: str, name: str):
    loop = asyncio.get_event_loop()
    # zarrs decodes chunks in Rust and does its own chunk IO, rather than going
    # through zarr-python's per-chunk buffer and store handling. The pipeline is
    # bound to an array when it is opened, so setting it here applies it to the
    # arrays read for training data without changing the pipeline used
    # elsewhere, such as by the inference data writers.
    with zarr.config.set({"codec_pipeline.path": ZARRS_CODEC_PIPELINE}):
        return loop.run_until_complete(_open_async_group(path).getitem(name))


async def _get_items(arrays_and_selections):
    return await asyncio.gather(
        *(array.getitem(selection) for array, selection in arrays_and_selections)
    )


def _load_all_variables_zarr_async(
    path: str, variables: Sequence[str], selection: tuple[slice | int, ...]
) -> MutableMapping[str, np.ndarray]:
    """Load data from a variables into memory.

    Assumes that path contains a zarr group and uses async.

    Assumes that the time dimension is the first dimension in the dataset.
    """
    # arrays are resolved before gathering so that the metadata reads hit the
    # cache instead of being repeated on every call
    arrays_and_selections = []
    for name in variables:
        async_array = _get_async_array(path, name)
        arrays_and_selections.append(
            (async_array, _get_array_selection(name, async_array.shape, selection))
        )
    loop = asyncio.get_event_loop()
    arrays = loop.run_until_complete(_get_items(arrays_and_selections))
    return {k: v for k, v in zip(variables, arrays)}


def _load_all_variables(
    ds: xr.Dataset, variables: Sequence[str], time_slice: slice = SLICE_NONE
) -> xr.Dataset:
    """Load data from a variables into memory.

    This function leverages xarray's lazy loading to load only the time slice
    (or chunk[s] for the time slice) of the variables we need.
    """
    if "time" in ds.dims:
        ds = ds.isel(time=time_slice)
    return ds[variables].compute()


@dataclasses.dataclass
class FillNaNsConfig:
    """
    Configuration to fill NaNs with a constant value or others.

    Parameters:
        method: Type of fill operation. Currently only 'constant' is supported.
        value: Value to fill NaNs with.
    """

    method: Literal["constant"] = "constant"
    value: float = 0.0


def load_series_data_zarr_async(
    idx: int,
    n_steps: int,
    path: str,
    names: list[str],
    final_dims: list[str],
    final_shape: list[int],
    nontime_selection: tuple[int | slice, ...],
    fill_nans: FillNaNsConfig | None = None,
):
    time_slice = slice(idx, idx + n_steps)
    selection = (time_slice, *nontime_selection)
    loaded = _load_all_variables_zarr_async(path, names, selection)
    if fill_nans is not None:
        for k, v in loaded.items():
            loaded[k] = np.nan_to_num(v, nan=fill_nans.value)
    arrays = {}
    for name in names:
        arrays[name] = _broadcast_array_to_tensor(loaded[name], final_dims, final_shape)
    return arrays


def load_series_data(
    idx: int,
    n_steps: int,
    ds: xr.Dataset,
    names: list[str],
    final_dims: list[str],
    final_shape: list[int],
    fill_nans: FillNaNsConfig | None = None,
):
    time_slice = slice(idx, idx + n_steps)
    loaded = _load_all_variables(ds, names, time_slice)
    # Fill NaNs after subsetting time slice to avoid triggering loading all
    # data, since we do not use dask.
    if fill_nans is not None:
        loaded = loaded.fillna(fill_nans.value)
    arrays = {}
    for n in names:
        variable = loaded[n].variable
        if len(variable.shape) != len(final_shape):
            if variable.shape[0] != final_shape[0]:
                raise ValueError("Must have matching time dimension")
            if len(variable.shape) != 1:
                raise ValueError("If broadcasting, array must be 1D")
        arrays[n] = as_broadcasted_tensor(variable, final_dims, final_shape)
    return arrays


def get_horizontal_coordinates(
    ds: xr.Dataset,
    spatial_dimensions: str,
    dtype: torch.dtype | None,
) -> tuple[HorizontalCoordinates, list[str]]:
    """Return the horizontal coordinate class and dimension names."""
    min_ndim = 3 if spatial_dimensions == "latlon" else 4
    coords: HorizontalCoordinates
    for da in ds.data_vars.values():
        if da.ndim >= min_ndim:
            dims = list(da.dims)
            break
    if spatial_dimensions == "latlon":
        lat_name, lon_name = dims[-2:]
        coords = LatLonCoordinates(
            lon=torch.tensor(ds[lon_name].values, dtype=dtype),
            lat=torch.tensor(ds[lat_name].values, dtype=dtype),
        )
        return coords, dims[-2:]
    elif spatial_dimensions == "healpix":
        face_name, height_name, width_name = dims[-3:]
        coords = HEALPixCoordinates(
            face=torch.tensor(ds[face_name].values, dtype=dtype),
            height=torch.tensor(ds[height_name].values, dtype=dtype),
            width=torch.tensor(ds[width_name].values, dtype=dtype),
        )
        return coords, dims[-3:]
    else:
        raise ValueError(
            f"spatial_dimensions must be either 'latlon' or 'healpix', "
            f"but got {spatial_dimensions}"
        )


def decode_timestep(microseconds: int) -> datetime.timedelta:
    return datetime.timedelta(microseconds=microseconds)


def encode_timestep(timedelta: datetime.timedelta) -> int:
    return timedelta // datetime.timedelta(microseconds=1)


def get_nonspacetime_dimensions(
    ds: xr.Dataset, horizontal_dims: list[str]
) -> list[str]:
    """Get all dimensions that are not time or horizontal dimensions."""
    nonspacetime_dims: list[str] = []
    dim_order: list[str] = []

    for da in ds.data_vars.values():
        if da.ndim >= len(horizontal_dims) + 1 and da.ndim > len(dim_order):
            dim_order = list(da.dims)
    for dim in dim_order:
        if dim not in horizontal_dims and dim != "time":
            nonspacetime_dims.append(dim)
    return nonspacetime_dims
