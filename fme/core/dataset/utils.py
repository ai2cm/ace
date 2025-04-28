import asyncio
import datetime
import warnings
from typing import Hashable, List, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import xarray as xr
import zarr

from fme.core.dataset.config import FillNaNsConfig

SLICE_NONE = slice(None)


def infer_horizontal_dimension_names(ds: xr.Dataset) -> List[str]:
    hdims: List[str]
    if "grid_xt" in ds.variables:
        hdims = ["grid_xt", "grid_yt"]
    elif "lon" in ds.variables:
        hdims = ["lon", "lat"]
    elif "longitude" in ds.variables:
        hdims = ["longitude", "latitude"]
    elif "face" in ds.variables:
        hdims = ["face", "height", "width"]
    else:
        reference_da = None
        for da in ds.data_vars.values():
            if len(da.dims) == 3:
                reference_da = da
                _, lat_dim, lon_dim = reference_da.dims
                warnings.warn(
                    f"Familiar latitude and longitude coordinate names could not be "
                    f"found in the dataset. Assuming that the trailing two dimensions, "
                    f"{lat_dim!r} and {lon_dim!r}, represent latitude and longitude "
                    f"of a lat/lon dataset respectively."
                )
                hdims = [lon_dim, lat_dim]
                break
            elif len(da.dims) == 4:
                reference_da = da
                _, face_dim, height_dim, width_dim = reference_da.dims
                warnings.warn(
                    f"Familiar latitude and longitude coordinate names could not be "
                    f"found in the dataset. Assuming that the trailing three "
                    f"dimensions, {face_dim!r}, {height_dim!r}, and {width_dim!r}, "
                    f"represent face, height, and width of a healpix dataset "
                    f" respectively."
                )
                hdims = [face_dim, height_dim, width_dim]
                break
        if reference_da is None:
            raise ValueError("Could not identify dataset's horizontal dimensions.")
    return hdims


def _get_indexers(
    variable: xr.Variable, dims: Sequence[Hashable]
) -> Tuple[slice | None, ...]:
    """Returns a tuple of indexers for the dimensions provided.

    Indexers select all data from dimensions that exist in the variable, and
    create new axes for dimensions that do not exist. The returned tuple will
    have the same length as the provided sequence of dimensions.

    Inspired by similar code in xarray:
    https://github.com/pydata/xarray/blob/1d43672574332615f225089d69f95a9f8d81d912/xarray/core/computation.py#L681-L688
    """
    indexers: List[slice | None] = []
    for dim in dims:
        if dim in variable.dims:
            indexers.append(SLICE_NONE)
        else:
            indexers.append(None)
    return tuple(indexers)


def as_broadcasted_tensor(
    variable: xr.Variable,
    dims: Sequence[Hashable],
    shape: Sequence[int],
) -> torch.tensor:
    """Load data from variable and broadcast to tensor with the given shape."""
    arr = variable.values
    indexers = _get_indexers(variable, dims)
    tensor = torch.as_tensor(arr[indexers])
    return torch.broadcast_to(tensor, shape)


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
        assert array.shape[0] == shape[0], "Must have matching time dimension"
        assert len(array.shape) == 1, "If broadcasting, array must be 1D"
        # insert new singleton dimensions for missing spatial dimensions
        n_spatial_dims = len(dims) - 1
        tensor = tensor[(...,) + (None,) * n_spatial_dims]
    return torch.broadcast_to(tensor, shape)


async def _get_item(group, name, selection):
    async_array = await group.getitem(name)
    return await async_array.getitem(selection)


async def _get_items(url, names, timestep):
    async_group = await zarr.api.asynchronous.open(store=url)
    coroutines = []
    for name in names:
        coroutines.append(_get_item(async_group, name, timestep))
    return await asyncio.gather(*coroutines)


def _load_all_variables_zarr_async(
    path: str, variables: Sequence[str], time_slice: slice = SLICE_NONE
) -> MutableMapping[str, np.ndarray]:
    """Load data from a variables into memory.

    Assumes that path contains a zarr group and uses async.

    Assumes that the time dimension is the first dimension in the dataset.
    """
    loop = asyncio.get_event_loop()
    arrays = loop.run_until_complete(_get_items(path, variables, time_slice))
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


def load_series_data_zarr_async(
    idx: int,
    n_steps: int,
    path: str,
    names: List[str],
    dims: List[str],
    shape: List[int],
    fill_nans: FillNaNsConfig | None = None,
):
    time_slice = slice(idx, idx + n_steps)
    loaded = _load_all_variables_zarr_async(path, names, time_slice)
    if fill_nans is not None:
        for k, v in loaded.items():
            loaded[k] = np.nan_to_num(v, nan=fill_nans.value)
    arrays = {}
    for n in names:
        arrays[n] = _broadcast_array_to_tensor(loaded[n], dims, shape)
    return arrays


def load_series_data(
    idx: int,
    n_steps: int,
    ds: xr.Dataset,
    names: List[str],
    dims: List[str],
    shape: List[int],
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
        arrays[n] = as_broadcasted_tensor(variable, dims, shape)
    return arrays


def get_horizontal_dimensions(
    ds: xr.Dataset, dtype: torch.dtype | None
) -> List[torch.Tensor]:
    hdims = infer_horizontal_dimension_names(ds)

    horizontal_values = []
    for dim in hdims:
        if dim in ds:
            horizontal_values.append(torch.tensor(ds[dim].values, dtype=dtype))
        else:
            raise ValueError(f"Expected {dim} in dataset: {ds}.")

    return horizontal_values


def decode_timestep(microseconds: int) -> datetime.timedelta:
    return datetime.timedelta(microseconds=microseconds)


def encode_timestep(timedelta: datetime.timedelta) -> int:
    return timedelta // datetime.timedelta(microseconds=1)
