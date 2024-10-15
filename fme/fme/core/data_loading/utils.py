import datetime
import warnings
from typing import Hashable, List, Optional, Sequence, Tuple

import torch
import xarray as xr

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
) -> Tuple[Optional[slice], ...]:
    """Returns a tuple of indexers for the dimensions provided.

    Indexers select all data from dimensions that exist in the variable, and
    create new axes for dimensions that do not exist. The returned tuple will
    have the same length as the provided sequence of dimensions.

    Inspired by similar code in xarray:
    https://github.com/pydata/xarray/blob/1d43672574332615f225089d69f95a9f8d81d912/xarray/core/computation.py#L681-L688
    """
    indexers: List[Optional[slice]] = []
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


def _load_all_variables(
    ds: xr.Dataset, variables: Sequence[str], time_slice: slice = SLICE_NONE
) -> xr.Dataset:
    """Load data from a variables into memory.

    This function leverages xarray's lazy loading to load only the time slice
    (or chunk[s] for the time slice) of the variables we need.

    Consolidating the dask tasks into a single call of .compute() sped up remote
    zarr loads by nearly a factor of 2.
    """
    if "time" in ds.dims:
        ds = ds.isel(time=time_slice)
    return ds[variables].compute()


def load_series_data(
    idx: int,
    n_steps: int,
    ds: xr.Dataset,
    names: List[str],
    time_dim: Hashable,
    spatial_dim_names: List[str],
):
    time_slice = slice(idx, idx + n_steps)
    dims = [time_dim] + spatial_dim_names
    shape = [n_steps] + [ds.sizes[spatial_dim] for spatial_dim in dims[1:]]
    loaded = _load_all_variables(ds, names, time_slice)
    arrays = {}
    for n in names:
        variable = loaded[n].variable
        arrays[n] = as_broadcasted_tensor(variable, dims, shape)
    return arrays


def get_horizontal_dimensions(
    ds: xr.Dataset, dtype: Optional[torch.dtype]
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
