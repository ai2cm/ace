import dataclasses
import datetime
import warnings
from typing import List, Optional, Sequence, Tuple

import cftime
import dask
import numpy as np
import torch
import xarray as xr
from torch.utils.data import default_collate

from fme.core.typing_ import TensorMapping

SLICE_NONE = slice(None)


def infer_horizontal_dimension_names(ds: xr.Dataset) -> Tuple[str, str]:
    if "grid_xt" in ds.variables:
        hdims = "grid_xt", "grid_yt"
    elif "lon" in ds.variables:
        hdims = "lon", "lat"
    elif "longitude" in ds.variables:
        hdims = "longitude", "latitude"
    else:
        reference_da = None
        for da in ds.data_vars.values():
            if len(da.dims) == 3:
                reference_da = da
                break
        if reference_da is None:
            raise ValueError("Could not identify dataset's horizontal dimensions.")
        else:
            _, lat_dim, lon_dim = reference_da.dims
            warnings.warn(
                f"Familiar latitude and longitude coordinate names could not be "
                f"found in the dataset. Assuming that the trailing two dimensions, "
                f"{lat_dim!r} and {lon_dim!r}, represent latitude and longitude "
                f"respectively."
            )
            hdims = lon_dim, lat_dim
    return hdims


def _get_indexers(
    variable: xr.Variable, dims: Sequence[str]
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


def _load_variable(variable: xr.Variable, time_slice) -> np.ndarray:
    """Load data from a variable into memory.

    This function leverages xarray's lazy loading to load only the time slice
    of a variable we need. It assumes that, if present, "time" is the leading
    dimension of the array.
    """
    if "time" in variable.dims:
        return variable[time_slice, ...].values
    else:
        return variable.values


def as_broadcasted_tensor(
    variable: xr.Variable,
    dims: Sequence[str],
    shape: Sequence[int],
    time_slice: slice = SLICE_NONE,
) -> torch.tensor:
    """Load data from variable and broadcast to tensor with the given shape."""
    arr = _load_variable(variable, time_slice)
    indexers = _get_indexers(variable, dims)
    tensor = torch.as_tensor(arr[indexers])
    return torch.broadcast_to(tensor, shape)


def load_series_data(
    idx: int,
    n_steps: int,
    ds: xr.Dataset,
    names: List[str],
):
    time_slice = slice(idx, idx + n_steps)
    lon_dim, lat_dim = infer_horizontal_dimension_names(ds)
    dims = ("time", lat_dim, lon_dim)
    shape = (n_steps, ds.sizes[lat_dim], ds.sizes[lon_dim])
    # disable dask threading to avoid warnings
    with dask.config.set(scheduler="synchronous"):
        arrays = {}
        for n in names:
            variable = ds[n].variable
            arrays[n] = as_broadcasted_tensor(variable, dims, shape, time_slice)
        return arrays


def get_lons_and_lats(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    hdims = infer_horizontal_dimension_names(ds)
    lons, lats = ds[hdims[0]].values, ds[hdims[1]].values
    return np.array(lons, dtype=np.float32), np.array(lats, dtype=np.float32)


def get_times(ds: xr.Dataset, start: int, n_steps: int) -> xr.DataArray:
    """
    Get the time coordinate segment from the dataset, check that it's a
    cftime.datetime object, and return it is a data array (not a coordinate),
    so that it can be concatenated with other samples' times.
    """
    time_segment = ds["time"][slice(start, start + n_steps)]
    assert isinstance(
        time_segment[0].item(), cftime.datetime
    ), "time must be cftime.datetime."
    if len(time_segment) != n_steps:
        raise ValueError(
            f"Expected {n_steps} time steps, but got {len(time_segment)} instead."
        )
    return time_segment.drop_vars(["time"])


@dataclasses.dataclass
class BatchData:
    """A container for the data and time coordinates of a batch.

    Attributes:
        data: Data for each variable in each sample, concatenated along samples
            to make a batch. To be used directly in training, validation, and
            inference.
        times: An array of times for each sample in the batch, concatenated along
            samples to make a batch. To be used in writing out inference
            predictions with time coordinates, not directly in ML.

    """

    data: TensorMapping
    times: xr.DataArray

    @classmethod
    def from_sample_tuples(
        cls,
        samples: Sequence[Tuple[TensorMapping, xr.DataArray]],
        sample_dim_name: str = "sample",
    ) -> "BatchData":
        """
        Collate function for use with PyTorch DataLoader. Needed since samples contain
        both tensor mapping and xarray time coordinates, the latter of which we do
        not want to convert to tensors.
        """
        sample_data, sample_times = zip(*samples)
        batch_data = default_collate(sample_data)
        batch_times = xr.concat(sample_times, dim=sample_dim_name)
        return cls(batch_data, batch_times)


def decode_timestep(microseconds: int) -> datetime.timedelta:
    return datetime.timedelta(microseconds=microseconds)


def encode_timestep(timedelta: datetime.timedelta) -> int:
    return timedelta // datetime.timedelta(microseconds=1)
