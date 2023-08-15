import dask
import torch
import numpy as np
import xarray as xr

from typing import List, Optional, Tuple


def load_series_data(
    idx: int,
    n_steps: int,
    ds: xr.Dataset,
    names: List[str],
    window_time_slice: Optional[slice] = None,
):
    outer_slice = slice(idx, idx + n_steps)
    if window_time_slice is not None:
        if (
            window_time_slice.start is None
            or window_time_slice.stop is None
            or window_time_slice.step is not None
        ):
            raise ValueError(
                "lead_time must have a start and stop value, and no step value."
            )
        time_slice = apply_slice(
            outer_slice=slice(idx, idx + n_steps), inner_slice=window_time_slice
        )
    else:
        time_slice = outer_slice
    # disable dask threading to avoid warnings
    with dask.config.set(scheduler="synchronous"):
        arrays = {}
        for n in names:
            arr = ds.variables[n][time_slice, :, :]
            arrays[n] = torch.as_tensor(arr.values)
        return arrays


def apply_slice(outer_slice: slice, inner_slice: slice) -> slice:
    """
    Given two slices, return a new slice that is the result of applying the
    outer slice and then the inner slice.

    For example, array[outer_slice][inner_slice] is equivalent to
    array[apply_slice(outer_slice, inner_slice)].

    Requires that both outer_slice and inner_slice have a start and stop value,
    and that neither has a step value.
    """
    if outer_slice.start is None or inner_slice.start is None:
        raise ValueError("Both slices must have a start value.")
    if outer_slice.stop is None or inner_slice.stop is None:
        raise ValueError("Both slices must have a stop value.")
    if outer_slice.step is not None or inner_slice.step is not None:
        raise ValueError("Slices must not have a step value.")
    start = outer_slice.start + inner_slice.start
    stop_outer = outer_slice.stop
    stop_inner = outer_slice.start + inner_slice.stop
    stop = min(stop_outer, stop_inner)
    start = min(start, stop)
    return slice(start, stop)


def get_lons_and_lats(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    if "grid_xt" in ds.variables:
        hdims = "grid_xt", "grid_yt"
    elif "lon" in ds.variables:
        hdims = "lon", "lat"
    elif "longitude" in ds.variables:
        hdims = "longitude", "latitude"
    else:
        raise ValueError("Could not identify dataset's horizontal dimensions.")
    lons, lats = ds[hdims[0]].values, ds[hdims[1]].values
    return np.array(lons), np.array(lats)
