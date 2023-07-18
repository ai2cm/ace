import dask
import torch
import netCDF4
import numpy as np
import xarray as xr

from typing import List, Union


def load_series_data(
    idx: int, n_steps: int, ds: Union[netCDF4.MFDataset, xr.Dataset], names: List[str]
):
    # disable dask threading to avoid warnings in the xr.Dataset case
    with dask.config.set(scheduler="synchronous"):
        arrays = {}
        for n in names:
            # flip the lat dimension so that it is increasing
            arr = np.flip(ds.variables[n][idx : idx + n_steps, :, :], axis=-2)
            arr = arr.values.copy() if isinstance(ds, xr.Dataset) else arr.copy()
            arrays[n] = torch.as_tensor(arr)
        return arrays
