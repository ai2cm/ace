import dataclasses
from typing import List, Mapping, Sequence, Tuple

import cftime
import dask
import numpy as np
import torch
import xarray as xr
from torch.utils.data import default_collate


def load_series_data(
    idx: int,
    n_steps: int,
    ds: xr.Dataset,
    names: List[str],
):
    time_slice = slice(idx, idx + n_steps)
    # disable dask threading to avoid warnings
    with dask.config.set(scheduler="synchronous"):
        arrays = {}
        for n in names:
            arr = ds.variables[n][time_slice, :, :]
            arrays[n] = torch.as_tensor(arr.values)
        return arrays


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

    data: Mapping[str, torch.Tensor]
    times: xr.DataArray

    @classmethod
    def from_sample_tuples(
        cls,
        samples: Sequence[Tuple[Mapping[str, torch.Tensor], xr.DataArray]],
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
