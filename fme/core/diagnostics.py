import os
from collections.abc import Mapping
from typing import Protocol

import numpy as np
import xarray as xr

from fme.core.distributed import Distributed


class GetDataset(Protocol):
    def get_dataset(self) -> xr.Dataset: ...


def get_reduced_diagnostics(
    sub_aggregators: Mapping[str, GetDataset],
    coords: Mapping[str, np.ndarray],
) -> Mapping[str, xr.Dataset]:
    """
    Returns datasets from sub-aggregators. Coordinates are assigned to the datasets
    if they are present in the dataset.

    Args:
        sub_aggregators: dictionary of sub-aggregators.
        coords: dictionary of coordinates.

    Returns:
        Dictionary of datasets from aggregators.
    """
    datasets = {}
    for name, aggregator in sub_aggregators.items():
        ds = aggregator.get_dataset()
        ds_coords = {k: v for k, v in coords.items() if k in ds.dims}
        datasets[name] = ds.assign_coords(ds_coords)
    return datasets


def write_reduced_diagnostics(
    reduced_diagnostics: Mapping[str, xr.Dataset],
    output_dir: str,
    subdir: str | None = None,
):
    """Write the reduced metrics to disk. Each sub-aggregator will write a netCDF file
    if its `get_dataset` method returns a non-empty dataset.

    Args:
        reduced_diagnostics: Dictionary of reduced diagnostics datasets.
        output_dir: Output directory.
        subdir: Sub-directory within the output_dir.
    """
    if subdir is not None:
        output_dir = os.path.join(output_dir, subdir)
    dist = Distributed.get_instance()
    if dist.is_root():
        os.makedirs(output_dir, exist_ok=True)
        for name, ds in reduced_diagnostics.items():
            if len(ds) > 0:
                ds.to_netcdf(os.path.join(output_dir, f"{name}_diagnostics.nc"))
