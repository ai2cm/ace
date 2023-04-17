"""Routines for the ensemble netCDF data-format
"""
import xarray
import pathlib


def open(path: str, group="Somewhere"):
    path = pathlib.Path(path)
    ensemble_files = list(path.glob("*_?.nc"))
    ds = xarray.concat(
        [
            xarray.open_dataset(f, group=group).drop_isel(ensemble=0)
            for f in ensemble_files
        ],
        dim="ensemble",
    )
    template = xarray.open_dataset(ensemble_files[0])
    ds = ds.assign_coords(time=template.time)
    return ds
