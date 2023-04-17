import datetime
import json
import pathlib
from typing import Optional, Any

import xarray
from datasets.era5 import time

__all__ = ["open_34_vars", "open_hdf5"]

METADATA = pathlib.Path(__file__).parent / "data.json"


def open_hdf5(*, path, f=None, metadata, time_step=datetime.timedelta(hours=6)):
    dims = metadata["dims"]
    h5_path = metadata["h5_path"]

    ds = xarray.open_dataset(f or path, engine="h5netcdf", phony_dims="sort")
    array = ds[h5_path]
    ds = array.rename(dict(zip(array.dims, dims)))
    year = time.filename_to_year(path)
    n = array.shape[0]
    ds = ds.assign_coords(
        time=time.datetime_range(year, time_step=time_step, n=n), **metadata["coords"]
    )
    ds = ds.assign_attrs(metadata["attrs"], path=path)
    return ds


def open_34_vars(path: str, f: Optional[Any] = None) -> xarray.DataArray:
    """Open 34Vars hdf5 file

    Args:
        path: local path to hdf5 file
        f: an optional file-like object to load the data from. Useful for
            remote data and fsspec.

    Examples:

        >>> import datasets
        >>> path = "/out_of_sample/2018.h5"
        >>> datasets.era5.open_34_vars(path)
        <xarray.DataArray 'fields' (time: 1460, channel: 34, lat: 721, lon: 1440)>
        dask.array<array, shape=(1460, 34, 721, 1440), dtype=float32, chunksize=(1, 1, 721, 1440), chunktype=numpy.ndarray> # noqa
        Coordinates:
        * time     (time) datetime64[ns] 2018-01-01 ... 2018-12-31T18:00:00
        * lat      (lat) float64 90.0 89.75 89.5 89.25 ... -89.25 -89.5 -89.75 -90.0
        * lon      (lon) float64 0.0 0.25 0.5 0.75 1.0 ... 359.0 359.2 359.5 359.8
        * channel  (channel) <U5 'u10' 'v10' 't2m' 'sp' ... 'v900' 'z900' 't900'
        Attributes:
            selene_path:  /lustre/fsw/sw_climate_fno/34Var
            description:  ERA5 data at 6 hourly frequency with snapshots at 0000, 060...
            path:         /out_of_sample/2018.h5
    """

    metadata = json.loads(METADATA.read_text())
    return open_hdf5(path=path, f=f, metadata=metadata)
