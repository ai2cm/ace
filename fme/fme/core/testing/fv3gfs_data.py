import dataclasses
import pathlib
from typing import List, Dict, Optional, Tuple
from fme.fcn_training.utils.data_loader_params import DataLoaderParams

import numpy as np
import xarray as xr


@dataclasses.dataclass
class DimSizes:
    n_time: int
    n_lat: int
    n_lon: int
    nz_interface: int

    @property
    def shape_2d(self) -> Tuple[int, int, int]:
        return (self.n_time, self.n_lat, self.n_lon)

    @property
    def dims_2d(self) -> Tuple[str, str, str]:
        return ("time", "grid_yt", "grid_xt")

    @property
    def coords_2d(self) -> Dict[str, xr.DataArray]:
        return {
            name: xr.DataArray(
                np.arange(shape, dtype=np.float32),
                dims=(name,),
            )
            for name, shape in zip(self.dims_2d, self.shape_2d)
        }

    @property
    def shape_vertical_interface(self) -> Tuple[int]:
        return (self.nz_interface,)


def _save_2d_netcdf(
    filename,
    dim_sizes: DimSizes,
    variable_names: List[str],
    time_varying_values: Optional[List[float]] = None,
):
    data_vars = {}
    for name in variable_names:
        data = np.random.randn(*dim_sizes.shape_2d).astype(np.float32)
        data_vars[name] = xr.DataArray(
            data, dims=dim_sizes.dims_2d, attrs={"units": "m", "long_name": name}
        )
        if time_varying_values is not None:
            for i in range(dim_sizes.n_time):
                data_vars[name].sel(time=i)[:] = time_varying_values[i]

    for i in range(dim_sizes.nz_interface):
        data_vars[f"ak_{i}"] = float(i)
        data_vars[f"bk_{i}"] = float(i + 1)

    ds = xr.Dataset(data_vars=data_vars, coords=dim_sizes.coords_2d)
    ds.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")


def _save_scalar_netcdf(
    filename,
    variable_names: List[str],
):
    data_vars = {}
    for name in variable_names:
        data = np.random.randn()
        data_vars[name] = xr.DataArray(data, attrs={"units": "m", "long_name": name})

    ds = xr.Dataset(data_vars=data_vars)
    ds.to_netcdf(filename, format="NETCDF4_CLASSIC")


@dataclasses.dataclass
class StatsData:
    path: pathlib.Path
    names: List[str]

    def __post_init__(self):
        _save_scalar_netcdf(
            self.mean_filename,
            variable_names=self.names,
        )
        _save_scalar_netcdf(
            self.std_filename,
            variable_names=self.names,
        )

    @property
    def mean_filename(self):
        return self.path / "stats-mean.nc"

    @property
    def std_filename(self):
        return self.path / "stats-std.nc"


@dataclasses.dataclass
class FV3GFSData:
    path: pathlib.Path
    names: List[str]
    dim_sizes: DimSizes
    time_varying_values: Optional[List[float]] = None

    def __post_init__(self):
        self._data_path.mkdir(parents=True, exist_ok=True)
        if (
            self.time_varying_values is not None
            and len(self.time_varying_values) != self.dim_sizes.n_time
        ):
            raise ValueError(
                f"Number of time-varying values ({len(self.time_varying_values)}) "
                f"must match number of time steps ({self.dim_sizes.n_time})"
            )
        _save_2d_netcdf(
            self._data_filename,
            dim_sizes=self.dim_sizes,
            variable_names=self.names,
            time_varying_values=self.time_varying_values,
        )

    @property
    def _data_path(self):
        # data must be in a separate path as loader loads all *.nc files
        return self.path / "data"

    @property
    def _data_filename(self):
        return self._data_path / "data.nc"

    @property
    def data_loader_params(self) -> DataLoaderParams:
        return DataLoaderParams(
            str(self._data_path),
            data_type="FV3GFS",
            batch_size=1,
            num_data_workers=0,
        )
