import dataclasses
import datetime
import pathlib
from collections.abc import Iterable, Sequence
from typing import Any

import cftime
import numpy as np
import xarray as xr

from fme.ace.data_loading.inference import (
    InferenceDataLoaderConfig,
    InferenceInitialConditionIndices,
)
from fme.core.coordinates import DimSize
from fme.core.dataset.config import XarrayDataConfig


def _coord_value(
    name, size, timestep: datetime.timedelta = datetime.timedelta(hours=6)
):
    # xarray data loader requires time to be a cftime.datetime object
    if name == "time":
        return [
            cftime.DatetimeProlepticGregorian(2000, 1, 1) + i * timestep
            for i in range(size)
        ]
    else:
        return np.arange(size, dtype=np.float32)


@dataclasses.dataclass
class DimSizes:
    n_time: int
    horizontal: list[DimSize]
    nz_interface: int

    @property
    def shape_nd(self) -> list[int]:
        return [self.n_time] + [dim.size for dim in self.horizontal]

    @property
    def dims_nd(self) -> list[str]:
        return ["time"] + [dim.name for dim in self.horizontal]

    @property
    def shape_vertical_interface(self) -> tuple[int]:
        return (self.nz_interface,)

    @property
    def items(self):
        return [("time", self.n_time)] + [
            (dim.name, dim.size) for dim in self.horizontal
        ]

    def get_size(self, key: str) -> int:
        for item in self.horizontal:
            if item.name == key:
                return item.size
        raise KeyError(f"Dimension with name '{key}' not found.")


def save_nd_netcdf(
    filename,
    dim_sizes: DimSizes,
    variable_names: list[str],
    timestep_days: float,
    time_varying_values: list[float] | None = None,
    save_vertical_coordinate: bool = True,
    return_ds: bool = False,
) -> xr.Dataset | None:
    """
    Save a ND netcdf file with random data for the given variable names and
    dimensions.

    Args:
        filename: The filename to save the netcdf file to.
        dim_sizes: The dimensions of the data.
        variable_names: The names of the variables to save.
        timestep_days: The number of days between each time step.
        time_varying_values: If not None, the values to use for each time step.
        save_vertical_coordinate: If True, save vertical coordinate variables.
        return_ds: If True, return the dataset in addition to saving it.
    """
    ds = get_nd_dataset(
        dim_sizes=dim_sizes,
        variable_names=variable_names,
        timestep_days=timestep_days,
        include_vertical_coordinate=save_vertical_coordinate,
    )
    if time_varying_values is not None:
        for name in variable_names:
            for i in range(dim_sizes.n_time):
                ds[name].isel(time=i).values[:] = time_varying_values[i]
    ds.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")
    if return_ds:
        return ds
    return None


def save_scalar_netcdf(
    filename,
    variable_names: list[str],
):
    ds = get_scalar_dataset(variable_names)
    ds.to_netcdf(filename, format="NETCDF4_CLASSIC")


@dataclasses.dataclass
class StatsData:
    path: pathlib.Path
    names: list[str]

    def __post_init__(self):
        save_scalar_netcdf(
            self.mean_filename,
            variable_names=self.names,
        )
        save_scalar_netcdf(
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
    names: list[str]
    dim_sizes: DimSizes
    timestep_days: float
    time_varying_values: list[float] | None = None
    num_data_workers: int = 0
    save_vertical_coordinate: bool = True

    def __post_init__(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        if (
            self.time_varying_values is not None
            and len(self.time_varying_values) != self.dim_sizes.n_time
        ):
            raise ValueError(
                f"Number of time-varying values ({len(self.time_varying_values)}) "
                f"must match number of time steps ({self.dim_sizes.n_time})"
            )
        self._ds: xr.Dataset = save_nd_netcdf(
            self.data_filename,
            dim_sizes=self.dim_sizes,
            variable_names=self.names,
            timestep_days=self.timestep_days,
            time_varying_values=self.time_varying_values,
            save_vertical_coordinate=self.save_vertical_coordinate,
            return_ds=True,
        )

    @property
    def data_path(self):
        # data must be in a separate path as a default loader loads all *.nc files
        return self.path / "data"

    @property
    def data_filename(self):
        return self.data_path / "data.nc"

    @property
    def horizontal_coords(self) -> dict[str, xr.DataArray]:
        return {dim.name: self._ds[dim.name] for dim in self.dim_sizes.horizontal}

    @property
    def inference_data_loader_config(self) -> InferenceDataLoaderConfig:
        return InferenceDataLoaderConfig(
            dataset=XarrayDataConfig(
                str(self.data_path),
            ),
            start_indices=InferenceInitialConditionIndices(
                first=0, n_initial_conditions=1, interval=1
            ),
            num_data_workers=self.num_data_workers,
        )


@dataclasses.dataclass
class MonthlyReferenceData:
    path: pathlib.Path
    names: list[str]
    dim_sizes: DimSizes
    n_ensemble: int

    def __post_init__(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        ds = get_nd_dataset(
            dim_sizes=self.dim_sizes,
            variable_names=self.names,
        )
        # drop default time coord from get_nd_dataset
        ds = ds.drop_vars(["time"])
        # add a time axis for months
        months = []
        for i in range(self.dim_sizes.n_time):
            year = 2000 + i // 12
            month = i % 12 + 1
            months.append(cftime.DatetimeProlepticGregorian(year, month, 15))
        ds["counts"] = xr.DataArray(
            np.random.randint(1, 10, size=(self.dim_sizes.n_time,)),
            dims=["time"],
            attrs={"units": "m", "long_name": "counts", "coordinates": "valid_time"},
        )
        member_datasets = []
        months_list = []
        for _ in range(self.n_ensemble):
            member_datasets.append(ds)
            months_list.append(xr.DataArray(months, dims=["time"]))
        ds = xr.concat(member_datasets, dim="sample")
        ds.coords["valid_time"] = xr.concat(months_list, dim="sample")
        ds.to_netcdf(self.data_filename, format="NETCDF4_CLASSIC")
        self.start_time = cftime.DatetimeProlepticGregorian(2000, 1, 1)

    @property
    def data_path(self):
        # data must be in a separate path as a default loader loads all *.nc files
        return self.path / "monthly_data"

    @property
    def data_filename(self):
        return self.data_path / "monthly.nc"


def get_nd_overrides(dim_sizes: DimSizes, timestep_days: float) -> dict[str, Any]:
    coords_override: dict[str, Any]
    time = [
        cftime.DatetimeProlepticGregorian(2000, 1, 1)
        + i * timestep_days * datetime.timedelta(days=1)
        for i in range(dim_sizes.n_time)
    ]
    coords_override = {"time": time}
    if "grid_yt" in dim_sizes.dims_nd:
        n_lat = dim_sizes.get_size("grid_yt")  # n_lat
        n_lon = dim_sizes.get_size("grid_xt")  # n_lon
        grid_yt = np.linspace(-89.5, 89.5, n_lat)
        grid_xt_start = 360.0 / n_lon / 2
        grid_xt = np.linspace(grid_xt_start, 360.0 - grid_xt_start, n_lon)
        coords_override["grid_yt"] = grid_yt
        coords_override["grid_xt"] = grid_xt
    return coords_override


def get_nd_dataset(
    dim_sizes: DimSizes,
    variable_names: Sequence[str],
    timestep_days: float = 1.0,
    include_vertical_coordinate: bool = True,
):
    """
    Gets a dataset of [time, <horizontal dims>] data.
    """
    data_vars = {}
    for name in variable_names:
        data = np.random.randn(*dim_sizes.shape_nd).astype(np.float32)
        data_vars[name] = xr.DataArray(
            data,
            dims=dim_sizes.dims_nd,
            attrs={"units": "m", "long_name": name},
        )
    coords_override = get_nd_overrides(dim_sizes=dim_sizes, timestep_days=timestep_days)

    coords = {
        dim_name: (
            xr.DataArray(
                np.arange(size, dtype=np.float64),
                dims=(dim_name,),
            )
            if dim_name not in coords_override
            else coords_override[dim_name]
        )
        for dim_name, size in dim_sizes.items
    }

    if include_vertical_coordinate:
        for i in range(dim_sizes.nz_interface):
            data_vars[f"ak_{i}"] = np.float64(i)
            data_vars[f"bk_{i}"] = np.float64(i + 1)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds


def get_scalar_dataset(
    variable_names: Iterable[str],
    fill_value: float | None = None,
):
    data_vars = {}
    for name in variable_names:
        if fill_value is not None:
            value = fill_value
        else:
            value = np.random.randn()
        data_vars[name] = xr.DataArray(value, attrs={"units": "m", "long_name": name})

    ds = xr.Dataset(data_vars=data_vars)
    return ds
