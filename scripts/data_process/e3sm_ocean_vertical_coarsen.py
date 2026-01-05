import dataclasses
import os
import time
import warnings
from typing import Mapping

import click
import dacite
import numpy as np
import xarray as xr
import yaml
from xgcm import Grid

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


@dataclasses.dataclass
class E3SMVerticalCoarseningConfig:
    """Configuration for coarsening E3SM MPAS ocean and ice raw data.

    Attributes:
        ocean_input_file_pattern: Path to the directory containing E3SM ocean input
            files including the pattern (e.g., "*.nc").
        ice_input_file_pattern: Path to the directory containing E3SM sea ice input
            files including the pattern (e.g., "*.nc").
        nc_grid_path: Path to the NetCDF file providing vertical coordinate information.
        reindex_time: If True, reindex the time axis for consistent temporal alignment.
        time_frequency: Time frequency string (e.g., "5D")
        time_dimension: Name of the time dimension in input datasets.
        spatial_dimension: Name of the main horizontal spatial dimension.
        vertical_layer_dimension: Name of the vertical layer dimension.
        vertical_interface_dimension: Name of the vertical interface dimension.
        resting_thickness_name: Name of the resting thickness variable.
        temperature_name: Name of the temperature variable.
        ocean_variable_names: List of variable names to select and coarsen.
        ice_variable_names: List of variable names to select and coarsen.
        add_sst: If True, adds sea surface temperature variable in Kelvin to the output.
        renaming: Optional mapping of variable or dimension names for renaming.
        ocean_vertical_target_layer_levels: List of target depth levels
            for vertical coarsening.
        ocean_vertical_target_interface_levels: List of target vertical interface
            depths for vertical coarsening.
        output_path: Directory path to write the coarsened netCDF outputs.
        output_prefix: Prefix for naming the coarsened output files.
    """

    ocean_input_file_pattern: str
    ice_input_file_pattern: str
    nc_grid_path: str
    reindex_time: bool = True
    time_frequency: str = "5D"
    time_dimension: str = "Time"
    spatial_dimension: str = "nCells"
    vertical_layer_dimension: str = "nVertLevels"
    vertical_interface_dimension: str = "nVertLevelsP1"
    resting_thickness_name: str = "restingThickness"
    temperature_name: str = "timeCustom_avg_activeTracers_temperature"
    ocean_variable_names: list = dataclasses.field(default_factory=list)
    ice_variable_names: list = dataclasses.field(default_factory=list)
    add_sst: bool = True
    renaming: Mapping[str, str] = dataclasses.field(default_factory=dict)
    ocean_vertical_target_layer_levels: list = dataclasses.field(default_factory=list)
    ocean_vertical_target_interface_levels: list = dataclasses.field(
        default_factory=list
    )
    output_path: str = "."
    output_prefix: str = "vertically_coarsened"

    @classmethod
    def from_file(cls, path: str) -> "E3SMVerticalCoarseningConfig":
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )

    def rename(self, ds: xr.Dataset):
        if len(self.renaming) == 0:
            return ds
        return ds.rename(self.renaming)


@click.command()
@click.option("--config", help="Config file")
@click.option("--year", help="Which year to process")
def main(config, year):
    config = E3SMVerticalCoarseningConfig.from_file(config)
    year = int(year)
    segment_time = time.time()
    current_year = f"{year:04d}"
    grid = xr.open_dataset(config.nc_grid_path)
    filenames_ocn = f"{config.ocean_input_file_pattern}*{current_year}*.nc"
    filenames_ice = f"{config.ice_input_file_pattern}*{current_year}*.nc"
    ice = xr.open_mfdataset(
        filenames_ice, combine="nested", concat_dim=config.time_dimension
    )
    ice = ice[config.ice_variable_names]
    ds = xr.open_mfdataset(filenames_ocn)[config.ocean_variable_names]
    ds = ds.assign_coords(
        {
            f"{config.vertical_layer_dimension}": grid[config.vertical_layer_dimension],
            f"{config.vertical_interface_dimension}": grid[
                config.vertical_interface_dimension
            ],
        }
    )
    ds[f"{config.resting_thickness_name}"] = grid[f"{config.resting_thickness_name}"]
    if config.add_sst:
        ds["sst"] = (
            ds[config.temperature_name].isel({config.vertical_layer_dimension: 0})
            + 273.15
        )
        ds["sst"].attrs["long_name"] = "sea surface temperature"
        ds["sst"].attrs["units"] = "K"
    vertical_vars = []
    for var in config.ocean_variable_names:
        if config.vertical_layer_dimension in ds[var].dims:
            vertical_vars.append(var)
    vars_2d = [
        var
        for var in ds.data_vars
        if config.vertical_layer_dimension not in ds[var].dims
        and config.spatial_dimension in ds[var].dims
    ]
    vars_2d = ds[vars_2d]
    grid = Grid(
        ds,
        coords={
            "Z": {
                "center": config.vertical_layer_dimension,
                "outer": config.vertical_interface_dimension,
            }
        },
        boundary="fill",
        periodic=False,
        fill_value=0.0,
    )
    vertical_target_interface_levels = np.array(
        config.ocean_vertical_target_layer_levels
    )
    ds_coarsened = xr.Dataset()
    coarsened_thickness = grid.transform(
        ds[config.resting_thickness_name],
        "Z",
        vertical_target_interface_levels,
        method="conservative",
    )
    ds_coarsened[config.resting_thickness_name] = coarsened_thickness
    for var in vertical_vars:
        current = (ds[var] * ds[config.resting_thickness_name]).fillna(0)
        current_coarsened = grid.transform(
            current,
            "Z",
            vertical_target_interface_levels,
            method="conservative",
        )
        ds_coarsened[var] = current_coarsened / coarsened_thickness
    if config.reindex_time:
        tstart = ds.xtime.values[0].decode("utf-8").replace("_", " ")
        tend = ds.xtime.values[-1].decode("utf-8").replace("_", " ")
        timestamps = xr.date_range(
            start=tstart,
            end=tend,
            freq=config.time_frequency,
            use_cftime=True,
            calendar="noleap",
        )
        ds = ds.assign_coords({config.time_dimension: timestamps})
    ds = xr.merge([ds_coarsened, vars_2d, ice])
    ds = ds.chunk({config.time_dimension: 1})
    ds = config.rename(ds)

    print(f"Processing {current_year}, output dataset size is {ds.nbytes / 1e9} GB")
    os.makedirs(config.output_path, exist_ok=True)
    ds.to_netcdf(
        f"{config.output_path}/{config.output_prefix}.{current_year}.nc",
        compute=True,
    )
    print(
        f"Time taken to process year {current_year}: "
        f"{time.time() - segment_time:0.2f} seconds"
    )


if __name__ == "__main__":
    main()
