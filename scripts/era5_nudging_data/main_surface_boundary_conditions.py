import argparse
import logging

import pandas as pd
import xarray as xr

from common import (
    CITATIONS,
    LATITUDE_DIMENSION,
    LOG_FORMAT,
    LONGITUDE_DIMENSION,
    PRESSURE_AND_SURFACE_LEVEL_STORE,
    RAW_LATITUDE_DIMENSION,
    RAW_LONGITUDE_DIMENSION,
    SHIELD_CALENDAR,
    TIME_DIMENSION,
    fill_missing_sst_and_sea_ice_values,
    logger,
    to_remote_netcdf,
)

# The raw store is hourly; step by 6 to match the 6-hourly nudging cadence.
STEP = 6
RAW_VARIABLE_SOURCES = {
    PRESSURE_AND_SURFACE_LEVEL_STORE: [
        "sea_ice_cover",
        "sea_surface_temperature",
    ],
}

HISTORY_ATTRIBUTE = f"""\
This file contains ERA5 data derived from the ARCO ERA5 dataset maintained by
Google, processed into surface boundary condition data (sea surface
temperature and sea ice fraction) compatible with SHiELD.

Raw source dataset and associated variables
--------------------------------------------

- {PRESSURE_AND_SURFACE_LEVEL_STORE}:
    - {", ".join(RAW_VARIABLE_SOURCES[PRESSURE_AND_SURFACE_LEVEL_STORE])}

Missing sea surface temperature values are filled via zonal-then-meridional
linear interpolation, and missing sea ice fraction values are filled with
zero. For consistency with the format expected by FMS-based models, the
latitude is sorted to be in ascending order (rather than descending in the
case of ERA5).

{CITATIONS}"""


def get_raw_surface_boundary_data(
    start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
) -> xr.Dataset:
    datasets = []
    for store, variables in RAW_VARIABLE_SOURCES.items():
        ds = xr.open_dataset(store, engine="zarr", chunks=None)
        ds = ds[variables]
        ds = ds.sel({TIME_DIMENSION: slice(start_datetime, end_datetime, STEP)})
        datasets.append(ds)
    ds = xr.merge(datasets)
    return ds.sortby(RAW_LATITUDE_DIMENSION)  # Ensure latitude is ascending.


def to_fms_format(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.rename(
        {
            RAW_LONGITUDE_DIMENSION: LONGITUDE_DIMENSION,
            RAW_LATITUDE_DIMENSION: LATITUDE_DIMENSION,
        }
    )
    ds = ds.rename({"sea_ice_cover": "sea_ice_fraction"})
    ds = ds.assign(lon=ds[LONGITUDE_DIMENSION].assign_attrs(axis="X"))
    ds = ds.assign(lat=ds[LATITUDE_DIMENSION].assign_attrs(axis="Y"))
    ds = ds.assign(time=ds[TIME_DIMENSION].assign_attrs(axis="T"))
    ds = ds.convert_calendar(SHIELD_CALENDAR, dim=TIME_DIMENSION)
    ds = ds.assign_attrs(history=HISTORY_ATTRIBUTE)
    return ds


def main_surface_boundary_conditions(start_datetime, end_datetime, destination):
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)

    logger.info(
        "Processing surface boundary conditions from %s to %s.",
        start_datetime,
        end_datetime,
    )

    ds = get_raw_surface_boundary_data(start_datetime, end_datetime)
    ds = ds.chunk({TIME_DIMENSION: 1})
    ds = fill_missing_sst_and_sea_ice_values(ds)
    ds = to_fms_format(ds)
    to_remote_netcdf(ds, destination, unlimited_dims=[TIME_DIMENSION])
    logger.info("Wrote %s.", destination)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-datetime",
        type=str,
        required=True,
        help="First timestamp to process, inclusive (e.g. 2020-01-01T00:00:00).",
    )
    parser.add_argument(
        "--end-datetime",
        type=str,
        required=True,
        help="Last timestamp to process, inclusive (e.g. 2020-01-02T00:00:00).",
    )
    parser.add_argument(
        "--destination",
        type=str,
        required=True,
        help=(
            "Path to write a single netCDF file containing surface boundary "
            "conditions for the full time range to."
        ),
    )
    args = parser.parse_args()
    main_surface_boundary_conditions(
        args.start_datetime,
        args.end_datetime,
        args.destination,
    )
    logger.info("Done.")
