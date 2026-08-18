import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
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

MODEL_LEVEL_STORE = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
FREQUENCY = "6h"

RAW_VARIABLE_SOURCES = {
    PRESSURE_AND_SURFACE_LEVEL_STORE: [
        "surface_pressure",
        "land_sea_mask",
        "sea_ice_cover",
        "sea_surface_temperature",
        "geopotential_at_surface",
    ],
    MODEL_LEVEL_STORE: [
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
    ],
}
RAW_VARIABLES = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential_at_surface",
    "surface_pressure",
]
VARIABLE_RENAME = {
    "temperature": "T",
    "specific_humidity": "Q",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "geopotential_at_surface": "PHIS",
    "surface_pressure": "PS",
}
RAW_HYBRID_COORDINATE_ATTRIBUTE = "GRIB_pv"
RAW_HYBRID_LEVEL_DIMENSION = "hybrid"

HYBRID_LEVEL_INTERFACE_DIMENSION = "ilev"
HYBRID_LEVEL_DIMENSION = "lev"

DIMENSION_RENAME = {
    RAW_HYBRID_LEVEL_DIMENSION: HYBRID_LEVEL_DIMENSION,
    RAW_LATITUDE_DIMENSION: LATITUDE_DIMENSION,
    RAW_LONGITUDE_DIMENSION: LONGITUDE_DIMENSION,
}

HISTORY_ATTRIBUTE = f"""\
This file contains ERA5 data derived from the ARCO ERA5 dataset maintained by
Google, processed into a form compatible for nudging GFDL's SHiELD model. The
schema is modeled after that used by Larry Horowitz of GFDL when preparing
GFS analysis data for nudging SHiELD.

Raw source datasets and associated variables
--------------------------------------------

- {PRESSURE_AND_SURFACE_LEVEL_STORE}:
    - {", ".join(RAW_VARIABLE_SOURCES[PRESSURE_AND_SURFACE_LEVEL_STORE])}
- {MODEL_LEVEL_STORE}:
    - {", ".join(RAW_VARIABLE_SOURCES[MODEL_LEVEL_STORE])}

For consistency with the format of GFS analysis, the latitude is sorted to be
in ascending order (rather than descending in the case of ERA5), and the
hybrid coordinate is defined such that pressure in a column at a level k is
given by:

    p(k) = P0 * a(k) + ps * b(k),

where P0 is equal to 100000.0 Pa, a(k) and b(k) define the hybrid coordinate,
and ps is the surface pressure for the column.

{CITATIONS}"""
FILENAME_PATTERN = "%Y%m%d_%HZ.nc"
P0 = 100000.0  # Scale hyai by 1 / P0 for consistency with GFS analysis definition.


def get_raw_data(timestamp: pd.Timestamp) -> xr.Dataset:
    datasets = []
    for store, variables in RAW_VARIABLE_SOURCES.items():
        ds = xr.open_dataset(store, engine="zarr", chunks=None)
        ds = ds[variables]
        ds = ds.sel(time=[timestamp])
        datasets.append(ds)
    ds = xr.merge(datasets)
    return ds.sortby(RAW_LATITUDE_DIMENSION)  # Ensure latitude is ascending.


def get_hybrid_coefficients(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    pv = None
    for da in ds.values():
        if RAW_HYBRID_COORDINATE_ATTRIBUTE in da.attrs:
            pv = da.attrs[RAW_HYBRID_COORDINATE_ATTRIBUTE]
            break
    if pv is None:
        raise ValueError(
            f"No hybrid coordinate attribute found among data variables in "
            f"dataset: {ds.keys()}."
        )
    n_hybrid_interfaces = ds.sizes[RAW_HYBRID_LEVEL_DIMENSION] + 1
    hyai = np.array(pv[:n_hybrid_interfaces])
    hybi = np.array(pv[n_hybrid_interfaces:])
    hyai = xr.DataArray(hyai, dims=[HYBRID_LEVEL_INTERFACE_DIMENSION])
    hybi = xr.DataArray(hybi, dims=[HYBRID_LEVEL_INTERFACE_DIMENSION])
    hyai = hyai / P0  # Adjust for consistency with GFS analysis.
    return hyai, hybi


def get_surface_type_mask(ds: xr.Dataset) -> xr.DataArray:
    land_fraction = ds.land_sea_mask
    sea_ice_cover = ds.sea_ice_cover

    sea_land_mask = (land_fraction > 0.5).astype(np.int32)
    sea_ice_mask = (sea_ice_cover > 0.15) & (sea_land_mask == 0)
    return xr.where(sea_ice_mask, 2, sea_land_mask)


def get_derived_variables(ds: xr.Dataset) -> xr.Dataset:
    ds = fill_missing_sst_and_sea_ice_values(ds)
    hyai, hybi = get_hybrid_coefficients(ds)
    ORO = get_surface_type_mask(ds)
    TS = ds.sea_surface_temperature
    return xr.Dataset({"hyai": hyai, "hybi": hybi, "ORO": ORO, "TS": TS})


def get_nudging_data(timestamp: pd.Timestamp) -> xr.Dataset:
    ds = get_raw_data(timestamp)
    derived_variables = get_derived_variables(ds)
    raw_variables = ds[RAW_VARIABLES]
    nudging_data = xr.merge([raw_variables, derived_variables])
    nudging_data = nudging_data.rename(VARIABLE_RENAME)
    nudging_data = nudging_data.rename(DIMENSION_RENAME)
    nudging_data = nudging_data.convert_calendar(SHIELD_CALENDAR, TIME_DIMENSION)
    nudging_data = nudging_data.assign_attrs(history=HISTORY_ATTRIBUTE)
    return nudging_data


def _configure_worker_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def process_timestamp(timestamp: pd.Timestamp, destination_dir: str) -> str:
    logger.info("[pid %s] Processing timestamp %s.", os.getpid(), timestamp)
    nudging_data = get_nudging_data(timestamp)
    filename = timestamp.strftime(FILENAME_PATTERN)
    destination = os.path.join(destination_dir, filename)
    to_remote_netcdf(nudging_data, destination)
    logger.info("[pid %s] Wrote %s.", os.getpid(), destination)
    return destination


def main_nudging(start_datetime, end_datetime, destination, n_workers):
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)
    timestamps = xr.date_range(start_datetime, end_datetime, freq=FREQUENCY)
    n_timestamps = len(timestamps)

    logger.info(
        "Processing %s timestamps from %s to %s using %s worker(s).",
        n_timestamps,
        start_datetime,
        end_datetime,
        n_workers,
    )

    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_configure_worker_logging
    ) as executor:
        futures = [
            executor.submit(process_timestamp, timestamp, destination)
            for timestamp in timestamps
        ]
        for n, future in enumerate(as_completed(futures), start=1):
            future.result()
            logger.info("Completed %s/%s timestamps.", n, n_timestamps)


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
            "Directory or bucket path to write one netCDF file per timestamp "
            "to, named according to the pattern "
            f"{FILENAME_PATTERN.replace('%', '%%')!r}."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        required=True,
        help=(
            "Number of worker processes to use to process timestamps in "
            "parallel. Size this to the container's memory limit, not its "
            "CPU count -- each worker eagerly loads several GB of model-"
            "level data and buffers a full netCDF file in memory."
        ),
    )
    args = parser.parse_args()
    main_nudging(
        args.start_datetime,
        args.end_datetime,
        args.destination,
        args.n_workers,
    )
    logger.info("Done.")
