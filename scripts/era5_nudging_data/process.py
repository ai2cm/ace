import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
logger = logging.getLogger(__name__)

PRESSURE_AND_SURFACE_LEVEL_STORE = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
MODEL_LEVEL_STORE = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
TIME_DIMENSION = "time"
FREQUENCY = "6h"
SHIELD_CALENDAR = "julian"

RAW_VARIABLE_SOURCES = {
    PRESSURE_AND_SURFACE_LEVEL_STORE: [
        "surface_pressure",
        "land_sea_mask",
        "sea_ice_cover",
        "sea_surface_temperature",
        "skin_temperature",
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
RAW_LATITUDE_DIMENSION = "latitude"
RAW_LONGITUDE_DIMENSION = "longitude"

HYBRID_LEVEL_INTERFACE_DIMENSION = "ilev"
HYBRID_LEVEL_DIMENSION = "lev"
LATITUDE_DIMENSION = "lat"
LONGITUDE_DIMENSION = "lon"

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

Citation for ARCO ERA5 dataset
------------------------------

Carver, Robert W, and Merose, Alex. (2023):
    ARCO-ERA5: An Analysis-Ready Cloud-Optimized Reanalysis Dataset.
    22nd Conf. on AI for Env. Science, Denver, CO, Amer. Meteo. Soc, 4A.1,
    https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842

Citation for ERA5 dataset
-------------------------

Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A.,
    Muñoz‐Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D.,
    Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G.,
    Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., De Chiara, G.,
    Dahlgren, P., Dee, D., Diamantakis, M., Dragani, R., Flemming, J.,
    Forbes, R., Fuentes, M., Geer, A., Haimberger, L., Healy, S.,
    Hogan, R.J., Hólm, E., Janisková, M., Keeley, S., Laloyaux, P.,
    Lopez, P., Lupu, C., Radnoti, G., de Rosnay, P., Rozum, I., Vamborg, F.,
    Guillaume, S., Thépaut, J-N. (2017): Complete ERA5: Fifth generation of
    ECMWF atmospheric reanalyses of the global climate. Copernicus Climate
    Change Service (C3S) Data Store (CDS). (Accessed on DD-MM-YYYY)
"""
FILENAME_PATTERN = "%Y%m%d_%HZ.nc"


def get_raw_data(timestamp: pd.Timestamp) -> xr.Dataset:
    datasets = []
    for store, variables in RAW_VARIABLE_SOURCES.items():
        ds = xr.open_dataset(store, engine="zarr", chunks=None)
        ds = ds[variables]
        ds = ds.sel(time=[timestamp])
        datasets.append(ds)
    return xr.merge(datasets)


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
    return hyai, hybi


def get_surface_type_mask(ds: xr.Dataset) -> xr.DataArray:
    land_fraction = ds.land_sea_mask
    sea_ice_cover = ds.sea_ice_cover

    sea_land_mask = (land_fraction > 0.5).astype(np.int32)
    sea_ice_mask = (sea_ice_cover > 0.15) & (sea_land_mask == 0)
    return xr.where(sea_ice_mask, 2, sea_land_mask)


def get_surface_temperature(ds: xr.Dataset) -> xr.DataArray:
    sst = ds.sea_surface_temperature
    skin_temperature = ds.skin_temperature
    return xr.where(sst.notnull(), sst, skin_temperature)


def get_derived_variables(ds: xr.Dataset) -> xr.Dataset:
    hyai, hybi = get_hybrid_coefficients(ds)
    ORO = get_surface_type_mask(ds)
    TS = get_surface_temperature(ds)
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


def to_remote_netcdf(ds: xr.Dataset, destination: str) -> None:
    netcdf_in_memory = ds.to_netcdf(path=None, engine="h5netcdf")
    with fsspec.open(destination, "wb") as file:
        file.write(netcdf_in_memory)


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

    start_datetime = pd.to_datetime(args.start_datetime)
    end_datetime = pd.to_datetime(args.end_datetime)
    timestamps = xr.date_range(start_datetime, end_datetime, freq=FREQUENCY)
    n_timestamps = len(timestamps)
    logger.info(
        "Processing %s timestamps from %s to %s using %s worker(s).",
        n_timestamps,
        start_datetime,
        end_datetime,
        args.n_workers,
    )

    with ProcessPoolExecutor(
        max_workers=args.n_workers, initializer=_configure_worker_logging
    ) as executor:
        futures = [
            executor.submit(process_timestamp, timestamp, args.destination)
            for timestamp in timestamps
        ]
        for n, future in enumerate(as_completed(futures), start=1):
            future.result()
            logger.info("Completed %s/%s timestamps.", n, n_timestamps)

    logger.info("Done.")
