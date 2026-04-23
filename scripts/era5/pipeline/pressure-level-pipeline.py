"""Pipeline to extract ERA5 q, T, u, v on the 13 standard pressure levels.

These are the pressure levels freely provided by ECMWF in the IFS analysis,
intended for use as initial conditions for the ACE2-ERA5 model after vertical
interpolation from pressure levels to the ACE2 hybrid sigma-pressure grid.

Source: ARCO-ERA5 full_37 pressure-level store on GCS.
"""

import argparse
import datetime
import logging

import apache_beam as beam
import numpy as np
import pandas as pd
import xarray as xr
import xarray_beam as xbeam
import xesmf as xe
from apache_beam.options.pipeline_options import PipelineOptions
from obstore.store import from_url
from zarr.storage import ObjectStore

# these are the pressure levels freely provided by ECMWF for IFS forecasts and analyses
# e.g. see https://www.ecmwf.int/en/forecasts/datasets/open-data
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

TIME_STEP = 6  # hours between output timesteps
DEFAULT_OUTPUT_GRID = "F90"

# Variables to extract
PRESSURE_LEVEL_VARS = [
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

RENAME = {
    "specific_humidity": "Q",
    "temperature": "TMP",
    "u_component_of_wind": "UGRD",
    "v_component_of_wind": "VGRD",
}

URL_FULL_37 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Gaussian grid specs: name -> N (grid number; nlat=2N, nlon=4N)
GAUSSIAN_GRID_N = {
    "F22.5": 22.5,
    "F90": 90,
    "F360": 360,
}


# ---------------------------------------------------------------------------
# Regridding utilities (shared with xr-beam-pipeline.py)
# ---------------------------------------------------------------------------


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[lo], midpoints, [hi]])


def _gaussian_latitudes(n: int | float) -> np.ndarray:
    from numpy.polynomial.legendre import leggauss

    nlat = round(2 * n)
    x, _ = leggauss(nlat)
    lat = np.degrees(np.arcsin(x))
    return np.sort(lat)


def _make_target_grid(output_grid: str) -> xr.Dataset:
    n = GAUSSIAN_GRID_N[output_grid]
    lat = _gaussian_latitudes(n)
    nlon = round(4 * n)
    dlon = 360.0 / nlon
    lon = np.linspace(dlon / 2, 360 - dlon / 2, nlon)
    lat_b = _cell_bounds(lat, -90, 90)
    lon_b = _cell_bounds(lon, 0, 360)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], lat_b),
            "lon_b": (["lon_b"], lon_b),
        }
    )


def _make_source_grid() -> xr.Dataset:
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 359.75, 1440)
    lat_b = _cell_bounds(lat, -90, 90)
    lon_b = _cell_bounds(lon, -0.125, 360 - 0.125)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], lat_b),
            "lon_b": (["lon_b"], lon_b),
        }
    )


_REGRIDDER_CACHE = {}


def _get_regridder(output_grid: str):
    if output_grid not in _REGRIDDER_CACHE:
        src = _make_source_grid()
        dst = _make_target_grid(output_grid)
        _REGRIDDER_CACHE[output_grid] = xe.Regridder(
            src, dst, "conservative", periodic=True
        )
    return _REGRIDDER_CACHE[output_grid]


def _regrid(ds: xr.Dataset, output_grid: str) -> xr.Dataset:
    regridder = _get_regridder(output_grid)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sortby("lat")
    regridded = regridder(ds, keep_attrs=True)
    regridded = regridded.rename({"lat": "latitude", "lon": "longitude"})
    return regridded


# ---------------------------------------------------------------------------
# Data opening
# ---------------------------------------------------------------------------


def _make_zarr_store(url: str, read_only: bool = True):
    if url.startswith("gs://"):
        return ObjectStore(from_url(url), read_only=read_only)
    else:
        return url


def open_full_37(variables, time_slice) -> xr.Dataset:
    ds = xr.open_zarr(_make_zarr_store(URL_FULL_37), chunks=None)
    ds = ds[variables]
    ds_start = pd.Timestamp(ds.time.min().values)
    ds_stop = pd.Timestamp(ds.time.max().values)
    assert time_slice.start >= ds_start, "Start time out of bounds"
    assert time_slice.stop <= ds_stop, "End time out of bounds"
    ds = ds.sel(time=time_slice)
    return ds


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def _process_pressure_levels(ds: xr.Dataset, output_grid: str) -> xr.Dataset:
    """Select the 13 standard pressure levels, flatten to 2D fields, and regrid."""
    output = xr.Dataset()
    for name in ds.data_vars:
        if "level" not in ds[name].dims:
            continue
        short = RENAME[name]
        for pressure in PRESSURE_LEVELS:
            out_name = f"{short}{pressure}"
            da = ds[name].sel(level=pressure)
            da.attrs["long_name"] = (
                ds[name].attrs.get("long_name", name) + f" at {pressure} hPa"
            )
            output[out_name] = da

    output = output.drop_vars("level", errors="ignore")
    regridded = _regrid(output, output_grid)
    regridded = regridded.drop_vars(["latitude", "longitude"])
    return regridded


def process_chunk(key, ds, output_grid=DEFAULT_OUTPUT_GRID):
    output = _process_pressure_levels(ds, output_grid)
    new_key = key.replace(
        offsets={"time": key.offsets["time"], "latitude": 0, "longitude": 0},
        vars=frozenset(output.keys()),
    )
    return new_key, output


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def _make_template(ds_pressure_level, output_grid, output_time):
    logging.info("Building template from first timestep")
    ds_one = ds_pressure_level.isel(time=0).load()
    ds_regridded = _process_pressure_levels(ds_one, output_grid)
    ds_regridded = ds_regridded.drop_encoding()
    template = xbeam.make_template(ds_regridded.drop_vars("time", errors="ignore"))
    template = template.expand_dims(dim={"time": output_time}, axis=0)
    return template


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def _get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract ERA5 q, T, u, v on the 13 standard pressure levels "
            "and regrid to a Gaussian grid."
        )
    )
    parser.add_argument(
        "output_path", type=str, help="Output path for the processed zarr dataset"
    )
    parser.add_argument(
        "start_time", type=str, help="Start time (e.g. 2020-01-01T00:00:00)"
    )
    parser.add_argument(
        "end_time", type=str, help="End time (e.g. 2020-12-31T18:00:00)"
    )
    parser.add_argument(
        "--output_grid",
        type=str,
        default=DEFAULT_OUTPUT_GRID,
        help="Output grid specification (default: F90).",
    )
    parser.add_argument(
        "--output_time_chunksize",
        type=int,
        default=1,
        help="Number of times per output chunk.",
    )
    parser.add_argument(
        "--output_time_shardsize",
        type=int,
        default=120,
        help="Number of times per output shard.",
    )
    parser.add_argument(
        "--process_time_chunksize",
        type=int,
        default=6,
        help="Time chunk size for intermediate processing.",
    )
    return parser


def main():
    parser = _get_parser()
    args, pipeline_args = parser.parse_known_args()
    print(pipeline_args)

    start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.datetime.strptime(args.end_time, "%Y-%m-%dT%H:%M:%S")

    assert start_time.hour % 6 == 0, "start_time hour must be a multiple of 6"
    assert end_time.hour % 6 == 0, "end_time hour must be a multiple of 6"
    assert args.output_time_shardsize % args.process_time_chunksize == 0
    assert args.output_time_shardsize % args.output_time_chunksize == 0

    output_time_slice = slice(start_time, end_time, TIME_STEP)
    output_time = pd.date_range(start_time, end_time, freq=f"{TIME_STEP}h")

    output_chunks = {"time": args.output_time_chunksize}
    output_shards = {"time": args.output_time_shardsize}
    process_chunks = {"time": args.process_time_chunksize}

    logging.info("Opening pressure-level dataset")
    ds_pressure_level = open_full_37(PRESSURE_LEVEL_VARS, output_time_slice)

    logging.info("Generating template")
    template = _make_template(ds_pressure_level, args.output_grid, output_time)

    logging.info("Starting pipeline")
    output_store = _make_zarr_store(args.output_path, read_only=False)
    print(PipelineOptions(pipeline_args).get_all_options())

    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        (
            p
            | xbeam.DatasetToChunks(ds_pressure_level, chunks=process_chunks)
            | beam.MapTuple(process_chunk, output_grid=args.output_grid)
            | xbeam.ConsolidateChunks(output_shards)
            | xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    main()
