# type: ignore
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

FORCING_NAMES = [
    "DSWRFtoa",
    "HGTsfc",
    "land_fraction",
    "ocean_fraction",
    "sea_ice_fraction",
    "surface_temperature",
    "global_mean_co2",
] + [f"{name}_{i}" for i in range(9) for name in ["ak", "bk"]]

PROGNOSTIC_NAMES = [
    "PRESsfc",
    "surface_temperature",
    "TMP2m",
    "Q2m",
    "UGRD10m",
    "VGRD10m",
] + [
    f"{name}_{i}"
    for i in range(8)
    for name in [
        "air_temperature",
        "specific_total_water",
        "eastward_wind",
        "northward_wind",
    ]
]
DEFAULT_INPUT_URL = (
    "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
)
OUTPUT_PATH = "/Users/oliverwm/scratch/sample_era5_data"
DEFAULT_FORCING_START_TIME = "1940"
DEFAULT_FORCING_END_TIME = "2022"

IC_YEARS = [1940, 1950, 1979, 2001, 2020]


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-url", type=str, default=DEFAULT_INPUT_URL)
    parser.add_argument(
        "--forcing-start-time", type=str, default=DEFAULT_FORCING_START_TIME
    )
    parser.add_argument(
        "--forcing-end-time", type=str, default=DEFAULT_FORCING_END_TIME
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information and don't write data.",
    )
    parser.add_argument(
        "--compress-forcing",
        action="store_true",
        help="Use some compression for forcing data.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Local path (directory) to write output. Must not exist yet.",
    )
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    os.makedirs(args.output)

    ds = xr.open_zarr(args.input_url)
    ds_forcing = ds[FORCING_NAMES].sel(
        time=slice(args.forcing_start_time, args.forcing_end_time)
    )
    for name in ds_forcing:
        # the CO2 and ak/bk coordinates are saved in double precision in source dataset
        ds_forcing[name] = ds_forcing[name].astype("float32")

    ds_ic = defaultdict(list)
    for year in IC_YEARS:
        for month in range(1, 13):
            if year == 1940 and month == 1:
                # ERA5 dataset does not have data at 0Z on 1940-01-01
                start_hour = 12
            else:
                start_hour = 0

            time = f"{year}{month:02d}01T{start_hour:02d}:00:00"
            current_snapshot = ds[PROGNOSTIC_NAMES].sel(time=time)
            ds_ic[year].append(current_snapshot)
        ds_ic[year] = xr.concat(ds_ic[year], dim="time")
        ds_ic[year] = ds_ic[year].chunk({"time": 1, "latitude": 180, "longitude": 360})
        for name in ds_ic[year]:
            del ds_ic[year][name].encoding["chunks"]

    ds_forcing = ds_forcing.chunk({"time": 20, "latitude": 180, "longitude": 360})

    if args.debug:
        print("Writing data to", args.output)
        print("Initial conditions:")
        for year, ic in ds_ic.items():
            print(year)
            print(ic)

        print("Forcing:")
        print(ds_forcing)
    else:
        output_dir = os.path.join(args.output, "initial_conditions")
        os.makedirs(output_dir)
        for year, ic in ds_ic.items():
            output_path = os.path.join(output_dir, f"ic_{year}.nc")
            print("Writing ", output_path)
            with ProgressBar():
                ic.to_netcdf(output_path)

        if args.compress_forcing:
            encoding = {name: {"zlib": True} for name in ds_forcing}
        else:
            encoding = {}
        output_dir = os.path.join(args.output, "forcing_data")
        os.makedirs(output_dir)
        yearly_forcing_ds = ds_forcing.resample(time="YS")
        for label, data in yearly_forcing_ds:
            if isinstance(label, np.datetime64):
                # np.datetime64 times do not have a strftime method,
                # so convert to pd.Timestamp
                label = pd.Timestamp(label)
            print(f"Processing forcing data for {label}")
            filename = os.path.join(
                output_dir, "forcing_" + label.strftime("%Y") + ".nc"
            )
            with ProgressBar():
                data.to_netcdf(filename, unlimited_dims=["time"], encoding=encoding)
