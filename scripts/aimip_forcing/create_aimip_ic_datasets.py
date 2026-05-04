import logging
import os
from typing import Tuple

import click
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

# this script is based on the notebook at
# https://github.com/ai2cm/explore2/blob/main/brianh/2025-06-18-ERA5-SHiELD-AMIP-forcing/2025-10-16-make-AIMIP-evaluation-IC-datasets-v3.ipynb

ERA5_GCS_DATA = "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
TARGET_TIMESTAMP = "1978-09-30T18:00:00"
IC_TIMESTAMPS = [
    "1978-09-29T00:00:00",
    "1978-09-30T00:00:00",
    "1978-10-01T00:00:00",
    "1978-10-02T00:00:00",
    "1978-10-03T00:00:00",
]
PROGNOSTIC_VARIABLES = (
    ["PRESsfc", "surface_temperature"]
    + [f"air_temperature_{i}" for i in range(8)]
    + [f"specific_total_water_{i}" for i in range(8)]
    + [f"eastward_wind_{i}" for i in range(8)]
    + [f"northward_wind_{i}" for i in range(8)]
)


def create_ic(
    era5: xr.Dataset,
    ic_timestamp: str,
    target_timestamp: np.datetime64,
) -> xr.Dataset:
    ic = era5.sel(time=ic_timestamp)
    return ic.assign_coords(time=target_timestamp)


@click.command()
@click.argument("local_output_dir", type=click.Path())
@click.option(
    "--era5-gcs-data",
    type=str,
    default=ERA5_GCS_DATA,
    help="Path to ERA5 1-degree 8-layer data in GCS.",
)
@click.option(
    "--target-timestamp",
    type=str,
    default=TARGET_TIMESTAMP,
    help="Timestamp to assign to all IC datasets.",
)
@click.option(
    "--ic-timestamp",
    "ic_timestamps",
    type=str,
    multiple=True,
    default=IC_TIMESTAMPS,
    help=(
        "Source timestamps for IC datasets. Provide once per IC member. "
        "Output files are named {target_date}_IC{i}.nc."
    ),
)
def main(
    local_output_dir: str,
    era5_gcs_data: str,
    target_timestamp: str,
    ic_timestamps: Tuple[str, ...],
):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(local_output_dir, exist_ok=True)

    logging.info(f"Opening ERA5 data from {era5_gcs_data}")
    era5 = xr.open_zarr(era5_gcs_data)[PROGNOSTIC_VARIABLES]

    target_dt = np.datetime64(target_timestamp)
    target_date = target_timestamp.split("T")[0]

    for i, ic_timestamp in enumerate(ic_timestamps):
        logging.info(f"Creating IC {i} from {ic_timestamp}")
        ic = create_ic(era5, ic_timestamp, target_dt)
        output_path = os.path.join(local_output_dir, f"{target_date}_IC{i}.nc")
        logging.info(f"Writing IC {i} to {output_path}")
        write_op = ic.to_netcdf(output_path, compute=False)
        with ProgressBar():
            write_op.compute()


if __name__ == "__main__":
    main()
