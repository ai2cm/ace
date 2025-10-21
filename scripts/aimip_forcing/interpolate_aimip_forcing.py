import logging
from typing import List

import click
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from encoding import clear_encoding, set_shards_chunks

# this script is based on the notebook at
# https://github.com/ai2cm/explore2/blob/main/brianh/2025-06-18-ERA5-SHiELD-AMIP-forcing/regrid/2025-08-26-ACE2-ERA5-forcing-AIMIP.ipynb

ACE2_ERA5_DATA = "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
SURFACE_TEMPERATURE_NAME = "surface_temperature"
MONTHLY_AIMIP_FORCING_VARIABLES = [
    "sea_ice_fraction",
    "land_fraction",
    "ocean_fraction",
    SURFACE_TEMPERATURE_NAME,
]
EXISTING_ERA5_FORCING_VARIABLES = [
    "HGTsfc",
    "DSWRFtoa",
]
START_TIME = "1978-10-01T00:00:00"
END_TIME = "2024-12-31T18:00:00"


def open_aimip_forcing_data(
    local_input_filepath: str,
    variables: List[str] = MONTHLY_AIMIP_FORCING_VARIABLES,
) -> xr.Dataset:
    """
    Open the AIMIP forcing data from a local netCDF file. Use a numeric time coordinate
    to ease interpolation to the desired time coordinate.
    """
    monthly_aimip_forcing = xr.open_dataset(
        local_input_filepath,
        chunks={},
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=False),
    )[variables]
    return monthly_aimip_forcing


def get_sst_mask(
    surface_temperature: xr.Dataset, time_dim: str = "time"
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Get the sea surface temperature mask and the masked sea surface temperature from the
    surface temperature field, which contains NaNs over land. Assumes that the surface
    temperature mask is the same for all time steps.

    Args:
        surface_temperature: Surface temperature data array with NaNs over land.
        time_dim: Name of the time dimension in the surface temperature data array.
    Returns:
        A tuple of the sea surface temperature mask and the masked sea surface
        temperature.
    """
    sst_mask = (~np.isnan(surface_temperature)).sum(dim=time_dim) > 0
    filled_surface_temeperature = surface_temperature.fillna(-999)
    return sst_mask, filled_surface_temeperature


def get_existing_era5_forcing(
    ace2_era5_gcs_data: str,
    start_time: str,
    end_time: str,
    variables: List[str] = EXISTING_ERA5_FORCING_VARIABLES,
    dtype=np.float32,
) -> xr.Dataset:
    """
    Get existing ERA5 forcing data from ACE2 ERA5 data in GCS.
    Use a numeric time coordinate to ease interpolation to the desired time coordinate.

    Args:
        ace2_era5_gcs_data: Path to ACE2 ERA5 data in GCS.
        start_time: Start time for the data to be retrieved.
        end_time: End time for the data to be retrieved.
        variables: List of variable names to be retrieved.
        dtype: Data type to which latitude and longitude coordinates should be cast.
    """

    existing_era5_forcing = xr.open_zarr(
        ace2_era5_gcs_data,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=False),
    )[variables]

    existing_era5_forcing = existing_era5_forcing.sel(time=slice(start_time, end_time))
    existing_era5_forcing = existing_era5_forcing.assign_coords(
        {
            "latitude": existing_era5_forcing.latitude.astype(dtype),
            "longitude": existing_era5_forcing.longitude.astype(dtype),
        }
    )
    return existing_era5_forcing


def get_time_coordinate(
    existing_time_coordinate: xr.DataArray,
    extension_start: str,
    extension_end: str,
) -> xr.DataArray:
    """
    Extends the existing time coordinate using a numeric coordinate
    (which eases interpolation of the monthly AIMIP forcing data to the
    desired time coordinate).
    """
    extended_time_coord = xr.DataArray(
        xr.date_range(
            start=extension_start,
            end=extension_end,
            freq="6h",
            use_cftime=False,
        ).values,
        dims=["time"],
        name="time",
    )

    time_coord = xr.concat(
        [
            existing_time_coordinate,
            extended_time_coord,
        ],
        dim="time",
    )
    return time_coord


def get_repeated_insolation(
    era5_forcing_DSWRFtoa: xr.DataArray,
    start_repeat: str,
    end_repeat: str,
    source_start: str,
    source_end: str,
):
    """
    Get repeated insolation data for the period beyond the end of the existing ERA5 data
    by the end of the existing data for the given day of the year.
    """
    repeated_era5_forcing_DSWRFtoa = era5_forcing_DSWRFtoa.sel(
        time=slice(source_start, source_end)
    ).drop_vars("time")
    new_time_coordinate = xr.date_range(
        start=start_repeat,
        end=end_repeat,
        freq="6h",
        use_cftime=False,
    )

    repeated_era5_forcing_DSWRFtoa = repeated_era5_forcing_DSWRFtoa.assign_coords(
        {"time": new_time_coordinate}
    )
    return repeated_era5_forcing_DSWRFtoa


def write_output_zarr(ds: xr.Dataset, output_data_file: str):
    """
    Write the output dataset to a zarr file, one variable at a time to avoid
    excessive memory usage that seems to be a problem with the dask graph.
    """
    initial = True
    for var in ds.data_vars:
        mode = "w" if initial else "a"
        zarr_write = ds[[var]].to_zarr(
            output_data_file, mode=mode, compute=False, consolidated=False
        )
        with ProgressBar():
            zarr_write.compute()
        initial = False


@click.command()
@click.argument(
    "input_data_file",
    type=click.Path(exists=True),
)
@click.argument(
    "output_data_file",
    type=click.Path(),
)
@click.option(
    "--start-time",
    type=str,
    default=START_TIME,
    help="Start time for the output data.",
)
@click.option(
    "--end-time",
    type=str,
    default=END_TIME,
    help="End time for the output data.",
)
@click.option(
    "--ace2-era5-gcs-data",
    type=str,
    default=ACE2_ERA5_DATA,
    help="Path to ACE2 ERA5 data in GCS.",
)
def main(
    input_data_file: str,
    output_data_file: str,
    ace2_era5_gcs_data: str,
    start_time: str,
    end_time: str,
):
    logging.basicConfig(level=logging.INFO)
    monthly_aimip_forcing = open_aimip_forcing_data(input_data_file)

    logging.info("Processing AIMIP sea surface temperature.")
    sst_mask, masked_sst = get_sst_mask(monthly_aimip_forcing[SURFACE_TEMPERATURE_NAME])
    monthly_aimip_forcing[SURFACE_TEMPERATURE_NAME] = masked_sst

    logging.info("Getting existing ERA5 forcing data.")
    existing_era5_forcing = get_existing_era5_forcing(
        ace2_era5_gcs_data,
        start_time,
        end_time,
    )

    time_coord = get_time_coordinate(
        existing_era5_forcing.time.drop_vars("time"),
        extension_start="2023-01-01T00:00:00",
        extension_end=end_time,
    )

    logging.info("Interpolating AIMIP forcing data to ACE2-ERA5 time coordinate.")
    interpolated_aimip_forcing = monthly_aimip_forcing.interp(time=time_coord)

    interpolated_aimip_forcing[SURFACE_TEMPERATURE_NAME] = interpolated_aimip_forcing[
        SURFACE_TEMPERATURE_NAME
    ].where(sst_mask)

    logging.info("Merging interpolated AIMIP forcing with existing ERA5 forcing.")
    repeated_era5_forcing_DSWRFtoa = get_repeated_insolation(
        existing_era5_forcing.DSWRFtoa,
        start_repeat="2023-01-01T00:00:00",
        end_repeat=end_time,
        source_start="2020-12-31T00:00:00",
        source_end="2022-12-31T18:00:00",
    )

    era5_forcing_DSWRFtoa = xr.concat(
        [
            existing_era5_forcing.DSWRFtoa,
            repeated_era5_forcing_DSWRFtoa,
        ],
        dim="time",
    )

    logging.info("Finalizing interpolated AIMIP forcing data.")
    interpolated_forcing = xr.merge(
        [
            interpolated_aimip_forcing,
            existing_era5_forcing.HGTsfc,
            era5_forcing_DSWRFtoa,
        ]
    )

    logging.info("Setting chunking and sharding for output AIMIP forcing data.")
    interpolated_forcing = clear_encoding(interpolated_forcing)
    interpolated_forcing_chunked = set_shards_chunks(interpolated_forcing)

    logging.info(f"Writing interpolated AIMIP forcing data to {output_data_file}")
    logging.info(interpolated_forcing_chunked.info())
    write_output_zarr(interpolated_forcing_chunked, output_data_file)


if __name__ == "__main__":
    main()
