import logging

import click
import numpy as np
import xarray as xr
import xesmf as xe
from dask.diagnostics import ProgressBar

# this script is based on the notebook at
# https://github.com/ai2cm/explore2/blob/main/brianh/2025-06-18-ERA5-SHiELD-AMIP-forcing/regrid/2025-08-20-regrid-AIMIP-forcing.ipynb

ACE2_ERA5_DATA = "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
LAND_THRESHOLD = 1.0  # minimum amount of land in order to exclude SSTs


def derive_ocean_fraction(land_fraction: xr.DataArray, sea_ice_fraction: xr.DataArray):
    ocean_fraction = 1.0 - land_fraction - sea_ice_fraction
    negative_ocean = ocean_fraction.where(ocean_fraction < 0, 0.0)
    sea_ice_fraction_out = sea_ice_fraction + negative_ocean
    ocean_fraction_out = ocean_fraction.where(ocean_fraction > 0, 0.0)
    return ocean_fraction_out, sea_ice_fraction_out


def get_regridder(
    input_latitude: xr.DataArray,
    input_longitude: xr.DataArray,
    output_latitude: xr.DataArray,
    output_longitude: xr.DataArray,
    method="conservative",
    dtype=np.float32,
) -> xe.Regridder:
    """
    Create a regridder from input grid to output grid.
    Assumes both grids are rectilinear.
    """
    input_grid = xr.Dataset(
        {
            "latitude": input_latitude.astype(dtype),
            "longitude": input_longitude.astype(dtype),
        }
    )
    output_grid = xr.Dataset(
        {
            "latitude": output_latitude.astype(dtype),
            "longitude": output_longitude.astype(dtype),
        }
    )
    return xe.Regridder(
        ds_in=input_grid,
        ds_out=output_grid,
        method=method,
    )


@click.command()
@click.argument(
    "local_input_filepath",
    type=click.Path(exists=True),
)
@click.argument(
    "local_output_filepath",
    type=click.Path(),
)
@click.option(
    "--land-threshold",
    type=float,
    default=LAND_THRESHOLD,
    help="Land threshold required to exclude SST values",
)
@click.option(
    "--ace2-era5-gcs-data",
    type=str,
    default=ACE2_ERA5_DATA,
    help="Path to ACE2 ERA5 data in GCS.",
)
def main(
    local_input_filepath: str,
    local_output_filepath: str,
    land_threshold: float,
    ace2_era5_gcs_data: str,
):
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Reading AIMIP forcing data from {local_input_filepath}")
    aimip_forcing_data = xr.open_dataset(
        local_input_filepath,
        chunks={},
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    logging.info("Processing AIMIP sea ice and ocean fractions.")
    # sea ice contains NaNs over land, fill with 0.0
    aimip_forcing_data = aimip_forcing_data.assign(
        {"sea_ice_cover": aimip_forcing_data.sea_ice_cover.fillna(0.0)}
    )
    ocean_fraction_out, sea_ice_fraction_out = derive_ocean_fraction(
        aimip_forcing_data.land_sea_mask, aimip_forcing_data.sea_ice_cover
    )
    aimip_forcing_data = aimip_forcing_data.assign(
        {
            "ocean_fraction": ocean_fraction_out,
            "sea_ice_cover": sea_ice_fraction_out,
            "mask": aimip_forcing_data.land_sea_mask < land_threshold,
        },
    )

    ace2_era5_data = xr.open_zarr(ace2_era5_gcs_data)
    logging.info("Getting regridder based on ACE2-ERA5 grid.")
    regridder_no_mask = get_regridder(
        input_latitude=aimip_forcing_data.latitude,
        input_longitude=aimip_forcing_data.longitude,
        output_latitude=ace2_era5_data.latitude,
        output_longitude=ace2_era5_data.longitude,
    )
    aimip_forcing_fractions_ace2_era5_grid = regridder_no_mask(
        aimip_forcing_data[
            [
                "land_sea_mask",
                "ocean_fraction",
                "sea_ice_cover",
            ]
        ]
    )
    for coord in ("latitude", "longitude"):
        xr.testing.assert_allclose(
            aimip_forcing_fractions_ace2_era5_grid[coord], ace2_era5_data[coord]
        )
    # regrid SST, in a way that masks out land points without inflating the land mask
    aimip_forcing_sst_ace2_era5_grid_adaptive_mask = regridder_no_mask(
        aimip_forcing_data.sea_surface_temperature.where(aimip_forcing_data.mask),
        skipna=True,
        na_thres=land_threshold,
    ).astype(np.float32)

    aimip_forcing_ace2_era5_grid = xr.merge(
        [
            aimip_forcing_fractions_ace2_era5_grid.rename(
                {
                    "sea_ice_cover": "sea_ice_fraction",
                    "land_sea_mask": "land_fraction",
                }
            ),
            aimip_forcing_sst_ace2_era5_grid_adaptive_mask.rename(
                "surface_temperature"
            ),
        ]
    )

    logging.info(f"Writing regridded AIMIP forcing data to {local_output_filepath}")
    logging.info(aimip_forcing_ace2_era5_grid.info())
    with ProgressBar():
        aimip_forcing_ace2_era5_grid.to_netcdf(local_output_filepath)


if __name__ == "__main__":
    main()
