import logging
from datetime import datetime

import cftime
import click
import xarray as xr
from dask.diagnostics import ProgressBar
from encoding import clear_encoding

# this script is based on the notebook at
# https://github.com/ai2cm/explore2/blob/main/troya/2025-08-06-AIMP-ERA5/2025-08-05-ARCO-ERA5-monthly-average-forcing-AIMIP.ipynb

ERA5_ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
MONTHLY_FORCING_VARIABLES = [
    "sea_ice_cover",
    "sea_surface_temperature",
]

STATIC_FORCING_VARIABLES = [
    "land_sea_mask",
]

DATE_UNITS = "hours since 1900-01-01T00:00:00.000000000"
CALENDAR = "proleptic_gregorian"
DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def get_era5_arco(
    era5_arco_path: str,
    start_date: str,
    end_date: str,
    variables: list[str],
    stride: int | None = None,
) -> xr.Dataset:
    era5_numeric_time_coord = xr.open_zarr(
        era5_arco_path,
        storage_options=dict(token="anon"),
        chunks={},
        decode_times=False,
    )
    era5_numeric_time_coord = era5_numeric_time_coord[variables]
    era5_numeric_time_coord_subset = era5_numeric_time_coord.sel(
        time=slice(datestr_to_datenum(start_date), datestr_to_datenum(end_date), stride)
    )
    return era5_numeric_time_coord_subset


def get_era5_static_snapshot(ds: xr.Dataset, timestamp: str) -> xr.Dataset:
    """
    Get a static snapshot of the ERA5 data at the given timestamp.
    """
    return ds.sel(time=datestr_to_datenum(timestamp)).drop_vars("time")


def datestr_to_datenum(
    datestr: str,
    units: str = DATE_UNITS,
    calendar: str = CALENDAR,
    date_fmt: str = DATE_FMT,
):
    """Convert a date string to a numeric date representation."""
    datetime_ = datetime.strptime(datestr, date_fmt)
    return datetime_to_datenum(datetime_, units=units, calendar=calendar).item()


def datetime_to_datenum(
    datetime_: datetime, units: str = DATE_UNITS, calendar: str = CALENDAR
):
    """
    Convert a datetime object to a numeric date representation using cftime.
    """
    return cftime.date2num(datetime_, units, calendar=calendar)


def get_monthly_midpoints(start_date: str, end_date: str):
    """
    Given start and end dates, return all intervening monthly midpoints.
    Used in binning time coordinate to monthly means with start-of-month-centered bins,
    where the monthly midpoints are the bin boundaries.

    Args:
        start_date: Start date in the format "YYYY-MM-DDTHH:MM"
        end_date: End date in the format "YYYY-MM-DDTHH:MM"

    Returns:
        monthly_midpoints (np.ndarray): Array of monthly midpoints in numeric date
            format.
        monthly_starts_datetime (xr.DataArray): Array of monthly start dates as datetime
            objects, length one greater than monthly_midpoints.
    """
    monthly_starts_datetime = xr.date_range(
        start=start_date,
        end=end_date,
        freq="MS",
        use_cftime=True,
        calendar=CALENDAR,
    )
    monthly_starts = datetime_to_datenum(monthly_starts_datetime)
    monthly_midpoints = (
        monthly_starts[:-1] + (monthly_starts[1:] - monthly_starts[:-1]) // 2
    )
    return monthly_midpoints, monthly_starts_datetime


def encoding_and_attrs(ds: xr.Dataset, attrs_source_ds: xr.Dataset) -> xr.Dataset:
    ds = clear_encoding(ds)
    ds = copy_variable_attributes(attrs_source_ds, ds)
    ds = set_global_attributes(ds)
    return ds


def set_global_attributes(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove attributes from the ARCO dataset and add new global attributes.
    """
    ds = ds.drop_attrs()
    ds.attrs["title"] = "AIMIP ERA5 Monthly Mean Forcing"
    ds.attrs["History"] = (
        "Made from ERA5 ARCO data, using start-of-month-centered monthly means"
    )
    return ds


def copy_variable_attributes(ds_in, ds_out):
    for variable in {**ds_out.coords, **ds_out.data_vars}.keys():
        ds_out[variable].attrs = ds_in[variable].attrs.copy()
    return ds_out


@click.command()
@click.argument(
    "output_filepath",
    type=click.Path(),
)
@click.option(
    "--month-start-date",
    type=str,
    default="1978-09-01T00:00:00",
    help=(
        "Start date for subset; should be a start of a month. Note that in computing "
        "monthly means centered on the start of the month, the resulting series will "
        "start one month after this month."
    ),
)
@click.option(
    "--month-end-date",
    type=str,
    default="2025-02-01T00:00:00",
    help=(
        "End date for subset; should be a start of a month. Note that in computing "
        "monthly means centered on the start of the month, the resulting series will "
        "end one month before this month."
    ),
)
@click.option(
    "--era5-arco", type=str, default=ERA5_ARCO, help="Path to ERA5 ARCO data in GCS."
)
def main(
    output_filepath: str, era5_arco: str, month_start_date: str, month_end_date: str
):
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Reading ERA5 ARCO data from {era5_arco}.")
    era5 = get_era5_arco(
        era5_arco,
        month_start_date,
        month_end_date,
        variables=(MONTHLY_FORCING_VARIABLES + STATIC_FORCING_VARIABLES),
        stride=6,  # sample six-hourly to get diurnal cycle without full dataset
    )

    monthly_midpoints, monthly_starts_datetime = get_monthly_midpoints(
        month_start_date,
        month_end_date,
    )

    logging.info("Computing monthly means for ERA5 dynamic data.")
    era5_forcing_monthly_mean_start_of_month_centered = (
        era5[MONTHLY_FORCING_VARIABLES]
        .groupby_bins(
            group="time",
            bins=monthly_midpoints,
            right=False,
            include_lowest=True,
            labels=monthly_starts_datetime[1:-1],
        )
        .mean()
        .rename({"time_bins": "time"})
    )

    logging.info("Getting ERA5 static data snapshot.")
    era5_static_snapshot = get_era5_static_snapshot(
        era5[STATIC_FORCING_VARIABLES],
        timestamp=month_start_date,
    )

    logging.info("Merging ERA5 static and dynamic data.")
    aimip_era5_forcing_dataset = xr.merge(
        [era5_static_snapshot, era5_forcing_monthly_mean_start_of_month_centered]
    )

    logging.info("Setting encoding and attributes.")
    aimip_era5_forcing_dataset = encoding_and_attrs(
        aimip_era5_forcing_dataset,
        attrs_source_ds=era5,
    )

    logging.info(f"Writing AIMIP ERA5 forcing data to {output_filepath}.")
    logging.info(aimip_era5_forcing_dataset.info())
    with ProgressBar():
        aimip_era5_forcing_dataset.to_netcdf(output_filepath)


if __name__ == "__main__":
    main()
