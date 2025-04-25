import cftime
import xarray as xr


def date_range(
    n_timesteps: int,
    start_time: cftime.datetime = cftime.DatetimeGregorian(2020, 1, 1, 0, 0, 0),
    freq: str = "6h",
) -> xr.DataArray:
    """
    Create a DataArray with a time dimension of shape (n_timesteps,).
    """
    return xr.DataArray(
        xr.date_range(
            start_time,
            freq=freq,
            periods=n_timesteps,
            use_cftime=True,
        ).values,
        dims="time",
    )
