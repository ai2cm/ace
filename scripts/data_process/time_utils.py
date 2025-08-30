import xarray as xr


def shift_timestamps_to_midpoint(ds: xr.Dataset, time_dim: str = "time"):
    """
    Shifts the time coordinate to the midpoint of the interval between
    subsequent timestamps.

    This is useful, e.g., for data where the timestamp represents the end of an
    averaging period and we instead want the timestamp to sit in the middle of
    that period.

    Args:
        ds: The input xarray.Dataset, with a time dimension.
        time_dim: The name of the time dimension.

    Returns:
        A new xr.Dataset with shifted timestamps if the config flag
        is True, otherwise the original dataset.
    """
    time_coord = ds[time_dim]
    dt = (time_coord.values[1] - time_coord.values[0]) / 2
    new_coord = time_coord - dt
    new_coord.attrs["long_name"] = "time, interval midpoint"
    ds = ds.assign_coords(time=new_coord)
    return ds
