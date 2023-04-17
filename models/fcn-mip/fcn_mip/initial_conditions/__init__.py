import config
import xarray
import datetime
from fcn_mip import schema
import joblib
import numpy as np
from fcn_mip.initial_conditions.era5 import open_era5_xarray
from fcn_mip.initial_conditions import ifs

# hack...shouldn't be imported from here
from datasets.era5 import METADATA
import json

__all__ = ["open_era5_xarray", "get"]


def get(
    n_history: int,
    time: datetime.datetime,
    channel_set: schema.ChannelSet,
    source: schema.InitialConditionSource = schema.InitialConditionSource.era5,
) -> xarray.DataArray:
    if source == schema.InitialConditionSource.era5:
        ds = open_era5_xarray(time, channel_set)
        subset = ds.sel(time=slice(None, time))[-n_history - 1 :]
        assert subset.sizes["time"] == n_history + 1
        return subset.load()
    elif source == schema.InitialConditionSource.ifs:
        if n_history > 0:
            raise NotImplementedError("IFS initializations only work with n_history=0.")
        ds = ifs.get(time, channel_set)
        ds = ds.expand_dims("time", axis=0)
        # move to fcn_mip.channels

        # TODO refactor interpolation to another place
        metadata = json.loads(METADATA.read_text())
        lat = np.array(metadata["coords"]["lat"])
        lon = np.array(metadata["coords"]["lon"])
        ds = ds.roll(lon=len(ds.lon) // 2, roll_coords=True)
        ds["lon"] = ds.lon.where(ds.lon >= 0, ds.lon + 360)
        assert min(ds.lon) >= 0, min(ds.lon)
        return ds.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})
    else:
        raise NotImplementedError(source)


local_cache = getattr(config, "LOCAL_CACHE", None)
if local_cache:
    memory = joblib.Memory(local_cache)
    get = memory.cache(get)


def ic(
    time: datetime,
    grid,
    n_history: int,
    channel_set: schema.ChannelSet,
    source: schema.InitialConditionSource,
):
    ds = get(n_history, time, channel_set, source)
    # TODO collect grid logic in one place
    if grid == schema.Grid.grid_720x1440:
        return ds.isel(lat=slice(0, -1))
    elif grid == schema.Grid.grid_721x1440:
        return ds
    else:
        raise NotImplementedError(f"Grid {grid} not supported")
