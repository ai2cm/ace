import datetime
from fcn_mip import schema, filesystem
from datasets.era5 import METADATA
import json
import xarray
import numpy as np


def _get_filename(time: datetime.datetime, lead_time: str):
    date_format = f"%Y%m%d/%Hz/0p4-beta/oper/%Y%m%d%H%M%S-{lead_time}-oper-fc.grib2"
    return time.strftime(date_format)


def _get_channel(c: str, **kwargs) -> xarray.DataArray:
    """

    Parameters:
    -----------
    c: channel id
    **kwargs: variables in ecmwf data
    """
    # handle 2d inputs
    if c in kwargs:
        return kwargs[c]
    else:
        varcode, pressure_level = c[0], int(c[1:])
        return kwargs[varcode].interp(isobaricInhPa=pressure_level)


def get(time: datetime.datetime, channel_set: schema.ChannelSet):
    root = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/"
    path = root + _get_filename(time, "0h")
    local_path = filesystem.download_cached(path)
    dataset_0h = xarray.open_dataset(local_path, engine="cfgrib")

    # get t2m and other things from 12 hour forecast initialized 12 hours before
    # The HRES is only initialized every 12 hours
    path = root + _get_filename(time - datetime.timedelta(hours=12), "12h")
    local_path = filesystem.download_cached(path)
    forecast_12h = xarray.open_dataset(local_path, engine="cfgrib")

    if channel_set == schema.ChannelSet.var34:
        # move to fcn_mip.channels
        metadata = json.loads(METADATA.read_text())
        channels = metadata["coords"]["channel"]
    else:
        raise NotImplementedError(channel_set)

    channel_data = [
        _get_channel(
            c,
            u10m=dataset_0h.u10,
            v10m=dataset_0h.v10,
            u100m=dataset_0h.u10,
            v100m=dataset_0h.v10,
            sp=dataset_0h.sp,
            t2m=forecast_12h.t2m,
            msl=forecast_12h.msl,
            tcwv=forecast_12h.tciwv,
            t=dataset_0h.t,
            u=dataset_0h.u,
            v=dataset_0h.v,
            r=dataset_0h.r,
            z=dataset_0h.gh * 9.81,
        )
        for c in channels
    ]

    array = np.stack([d for d in channel_data], axis=0)
    darray = xarray.DataArray(
        array,
        dims=["channel", "lat", "lon"],
        coords={
            "channel": channels,
            "lon": dataset_0h.longitude.values,
            "lat": dataset_0h.latitude.values,
            "time": time,
        },
    )
    return darray
