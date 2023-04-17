import os
from datasets import era5
import config
import xarray
import datetime
import s3fs
import json
from fcn_mip import filesystem, schema

__all__ = ["open_era5_xarray"]


def _get_path(path, time):
    filename = time.strftime("%Y.h5")
    if time.year < 2015:
        return os.path.join(path, "train", filename)
    elif time.year < 2018:
        return os.path.join(path, "test", filename)
    elif time.year == 2018:
        return os.path.join(path, "out_of_sample", filename)
    else:
        raise KeyError(time.year)


def open_era5_xarray(
    time: datetime.datetime, channel_set: schema.ChannelSet
) -> xarray.DataArray:

    if channel_set == schema.ChannelSet.var34:
        try:
            path = config.INITIAL_CONDITION_DIRECTORY
        except AttributeError:
            pass
        else:
            path = _get_path(path, time)

        # A second best option. A full mirror of the data is not available in NGC
        try:
            path = config.ERA5_2018_PATH
        except AttributeError:
            pass
    elif channel_set == schema.ChannelSet.var73:
        path = _get_path(config.CHANNEL_76_DATA, time)
    else:
        raise NotImplementedError(channel_set)

    if path.endswith(".h5"):
        if path.startswith("s3://"):
            fs = s3fs.S3FileSystem(
                client_kwargs=dict(endpoint_url="https://pbss.s8k.io")
            )
            f = fs.open(path)
        else:
            f = None
        if channel_set == schema.ChannelSet.var34:
            ds = era5.open_34_vars(path, f=f)
        else:
            metadata_path = os.path.join(config.CHANNEL_76_DATA, "data.json")
            metadata_path = filesystem.download_cached(metadata_path)
            with open(metadata_path) as mf:
                metadata = json.load(mf)
            ds = era5.open_hdf5(path=path, f=f, metadata=metadata)
    elif path.endswith(".nc"):
        ds = xarray.open_dataset(path).fields

    return ds
