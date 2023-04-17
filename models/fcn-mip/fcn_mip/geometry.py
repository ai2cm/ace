"""Routines for working with geometry"""
import numpy as np
import torch
import xarray as xr

LAT_AVERAGE = "LatitudeAverage"


def sel_channel(model, channel_info, data, channels):
    channels = np.asarray(channels)
    torch_indices = list(model.channels)
    channels_in_data = np.asarray(channel_info)[torch_indices].tolist()
    index_to_select = [channels_in_data.index(ch) for ch in channels]
    return data[:, index_to_select]


def get_batch_size(data):
    return data.shape[0]


def get_bounds_window(geom, lat, lon):
    i_min = np.where(lat <= geom.lat_max)[0][0]
    i_max = np.where(lat >= geom.lat_min)[0][-1]
    j_min = np.where(lon >= geom.lon_min)[0][0]
    j_max = np.where(lon <= geom.lon_max)[0][-1]
    return slice(i_min, i_max + 1), slice(j_min, j_max + 1)


def select_space(data, lat, lon, domain):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    assert data.ndim == 4, data.ndim
    assert data.shape[2] == lat.size, lat.size
    assert data.shape[3] == lon.size, lon.size
    domain_type = domain.type
    if domain_type == "Window" or domain_type == LAT_AVERAGE or domain_type == "global":
        lat_sl, lon_sl = get_bounds_window(domain, lat, lon)
        domain_lat = lat[lat_sl]
        domain_lon = lon[lon_sl]
        return domain_lat, domain_lon, data[:, :, lat_sl, lon_sl]
    elif domain_type == "MultiPoint":
        # Convert lat-long points to array index (just got to closest 0.25 degree)
        i = lat.size - np.searchsorted(lat[::-1], domain.lat, side="right")
        j = np.searchsorted(lon, domain.lon, side="left")
        # TODO refactor this assertion to a test
        np.testing.assert_array_equal(domain.lat, lat[i])
        np.testing.assert_array_equal(domain.lon, lon[j])
        return lat[i], lon[j], data[:, :, i, j]
    elif domain_type == "CWBDomain":
        cwb_path = "/lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"
        xlat = xr.open_zarr(cwb_path)["XLAT"]
        xlong = xr.open_zarr(cwb_path)["XLONG"]
        array = data.cpu().numpy()
        diagnostic = domain["diagnostics"][0]
        darray = xr.DataArray(
            array,
            dims=["batch", "channel", "lat", "lon"],
            coords={"lat": lat, "lon": lon, "channel": diagnostic.channels},
        )
        interpolated = darray.interp(lat=xlat, lon=xlong)
        return xlat, xlong, torch.from_numpy(interpolated.values)
    else:
        raise ValueError(
            f"domain {domain_type} is not supported. Check the weather_events.json"
        )


def bilinear(data: torch.tensor, dims, source_coords, target_coords):

    return
