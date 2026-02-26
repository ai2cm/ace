# Compute the 1940-2021 ENSO/Nino 3.4 index from the CMIP6 AMIP SST forcing data
# Based on notebook here: github.com/ai2cm/explore/blob/master/brianh/2024-05-06-AMIP-interannual-variability/2024-07-03-compute-CMIP6-AMIP-ENSO-index.ipynb # noqa

import argparse
import dataclasses
from datetime import timedelta

import numpy as np
import xarray as xr

# SST_SOURCE = "gs://vcm-ml-intermediate/2023-10-06-AMIP-forcing-data/sst.nc"
# !gsutil cp {SST_SOURCE} .
SST_DATASET = "./sst.nc"
OCEAN_MASK_SOURCE = "gs://vcm-ml-intermediate/2023-10-27-vertically-resolved-1deg-fme-amip-ensemble-dataset/ic_0001.zarr"  # noqa
NINO_REGION_BOUNDS = dict(lat=(-5, 5), lon=(190, 240))
TROPICAL_REGION_BOUNDS = dict(lat=(-5, 5), lon=(0, 360))
OUTPUT_FILE = "./index.py"


@dataclasses.dataclass
class RegionBounds:
    lat: tuple[int, int]
    lon: tuple[int, int]

    def get_regional_average(self, ds, weights, lat_dim="lat", lon_dim="lon"):
        masked = (
            ds.where(ds[lat_dim] >= self.lat[0])
            .where(ds[lat_dim] <= self.lat[1])
            .where(ds[lon_dim] >= self.lon[0])
            .where(ds[lon_dim] <= self.lon[1])
        )
        return masked.weighted(weights).mean(dim=[lat_dim, lon_dim])


def open_dataset(path):
    """Open a dataset from a zarr store or a netCDF file."""
    if path.endswith(".zarr") or ".zarr/" in path:
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def get_ocean_mask(
    source,
    template: xr.DataArray,
    lat_dim="lat",
    lon_dim="lon",
    mask_var="ocean_fraction",
    mask_lat_dim="grid_yt",
    mask_lon_dim="grid_xt",
):
    """
    Get an ocean mask from the FV3GFS dataset. Regrid to the 1deg lat/lon grid
    of the SST dataset.
    """
    ds = open_dataset(source)
    ocean_fraction = (
        ds[mask_var]
        .isel(time=-1)
        .rename({mask_lon_dim: lon_dim, mask_lat_dim: lat_dim})
    )
    ocean_fraction_1deg = ocean_fraction.interp_like(template).load()
    ocean_mask = xr.where(ocean_fraction_1deg > 0.5, True, False)
    return ocean_mask


def get_time_average(da):
    # this version of xarray's resample method doesn't allow
    # data shifting to create a centered 3-month average, so do it manually
    da = da.assign_coords({"time": da.time + timedelta(days=45)})
    # the label is at the start of the 3-month season
    da_out = da.resample(time="QS").mean()
    return da_out


def get_anomalies(ds, time_groupby_key, time_dim="time"):
    full_groupby_key = f"{time_dim}.{time_groupby_key}"
    ds_monthly_climo = ds.groupby(full_groupby_key).mean(time_dim)
    ds_monthly_anomalies = ds.groupby(full_groupby_key) - ds_monthly_climo
    return ds_monthly_anomalies


def get_time_trendline(da):
    coeff = np.polyfit(np.arange(da.sizes["time"]), da.values, deg=1)
    da_trend = xr.DataArray(
        coeff[0] * np.arange(da.sizes["time"]) + coeff[1],
        dims="time",
        coords={"time": da.time},
    )
    return da_trend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst-dataset", default=SST_DATASET)
    parser.add_argument("--sst-var", default="sea_surface_temperature")
    parser.add_argument("--ocean-mask-source", default=OCEAN_MASK_SOURCE)
    parser.add_argument("--lat-dim", default="lat")
    parser.add_argument("--lon-dim", default="lon")
    parser.add_argument("--ocean-mask-var", default="ocean_fraction")
    parser.add_argument("--ocean-mask-lat-dim", default="grid_yt")
    parser.add_argument("--ocean-mask-lon-dim", default="grid_xt")
    parser.add_argument("--start-time", default="1940-01-01")
    parser.add_argument("--stop-time", default="2021-01-01")
    parser.add_argument("--detrend", action="store_true", default=False)
    args = parser.parse_args()

    surface_temperature = open_dataset(args.sst_dataset)[args.sst_var]
    surface_temperature = surface_temperature.sel(
        time=slice(args.start_time, args.stop_time)
    )
    ocean_mask = get_ocean_mask(
        args.ocean_mask_source,
        template=surface_temperature,
        lat_dim=args.lat_dim,
        lon_dim=args.lon_dim,
        mask_var=args.ocean_mask_var,
        mask_lat_dim=args.ocean_mask_lat_dim,
        mask_lon_dim=args.ocean_mask_lon_dim,
    )
    area_weights = np.cos(np.deg2rad(surface_temperature[args.lat_dim]))
    nino34_region = RegionBounds(**NINO_REGION_BOUNDS)
    tropical_region = RegionBounds(**TROPICAL_REGION_BOUNDS)

    nino34_temperature = nino34_region.get_regional_average(
        surface_temperature, area_weights, lat_dim=args.lat_dim, lon_dim=args.lon_dim
    )
    tropical_sst = tropical_region.get_regional_average(
        surface_temperature,
        weights=(area_weights * ocean_mask),
        lat_dim=args.lat_dim,
        lon_dim=args.lon_dim,
    )
    nino34_temperature_anom = nino34_temperature - tropical_sst
    nino34_temperature_anom_index = get_anomalies(
        nino34_temperature_anom, time_groupby_key="month"
    )
    if args.detrend:
        nino34_temperature_anom_index = (
            nino34_temperature_anom_index
            - get_time_trendline(nino34_temperature_anom_index)
        )
    nino34_anom_index = get_time_average(nino34_temperature_anom_index)

    with open(OUTPUT_FILE, "w") as f:
        print(
            (
                "# Nino3.4 index anomaly from tropical SST average, "
                "3-monthly centered running mean [K]"
            ),
            file=f,
        )
        print(
            "# computed from the CMIP6 AMIP SST forcing data using the script in",
            file=f,
        )
        print("# `scripts/monthly_data/compute_enso_index.py`", file=f)
        print("NINO34_INDEX = [", file=f)
        for point in nino34_anom_index:
            print(
                (
                    f"    (({point.time.item().year}, {point.time.item().month}, "
                    f"{point.time.item().day}), {point.item():0.3f}),"
                ),
                file=f,
            )
        print("]", file=f)


if __name__ == "__main__":
    main()
