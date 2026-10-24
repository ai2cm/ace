"""Forcing datasets for a coupled SamudrACE forecast horizon.

During a coupled rollout the model reads, beyond the initial condition, only
exogenous forcing: for the atmosphere, TOA insolation (``DSWRFtoa``),
``global_mean_co2``, ``HGTsfc`` and ``land_fraction``; for the ocean, the
static ``sea_surface_fraction``, ``deptho`` and ``hfgeou``. Everything else is
exchanged between the components. This writes exactly those, on the time axes
``fme.coupled.inference`` expects (daily atmosphere, 5-daily ocean, starting
at the initial condition), replacing the extended-store hack of
``extend_stores_2026.py``:

- ``DSWRFtoa``: the calendar-day climatology of the reference atmosphere
  store (TOA insolation has no weather and a negligible year-to-year change);
- ``global_mean_co2``: the ERA5 CO2 dataset, extrapolated linearly from its
  last two years beyond its end;
- static fields: copied from the training stores through the ocean statics
  and the atmosphere statics of the initial condition.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from common import (
    CACHE_DIR,
    CO2_URL,
    OCEAN_STRIDE_DAYS,
    REFERENCE_ATMOS_STORE,
    open_zarr,
    set_time_encoding,
)

log = logging.getLogger("operational_ic.forcing")

CLIM_YEARS = (2010, 2024)


def toa_climatology(
    reference_store: str = REFERENCE_ATMOS_STORE, years: tuple[int, int] = CLIM_YEARS
) -> xr.DataArray:
    """``DSWRFtoa`` by calendar day (month-day, 366 entries) on the output
    grid, from the reference daily store relabelled to the coupled stamp
    (-24 h). Cached under CACHE_DIR."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"dswrftoa-calendar-day-climatology-{years[0]}-{years[1]}.nc"
    if not path.exists():
        ref = open_zarr(reference_store)["DSWRFtoa"]
        da = ref.sel(time=slice(f"{years[0]}-01-02", f"{years[1] + 1}-01-01")).load()
        da = da.assign_coords(time=da.time - pd.Timedelta("24h"))
        da = da.rename(
            {
                k: v
                for k, v in {"latitude": "lat", "longitude": "lon"}.items()
                if k in da.dims
            }
        )
        md = da.time.dt.month * 100 + da.time.dt.day
        clim = da.groupby(md.rename("monthday")).mean("time")
        clim.name = "DSWRFtoa"
        clim.attrs = {
            "long_name": "Downward SW radiative flux at TOA, calendar-day climatology",
            "units": "W/m**2",
            "source": f"{reference_store}, {years[0]}-{years[1]}, "
            "daily means relabelled -24 h",
        }
        clim.to_netcdf(path)
    return xr.open_dataarray(path).load()


def co2_series(times: pd.DatetimeIndex) -> xr.DataArray:
    co2 = open_zarr(CO2_URL)["global_mean_co2"].load()
    last = pd.Timestamp(co2.time.values[-1])
    inside = times[times <= last]
    values = np.empty(len(times), dtype=np.float64)
    if len(inside):
        values[: len(inside)] = co2.sel(time=inside, method="nearest").values
    if len(inside) < len(times):
        tail = co2.sel(time=slice(last - pd.Timedelta(days=730), last))
        x = (pd.DatetimeIndex(tail.time.values) - last).days.values.astype(float)
        slope, intercept = np.polyfit(x, tail.values, 1)
        beyond = times[len(inside) :]
        values[len(inside) :] = intercept + slope * (beyond - last).days.values
        note = (
            f"ERA5 CO2 dataset to {last.date()}, then linear extrapolation "
            f"({slope * 365:.2e}/yr)"
        )
    else:
        note = "ERA5 CO2 dataset"
    return xr.DataArray(
        values,
        dims=("time",),
        coords={"time": times},
        attrs={
            "long_name": "global mean CO2 concentration",
            "units": "volume mixing ratio",
            "source": note,
        },
    )


def atmosphere_forcing(
    day: pd.Timestamp,
    n_coupled_steps: int,
    atmos_ic: xr.Dataset,
    reference_store: str = REFERENCE_ATMOS_STORE,
) -> xr.Dataset:
    """Daily atmosphere forcing from ``day`` through the horizon (+ one spare
    day), with the initial condition's own values on the first day."""
    n_days = n_coupled_steps * OCEAN_STRIDE_DAYS + 1
    times = pd.date_range(day, periods=n_days + 1, freq="1D")
    clim = toa_climatology(reference_store)
    md = times.month * 100 + times.day
    toa = clim.sel(monthday=xr.DataArray(md, dims="time")).drop_vars("monthday")
    toa = (
        toa.assign_coords(time=times).transpose("time", "lat", "lon").astype("float32")
    )
    toa = toa.assign_coords(lat=atmos_ic["lat"].values, lon=atmos_ic["lon"].values)
    toa[0] = atmos_ic["DSWRFtoa"].isel(time=0).values
    toa.attrs = {
        "long_name": "Downward SW radiative flux at TOA",
        "units": "W/m**2",
        "source": "day 0 from the ERA5 initial condition, "
        "then the calendar-day climatology",
    }
    out = xr.Dataset({"DSWRFtoa": toa, "global_mean_co2": co2_series(times)})
    for v in ("HGTsfc", "land_fraction", "sea_surface_fraction"):
        out[v] = atmos_ic[v]
    out.attrs = {
        "note": "exogenous forcing for fme.coupled.inference: TOA insolation "
        "climatology, CO2 trend, static fields",
        "initial_condition": str(day.date()),
    }
    return set_time_encoding(out)


def ocean_forcing(
    day: pd.Timestamp, n_coupled_steps: int, ocean_statics: xr.Dataset
) -> xr.Dataset:
    """5-daily ocean forcing: the training store's static fields on the
    horizon's time axis (the atmosphere supplies every time-varying input)."""
    times = pd.date_range(
        day, periods=n_coupled_steps + 2, freq=f"{OCEAN_STRIDE_DAYS}D"
    )
    out = ocean_statics.copy()
    out = out.assign_coords(time=times)
    # the data loader infers the horizontal dims from a (time, lat, lon)
    # variable, so a store of static fields alone cannot be opened
    marker = xr.zeros_like(out["sea_surface_fraction"]).expand_dims(time=times)
    out["ocean_forcing_placeholder"] = marker.astype("float32")
    out["ocean_forcing_placeholder"].attrs = {
        "long_name": "zero placeholder so the store has a time-varying variable",
        "units": "1",
    }
    out.attrs = {
        "note": "static ocean inputs (sea_surface_fraction, deptho, hfgeou, "
        "land_fraction, masks) for fme.coupled.inference",
        "initial_condition": str(day.date()),
        "source": ocean_statics.attrs.get("source", ""),
    }
    return set_time_encoding(out)
