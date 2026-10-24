"""ERA5 / ERA5T -> ACE2S 4-degree daily atmosphere initial condition.

The ACE2S atmosphere the GLO12 fine-tune couples to steps once a day on the
``2026-09-08-era5-4deg-8layer-daily`` store: a factor-4 time coarsening of the
6-hourly ``scripts/era5`` output in which *snapshot* fields (surface pressure,
skin temperature, the 8-layer 3-D state, ...) are the 6-hourly sample at the
window end and *window* fields (fluxes, TOA insolation, ``*_mean``) are the
mean over the four 6-hourly means. A daily sample is stamped at its window
end, ``D+1 00:00``; the coupled store relabels it by -24 h so the sample for
day D carries the ocean's stamp ``D 00:00``.

So the atmosphere initial condition for issue day D needs ERA5 at one instant,
``D+1 00:00`` (the "one sample"), plus the hourly TOA insolation of day D for
the window mean, plus CO2 and the static fields. The instant is processed by
``scripts/era5``'s own ``_process_model_level_data`` (total water, pressure-
weighted 137 -> 8 layer coarsening at 0.25 degree, conservative regrid), from
either source:

- ``final``: the analysis-ready model-level store (final ERA5, ~3 months
  behind);
- ``era5t``: the cloud-optimized stores (ERA5T, ~1 week behind) through
  ``era5t_transform``.

Surface fields and TOA insolation come from ``full_37`` either way (ERA5T is
in that store). The land/sea/ice fractions are the GLO12 ocean's, as in the
coupled training store; HGTsfc and ak/bk are copied from the reference
atmosphere store.
"""

import datetime
import json
import logging

import era5t_transform
import numpy as np
import pandas as pd
import xarray as xr
from common import (
    CACHE_DIR,
    CO2_URL,
    OUTPUT_GRID,
    REFERENCE_ATMOS_STORE,
    era5_pipeline,
    open_zarr,
)

log = logging.getLogger("operational_ic.atmos")

ARCO_FULL_37 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_MODEL_LEVEL = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
TOA_VAR = "mean_top_downward_short_wave_radiation_flux"

# What the daily sample carries. The fine-tuned ACE2S reads the first group
# (its in_names less the fractions, which come from the ocean) and writes the
# second; the second is included only with ``full=True``.
PROGNOSTIC_AND_FORCING = (
    ["PRESsfc", "surface_temperature", "DSWRFtoa", "global_mean_co2", "HGTsfc"]
    + [
        f"{v}_{k}"
        for v in (
            "air_temperature",
            "specific_total_water",
            "eastward_wind",
            "northward_wind",
        )
        for k in range(8)
    ]
    + [f"{c}_{k}" for c in ("ak", "bk") for k in range(9)]
)
FREE_BY_PRODUCTS = ["TMP2m", "Q2m", "UGRD10m", "VGRD10m", "DPT2m", "PRMSL"]


def snapshot_time(day: pd.Timestamp) -> pd.Timestamp:
    """The ERA5 instant that is the daily sample of ``day``."""
    return day + pd.Timedelta(days=1)


def window_hours(day: pd.Timestamp) -> pd.DatetimeIndex:
    """Hourly ERA5 labels whose means cover day ``day``: (D 00, D+1 00]."""
    return pd.date_range(
        day + pd.Timedelta(hours=1), day + pd.Timedelta(hours=24), freq="1h"
    )


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def _open_arco(url: str) -> xr.Dataset:
    return xr.open_dataset(
        url, engine="zarr", chunks=None, cache=False, storage_options={"token": "anon"}
    )


def _store_stops(url: str) -> dict:
    a = _open_arco(url).attrs
    return {
        "final": a.get("valid_time_stop"),
        "era5t": a.get("valid_time_stop_era5t"),
        "last_updated": a.get("last_updated"),
    }


def _covers(stop: str | None, t: pd.Timestamp) -> bool:
    """Whether a ``valid_time_stop`` date (inclusive, whole day) covers ``t``."""
    return stop is not None and t <= pd.Timestamp(stop) + pd.Timedelta(hours=23)


def co2_coverage() -> pd.Timestamp:
    return pd.Timestamp(open_zarr(CO2_URL)["time"].values[-1])


def validate_atmos_day(day: pd.Timestamp, source: str = "auto") -> dict:
    """Decide the ERA5 source for ``day`` and report what is available.

    ``source``: ``final`` (analysis-ready model levels), ``era5t`` (cloud-
    optimized stores), or ``auto`` (final when it covers the day, else ERA5T).
    """
    t = snapshot_time(day)
    full37 = _store_stops(ARCO_FULL_37)
    ar = _store_stops(ARCO_MODEL_LEVEL)
    co = era5t_transform.valid_time_stops()
    report: dict = {
        "day": str(day.date()),
        "snapshot_time": str(t),
        "stores": {"full_37": full37, "ar_model_level": ar, "co_model_level": co},
        "problems": [],
        "warnings": [],
    }
    # surface fields at the instant + hourly TOA over the day: full_37 (ERA5T)
    if not (_covers(full37["era5t"], t) or _covers(full37["final"], t)):
        report["problems"].append(
            f"full_37 surface fields end {full37['era5t']} (ERA5T); need {t}"
        )
    final_ok = _covers(ar["final"], t)
    era5t_ok = all(_covers(co[k]["era5t"], t) for k in ("wind", "moisture"))
    if source == "auto":
        chosen = "final" if final_ok else ("era5t" if era5t_ok else None)
    elif source == "final":
        chosen = "final" if final_ok else None
    elif source == "era5t":
        chosen = "era5t" if era5t_ok else None
    else:
        raise ValueError(f"unknown atmosphere source {source!r}")
    if chosen is None:
        report["problems"].append(
            f"no model-level ERA5 for {t}: analysis-ready (final) store ends "
            f"{ar['final']}, cloud-optimized ERA5T ends wind {co['wind']['era5t']} / "
            f"moisture {co['moisture']['era5t']}"
        )
    elif chosen == "era5t":
        if not era5t_transform.is_populated(t):
            report["problems"].append(
                f"ERA5T chunks at {t} are not populated yet (store attributes run "
                "ahead of the data); try again later or an earlier day"
            )
        if not _covers(ar["final"], t):
            report["warnings"].append(
                "ERA5T initial condition: preliminary ERA5, may be revised when "
                "final ERA5 lands (~3 months)"
            )
    report["source"] = chosen
    co2_last = co2_coverage()
    report["co2_dataset_last"] = str(co2_last.date())
    if t > co2_last:
        report["warnings"].append(
            f"CO2 dataset ends {co2_last.date()}: global_mean_co2 extrapolated "
            "linearly from its last two years"
        )
    return report


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def model_level_final(t: pd.Timestamp) -> xr.Dataset:
    ep = era5_pipeline()
    log.info("final ERA5 model levels at %s (8 x 137 levels at 0.25 degree)", t)
    ds = _open_arco(ARCO_MODEL_LEVEL)[ep.MODEL_LEVEL_3D_VARS].sel(time=[t]).load()
    ep._check_data_validity(ds)
    return ds


def model_level(t: pd.Timestamp, source: str) -> xr.Dataset:
    if source == "final":
        return model_level_final(t)
    ds = era5t_transform.model_level_dataset(t)
    era5_pipeline()._check_data_validity(ds)
    return ds


def surface_fields(t: pd.Timestamp) -> xr.Dataset:
    ep = era5_pipeline()
    ds = (
        _open_arco(ARCO_FULL_37)[ep.FULL_37_MODEL_LEVEL_SURFACE_VARS]
        .sel(time=[t])
        .load()
    )
    ep._check_data_validity(ds)
    return ds


def toa_insolation_day(day: pd.Timestamp) -> xr.DataArray:
    """``DSWRFtoa`` of day ``day`` on the output grid: the mean of the 24
    hourly means, conservatively regridded (identical to the daily store's
    mean of four 6-hourly means)."""
    ep = era5_pipeline()
    hours = window_hours(day)
    da = _open_arco(ARCO_FULL_37)[TOA_VAR].sel(time=hours).load()
    if not np.isfinite(da.values).all():
        raise ValueError(f"{TOA_VAR} has NaN in {hours[0]}..{hours[-1]}")
    mean = da.mean("time", keepdims=True).assign_coords(
        time=[snapshot_time(day).to_datetime64()]
    )
    out = ep._regrid(mean.to_dataset(name="DSWRFtoa"), OUTPUT_GRID)["DSWRFtoa"]
    out.attrs = dict(ep.DESIRED_ATTRS.get("DSWRFtoa", {}))
    return out.drop_vars(["latitude", "longitude"], errors="ignore")


def co2_at(t: pd.Timestamp) -> tuple[float, str]:
    """``global_mean_co2`` at ``t`` from the ERA5 CO2 dataset, or a linear
    extrapolation of its last two years when ``t`` is beyond it."""
    co2 = open_zarr(CO2_URL)["global_mean_co2"].load()
    last = pd.Timestamp(co2.time.values[-1])
    if t <= last:
        return float(co2.sel(time=t, method="nearest")), "ERA5 CO2 dataset"
    tail = co2.sel(time=slice(last - pd.Timedelta(days=730), last))
    x = (pd.DatetimeIndex(tail.time.values) - last).days.values.astype(float)
    slope, intercept = np.polyfit(x, tail.values, 1)
    value = intercept + slope * (t - last).days
    return float(value), f"linear extrapolation from {str(last.date())}"


def atmosphere_statics(reference_store: str = REFERENCE_ATMOS_STORE) -> xr.Dataset:
    """HGTsfc and ak/bk of the training store (cached locally)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (
        reference_store.rsplit("/", 1)[-1].replace(".zarr", "-statics.nc")
    )
    if not path.exists():
        ref = open_zarr(reference_store)
        names = ["HGTsfc"] + [f"{c}_{k}" for c in ("ak", "bk") for k in range(9)]
        statics = ref[names].load()
        statics.attrs = {"source": reference_store}
        statics.to_netcdf(path)
    ds = xr.open_dataset(path).load()
    return ds.rename(
        {
            k: v
            for k, v in {"latitude": "lat", "longitude": "lon"}.items()
            if k in ds.dims
        }
    )


# ---------------------------------------------------------------------------
# The daily sample
# ---------------------------------------------------------------------------


def build_atmosphere_sample(
    day: pd.Timestamp,
    source: str,
    ocean_statics: xr.Dataset,
    ocean_sea_ice_fraction: xr.DataArray,
    reference_store: str = REFERENCE_ATMOS_STORE,
    full: bool = False,
) -> xr.Dataset:
    """The ACE2S daily sample for ``day``, stamped ``day 00:00``."""
    ep = era5_pipeline()
    t = snapshot_time(day)
    ds_model = model_level(t, source)
    ds_sfc = surface_fields(t)
    ak, bk = ep._get_ak_bk(ds_model)
    log.info("vertical coarsening + regrid to %s", OUTPUT_GRID)
    ml = ep._process_model_level_data(
        ds_model, ds_sfc, ak, bk, OUTPUT_GRID, ep.DEFAULT_OUTPUT_LAYER_INDICES
    )
    del ds_model
    out = xr.Dataset()
    for name in ml.data_vars:
        out[name] = ml[name]
    out["DSWRFtoa"] = toa_insolation_day(day).squeeze("time", drop=True)
    if full:
        out = out.merge(full_extras(day, t))
    out = out.rename({"latitude": "lat", "longitude": "lon"})
    out = out.expand_dims(time=[day.to_datetime64()])
    co2, co2_source = co2_at(t)
    out["global_mean_co2"] = xr.DataArray(
        np.array([co2], dtype=np.float64),
        dims=("time",),
        attrs={
            "long_name": "global mean CO2 concentration",
            "units": "volume mixing ratio",
            "source": co2_source,
        },
    )
    statics = atmosphere_statics(reference_store)
    for v in statics.data_vars:
        out[v] = statics[v].assign_coords(
            {c: out[c] for c in statics[v].dims if c in out.coords}
        )
    out = out.assign_coords(
        lat=ocean_statics["lat"].values, lon=ocean_statics["lon"].values
    )
    out = add_fractions(out, ocean_statics, ocean_sea_ice_fraction)
    keep = [
        v
        for v in out.data_vars
        if full or v in PROGNOSTIC_AND_FORCING + FREE_BY_PRODUCTS + FRACTIONS
    ]
    out = out[keep]
    for v in out.data_vars:
        if out[v].dtype == np.float64 and v != "global_mean_co2":
            out[v] = out[v].astype(np.float32)
    out.attrs = {
        "source": f"ERA5 ({source}) processed by "
        "scripts/era5/pipeline/xr-beam-pipeline.py functions",
        "era5_snapshot_time": str(t),
        "time_convention": "the daily sample of day D (ERA5 instant D+1 00Z, "
        "window means over day D) is stamped D 00:00, the GLO12 stamp",
        "fractions_source": "GLO12 ocean (land_fraction, sea_surface_fraction, "
        "sea_ice_fraction = concentration x sea_surface_fraction, "
        "ocean_fraction = 1 - land - ice)",
        "statics_source": statics.attrs.get("source", reference_store),
    }
    return out


FRACTIONS = [
    "land_fraction",
    "sea_surface_fraction",
    "sea_ice_fraction",
    "ocean_fraction",
]


def add_fractions(
    ds: xr.Dataset, ocean_statics: xr.Dataset, ocean_sea_ice_fraction: xr.DataArray
) -> xr.Dataset:
    """The coupled store's fractions: GLO12 land/sea fractions, whole-cell ice
    fraction (NaN where there is no ocean), ocean_fraction = 1 - land - ice."""
    lf = ocean_statics["land_fraction"].astype("float32")
    ssf = ocean_statics["sea_surface_fraction"].astype("float32")
    sic = ocean_sea_ice_fraction
    if "time" in sic.dims:
        sic = sic.isel(time=0, drop=True)
    sif = (sic.fillna(0) * ssf).where(ssf > 0).astype("float32")
    of = (1 - lf - sif.fillna(0)).clip(0, 1).astype("float32")
    ds["land_fraction"] = lf.assign_attrs(
        long_name="land fraction (GLO12 4deg build)", units="1"
    )
    ds["sea_surface_fraction"] = ssf.assign_attrs(
        long_name="ocean fraction of the cell (GLO12 4deg build)", units="1"
    )
    ds["sea_ice_fraction"] = sif.expand_dims(time=ds.time).assign_attrs(
        long_name="sea ice fraction of the whole cell "
        "(GLO12 concentration x sea_surface_fraction)",
        units="1",
    )
    ds["ocean_fraction"] = of.expand_dims(time=ds.time).assign_attrs(
        long_name="1 - land_fraction - sea_ice_fraction", units="1"
    )
    return ds


def full_extras(day: pd.Timestamp, t: pd.Timestamp) -> xr.Dataset:
    """The rest of the daily store's variable set (diagnostic outputs): the
    window means of the fluxes and surface fields over day ``day``, and the
    pressure-level and surface-analysis snapshots at ``t``. Costs ~3 GB more."""
    ep = era5_pipeline()
    hours = window_hours(day)
    f37 = _open_arco(ARCO_FULL_37)
    log.info("full mode: hourly fluxes over %s", day.date())
    flux = f37[ep.FULL_37_MEAN_FLUX_VARS].sel(time=hours).load()
    ep._check_data_validity(flux)
    flux_mean = flux.mean("time", keepdims=True).assign_coords(time=[t.to_datetime64()])
    out = ep._process_mean_flux(flux_mean, OUTPUT_GRID).squeeze("time", drop=True)
    log.info("full mode: hourly surface fields over %s", day.date())
    sfc = f37[ep.FULL_37_SURFACE_MEAN_VARS].sel(time=hours).load()
    ep._check_data_validity(sfc)
    # _process_surface_mean regrids hourly, derives Q2m/wind speed, then
    # averages its input window; feeding it the 24 hours gives the daily mean
    sm = ep._process_surface_mean(sfc, OUTPUT_GRID).squeeze("time", drop=True)
    out = out.merge(sm)
    log.info("full mode: pressure levels + surface analysis at %s", t)
    pl = f37[ep.FULL_37_PRESSURE_LEVEL_VARS].sel(time=[t]).load()
    ep._check_data_validity(pl)
    out = out.merge(
        ep._process_pressure_level_data(pl.squeeze("time", drop=True), OUTPUT_GRID)
    )
    inv = f37[ep.FULL_37_INVARIANT_VARS].sel(time=t).load()
    inv_r = ep._process_invariant(inv, OUTPUT_GRID)
    sa = f37[ep.FULL_37_SURFACE_ANALYSIS_VARS].sel(time=t).load()
    sa_r = ep._process_surface_analysis(sa, inv_r, OUTPUT_GRID)
    # the coupled store takes the fractions from the ocean, not from ERA5
    sa_r = sa_r.drop_vars(["sea_ice_fraction", "ocean_fraction"], errors="ignore")
    out = out.merge(sa_r)
    return out.drop_vars(["latitude", "longitude"], errors="ignore")


def availability_summary(day: pd.Timestamp, source: str = "auto") -> str:
    return json.dumps(validate_atmos_day(day, source), indent=1, default=str)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    d = (
        pd.Timestamp(sys.argv[1])
        if len(sys.argv) > 1
        else pd.Timestamp(datetime.date.today())
    )
    print(availability_summary(d))
