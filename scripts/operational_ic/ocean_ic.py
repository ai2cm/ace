"""GLO12 operational analysis -> SamudrACE 4-degree ocean initial condition.

One GLO12 daily mean (1/12 degree, 50 levels) is turned into the training
store's layout by the GLORYS pipeline's own per-chunk functions
(``process_ocean_3d`` / ``process_ocean_2d``: fractional vertical remap to the
19 CM4 layers, conservative regrid with the precomputed F22.5 weights, masking,
coastal nearest-neighbour fill, sea-ice conventions). The static geometry is
not rebuilt: masks, fractions, ``deptho``, ``hfgeou`` and ``idepth`` are copied
from the training store, so the initial condition sits on exactly the land-sea
geometry the checkpoint was trained on (a different mask leaves unfilled NaN
cells and the rollout goes NaN at the first step).

Conventions preserved from the training store (``2026-10-22-glo12-ocean-4deg
-19level-5day-2022-2025`` + its ``cm4vars-v2`` derivative):

- a state stamped ``D 00:00`` is the GLO12 daily mean of day D;
- ``sea_ice_volume`` equals ``HI`` (the UFS replay convention the checkpoint
  reads; ``--sea_ice_volume per_area`` is available but is NOT what the
  deployed checkpoint saw);
- ``sfdsi``, ``UI``, ``VI`` are zero over ocean / NaN on land (GLO12's real ice
  velocities are deliberately not used), and ``hfds_total_area``, ``tauuo``,
  ``tauvo``, ``wfo``, ``unaccounted_heating`` are zero-filled (loss weight 0
  in the fine-tune; the corrector rebuilds hfds from the atmosphere).
"""

import datetime
import logging

import numpy as np
import pandas as pd
import xarray as xr
from common import (
    OUTPUT_GRID,
    REFERENCE_OCEAN_STORE,
    REGRID_WEIGHTS_URL,
    cached_gcs_file,
    glorys_pipeline,
    on_training_time_grid,
    open_zarr,
)

log = logging.getLogger("operational_ic.ocean")

# Channels the UFS-trained ocean reads/writes that GLO12 cannot supply; written
# as zeros over ocean exactly as the fine-tune's training store carries them.
ZERO_PLACEHOLDERS = {
    "sfdsi": ("placeholder (zero over ocean) for the sfdsi channel", "kg m-2 s-1"),
    "UI": (
        "placeholder (zero over ocean) for the UI channel; GLO12's usi is NOT used",
        "m s-1",
    ),
    "VI": (
        "placeholder (zero over ocean) for the VI channel; GLO12's vsi is NOT used",
        "m s-1",
    ),
    "hfds_total_area": (
        "zero-filled: not published by GLO12; loss weight 0, rebuilt by the "
        "corrector from the atmosphere",
        "W m-2",
    ),
    "tauuo": ("zero-filled: not published by GLO12; loss weight 0", "N m-2"),
    "tauvo": ("zero-filled: not published by GLO12; loss weight 0", "N m-2"),
    "wfo": ("zero-filled: not published by GLO12; loss weight 0", "kg m-2 s-1"),
    "unaccounted_heating": (
        "zero-filled: the corrector uses unaccounted_heating_source: constant",
        "W m-2",
    ),
}

# Fields the coupled stepper takes from the atmosphere at every step (the
# ocean's next_step_forcing_names): not part of an initial condition.
ATMOSPHERE_SUPPLIED = list(glorys_pipeline().FORCING_VARS.values())


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def glo12_times() -> pd.DatetimeIndex:
    """Time axis of the GLO12 analysis-forecast thetao store (daily means;
    the tail of the axis is the 10-day forecast, see ``validate_ocean_days``)."""
    gp = glorys_pipeline()
    ds = gp.open_cmems(gp.URLS_GLO12["thetao"])
    return pd.DatetimeIndex(ds["time"].values)


def _surface_sample_finite_fraction(day: pd.Timestamp) -> float:
    """Fraction of finite values in one surface tile of GLO12 thetao on
    ``day``: an unpublished day reads back as a missing chunk (all NaN)."""
    gp = glorys_pipeline()
    ds = gp.open_cmems(gp.URLS_GLO12["thetao"])["thetao"]
    surface = float(ds[gp.VDIM].max())  # elevation is negative, deepest first
    tile = (
        ds.sel(time=day)
        .sel({gp.VDIM: surface}, method="nearest")
        .sel(latitude=slice(-5, 5), longitude=slice(-170, -120))
        .load()
    )
    return float(np.isfinite(tile.values).mean())


def validate_ocean_days(
    days: pd.DatetimeIndex, today: datetime.date | None = None
) -> dict:
    """Check that every requested GLO12 day is an analysed, published day.

    GLO12 (GLOBAL_ANALYSISFORECAST_PHY_001_024) is updated daily: the days up
    to yesterday are analyses (the latest week may still be revised by the
    weekly best-analysis update), the days after today are the 10-day forecast
    that the ARCO store also carries. Forecast days are refused.
    """
    today = today or datetime.datetime.now(datetime.timezone.utc).date()
    axis = glo12_times()
    report: dict = {
        "glo12_first": str(axis[0].date()),
        "glo12_last_on_axis": str(axis[-1].date()),
        "today_utc": str(today),
        "days": [str(d.date()) for d in days],
        "problems": [],
        "warnings": [],
    }
    last_analysis = pd.Timestamp(today) - pd.Timedelta(days=1)
    for d in days:
        if d < axis[0]:
            report["problems"].append(f"{d.date()} precedes the GLO12 record")
        elif d not in axis:
            report["problems"].append(f"{d.date()} is not on the GLO12 time axis")
        elif d > last_analysis:
            report["problems"].append(
                f"{d.date()} is a GLO12 forecast day "
                f"(analyses end {last_analysis.date()})"
            )
        elif d > last_analysis - pd.Timedelta(days=7):
            report["warnings"].append(
                f"{d.date()} is within the last week: GLO12 may still revise it "
                "(weekly best analysis)"
            )
    if not report["problems"]:
        for d in (days[0], days[-1]):
            frac = _surface_sample_finite_fraction(d)
            if frac < 0.5:
                report["problems"].append(
                    f"{d.date()}: GLO12 thetao surface tile is {1 - frac:.0%} NaN "
                    "(not yet published?)"
                )
        report["sample_tile_checked"] = True
    report["ic_on_training_time_grid"] = bool(on_training_time_grid(days[-1]))
    if not report["ic_on_training_time_grid"]:
        report["warnings"].append(
            f"{days[-1].date()} is not on the training store's 1993-01-01 + 5-day "
            "grid (harmless for a forecast; noted for comparisons)"
        )
    return report


# ---------------------------------------------------------------------------
# Static inputs
# ---------------------------------------------------------------------------


def load_ocean_statics(reference_store: str = REFERENCE_OCEAN_STORE) -> xr.Dataset:
    """Every time-independent field of the training store, loaded."""
    ref = open_zarr(reference_store)
    names = [v for v in ref.data_vars if "time" not in ref[v].dims]
    statics = ref[names].load()
    statics.attrs = {"source": reference_store}
    for v in ("mask_2d", "mask_0", "sea_surface_fraction", "deptho", "hfgeou"):
        if v not in statics:
            raise ValueError(f"{reference_store} has no static {v}")
    return statics


class OceanProcessor:
    """Holds the one-off inputs (overlap weights, source grid, regrid weights,
    statics) and turns GLO12 days into training-store states."""

    def __init__(
        self,
        reference_store: str = REFERENCE_OCEAN_STORE,
        regrid_weights: str = REGRID_WEIGHTS_URL,
        sea_ice_volume: str = "thickness",
        placeholders: bool = True,
    ):
        gp = glorys_pipeline()
        self.gp = gp
        self.sea_ice_volume = sea_ice_volume
        self.placeholders = placeholders
        self.statics = load_ocean_statics(reference_store)
        coords = gp.open_cmems(gp.URL_COORDS)
        self.weights = gp.overlap_matrix(coords["e3t"].load(), gp.TARGET_INTERFACES)
        bathy = gp.open_cmems(gp.URL_BATHY)
        self.source_grid = gp._make_source_grid(
            bathy["latitude"].values, bathy["longitude"].values
        )
        self.weights_path = (
            str(cached_gcs_file(regrid_weights))
            if regrid_weights.startswith("gs://")
            else regrid_weights
        )
        self._common = dict(
            output_grid=OUTPUT_GRID,
            source_grid=self.source_grid,
            invariant_ds=self.statics,
            weights_path=self.weights_path,
        )

    def process_days(self, days: pd.DatetimeIndex) -> xr.Dataset:
        """Training-store states for ``days`` (time, lat, lon), statics included."""
        import xarray_beam as xbeam

        gp = self.gp
        ds_3d = gp.open_ocean(gp.VARS_3D, days, "glo12")
        ds_2d = gp.open_ocean(gp.VARS_2D, days, "glo12")
        states = []
        for i, day in enumerate(days):
            parts = []
            for name in gp.VARS_3D:
                log.info("%s: GLO12 %s (50 levels at 1/12 degree)", day.date(), name)
                key = xbeam.Key({"time": i}, vars={name})
                _, out = gp.process_ocean_3d(
                    key,
                    ds_3d[[name]].isel(time=[i]),
                    weights=self.weights,
                    **self._common,
                )
                parts.append(out)
            log.info("%s: GLO12 2-D fields", day.date())
            key = xbeam.Key({"time": i}, vars=set(gp.VARS_2D))
            _, out = gp.process_ocean_2d(
                key,
                ds_2d.isel(time=[i]),
                sea_ice_volume=self.sea_ice_volume,
                diagnostic_placeholders=False,
                **self._common,
            )
            parts.append(out)
            states.append(xr.merge(parts))
        ds = xr.concat(states, dim="time")
        attrs = gp._output_attrs(ds_3d, xr.Dataset(), self.sea_ice_volume)
        for v in ds.data_vars:
            ds[v].attrs = dict(attrs.get(v, ds[v].attrs))
        if self.placeholders:
            ds = add_placeholders(ds)
        for v in self.statics.data_vars:
            ds[v] = self.statics[v]
        ds.attrs = {
            "source": "GLO12 GLOBAL_ANALYSISFORECAST_PHY_001_024 daily means, "
            "processed by scripts/glorys/pipeline/glorys-pipeline.py functions",
            "statics_source": self.statics.attrs["source"],
            "sea_ice_volume_convention": self.sea_ice_volume,
            "time_convention": "a state stamped D 00:00 is the GLO12 daily mean "
            "of day D",
        }
        return ds


def add_placeholders(ds: xr.Dataset) -> xr.Dataset:
    """Zero placeholder channels over ocean (NaN on land), overwriting UI/VI."""
    ocean = ds["thetao_0"].notnull()
    for name, (long_name, units) in ZERO_PLACEHOLDERS.items():
        ds[name] = xr.zeros_like(ds["thetao_0"]).where(ocean).astype("float32")
        ds[name].attrs = {"long_name": long_name, "units": units}
    return ds


def initial_condition(states: xr.Dataset, how: str) -> xr.Dataset:
    """The single ocean initial state from the processed daily states.

    ``day``: the last day's state (the training convention: one daily mean
    every 5 days). ``mean5``: the mean of the last five daily states, stamped
    at the last day; smoother than any training state, offered for experiments.
    """
    if how == "day":
        ic = states.isel(time=[-1])
        note = "GLO12 daily mean of the stamped day (training convention)"
    elif how == "mean5":
        if states.sizes["time"] < 5:
            raise ValueError("mean5 needs five processed days")
        last = states.isel(time=slice(-5, None))
        tv = [v for v in last.data_vars if "time" in last[v].dims]
        ic = last[tv].mean("time", keep_attrs=True).expand_dims(time=last.time[-1:])
        for v in states.data_vars:
            if v not in tv:
                ic[v] = states[v]
        note = (
            f"mean of the GLO12 daily means {str(last.time.values[0])[:10]}.."
            f"{str(last.time.values[-1])[:10]}, stamped at the window end; NOT the "
            "training convention (one daily mean), use for experiments only"
        )
    else:
        raise ValueError(f"unknown ocean IC recipe {how!r}")
    ic = ic.drop_vars([v for v in ATMOSPHERE_SUPPLIED if v in ic], errors="ignore")
    ic.attrs = dict(states.attrs, initial_condition=note)
    return ic
