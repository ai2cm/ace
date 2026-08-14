# %% [markdown]
# # CRPS and spread-skill ratio for the video PMD test-set ensemble
#
# Verifies the 32-member ensemble from `global-1degree-24to3-pcn-v1`'s test-set
# inference (endpoint-conditioned video diffusion: given observed 0h/24h daily
# snapshots, infill the 7 interior 3-hourly frames). Two standard probabilistic
# scores:
#
# - **CRPS** (Continuous Ranked Probability Score): generalizes MAE to a full
#   ensemble/distribution. Lower is better; a deterministic forecast's CRPS
#   equals its MAE, so CRPS below the model's own ensemble-mean MAE means the
#   ensemble spread is adding real skill, not just noise.
# - **Spread-skill ratio**: ensemble spread (std across members, finite-ensemble
#   corrected) divided by the ensemble-mean's RMSE against truth. A
#   *reliable* ensemble has ratio approx 1 -- the spread should predict the
#   actual error. Ratio < 1 is **underdispersive** (overconfident: true error
#   is bigger than the ensemble admits); ratio > 1 is **overdispersive**
#   (the ensemble is wider than it needs to be).
#
# ### Why "fair" CRPS
#
# The naive plug-in CRPS estimator (`E|X-y| - 0.5*E|X-X'|` over the raw
# ensemble, including self-pairs) is **biased high** for finite ensemble size
# M -- at M=32 the bias is still a few percent. The "fair" estimator (Ferro et
# al. 2008) excludes self-pairs, correcting this:
#
# `CRPS_fair = mean_i|x_i - y| - 1/(M(M-1)) * sum_{i<j} |x_i - x_j|`
#
# computed here via a memory-efficient sorted-order-statistics identity
# (avoids an O(M^2) pairwise array at grid scale):
# `sum_{i<j}|x_i-x_j| = sum_k (2k-M-1) x_(k)` for sorted `x_(1)<=...<=x_(M)`.
#
# ### Scope
#
# Only **interior (generated) frames** are scored -- the 0h/24h endpoints are
# deterministic broadcasts of the observed truth (by construction, see
# `fme/downscaling/video_inference.py`), so they trivially have zero spread
# and are not informative for probabilistic skill. `frame_source` in the
# output zarr flags which is which.
#
# ### Sampling
#
# A single contiguous window is weather-regime- and season-biased (e.g. one
# January week only samples NH winter synoptics). Instead this samples a few
# days from each of four seasons (`SAMPLE_WINDOWS` below) -- the store's
# per-`(time, ensemble)` chunking makes reading the *entire* test period
# (2023-01-01 to 2024-01-04) expensive for an interactive notebook, so this is
# a middle ground: representative of the full annual cycle without reading
# everything. Extend `SAMPLE_WINDOWS` (more/longer windows) for tighter
# confidence intervals -- each 3-day window took ~70s to read+score in
# testing, so budget accordingly (4 windows ~ a few minutes).

# %%
import argparse
import gc

import cftime
import matplotlib

matplotlib.use("Agg")  # headless: this runs as a batch script, not a notebook kernel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

plt.rcParams["figure.dpi"] = 100
# Un-truncate printed tables -- the default column/width limits were silently
# hiding the CRPS/spread/RMSE columns of the "Overall" summary table (and the
# "spread" column of the lead-time table) behind "..." in the log, leaving
# only "ratio" fully legible.
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %%
# Known inference outputs, keyed by a short label. Add an entry here for each
# new test-set inference run rather than duplicating this whole script --
# `--pred-zarr` also accepts an arbitrary path directly for one-offs.
KNOWN_MODELS = {
    "pcn-v1": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "global-1degree-24to3-pcn-v1/test-2023-2024-ens32.zarr"
    ),
    "bb-pcn": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-bb-pcn-global-1degree-24to3-v1/test-2023-2024-ens32.zarr"
    ),
    "bb-subset-cons0": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-bb-pcn-subset-cons0-global-1degree-24to3-v1/"
        "test-2023-2024-ens32.zarr"
    ),
    "bb-subset-cons10": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-bb-pcn-subset-cons10-global-1degree-24to3-v1/"
        "test-2023-2024-ens32.zarr"
    ),
    "5ch-flat": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-5ch-flat-global-1degree-24to3-v1/test-2023-2024-ens32.zarr"
    ),
    "5ch-kernel": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-5ch-per-channel-kernel-global-1degree-24to3-v1/"
        "test-2023-2024-ens32.zarr"
    ),
    "5ch-kernel-subset-cons0": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-5ch-per-channel-kernel-subset-cons0-global-1degree-24to3-v1/"
        "test-2023-2024-ens32.zarr"
    ),
}
TRUTH_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2025-07-25-X-SHiELD-AMIP-FME-3h.zarr"
)
# Stage-2 (spatiotemporal, 25km/100km two-stage PMD) models score against
# the 25km fine-resolution truth store instead of the 1-degree TRUTH_ZARR
# used by the stage-1 (temporal-only, 1-degree) models above.
FINE_TRUTH_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr"
)
# Stage-2 models' test inference was too memory-heavy to run as one global
# job (see configs/experiments/2026-07-24-video-pmd-spatiotemporal-25km-100km-test-inference/
# video_inference.yaml's header), so it was split across 4 disjoint regions
# that together tile the full modeled domain (lat -88..88, lon 0..360) with
# no gap/overlap: the mid-latitude band (-44..44) split into west/east lon
# halves, plus full-longitude north/south polar caps. This is NOT a regular
# lat x lon grid of tiles (the caps span the full lon range while the mid
# band is lon-split), so it can't be stitched with xr.combine_by_coords'
# automatic hypercube detection -- open_pred_zarr() below does the two-stage
# manual concat (lon-concat the mid band, then lat-concat with the caps)
# instead. See configs/experiments/2026-08-03-video-pmd-spatiotemporal-25km-100km-*
# for the 3 region-2/3/4 configs and their boundary-alignment verification.
PATCHED_MODELS = {
    # Single-stage coarse-endpoints (v2 of the single-stage architecture --
    # true LR-endpoints-in/HR-full-out, no stage-A network, replaces the
    # non-deployable single-stage-v1 design; see
    # configs/experiments/2026-08-07-video-pmd-spatiotemporal-25km-100km-singlestage-coarse-endpoints-{flat,ou}/).
    # Global patch-tiled inference via VideoPatchPredictor
    # (fme/downscaling/predictors/video_composite.py, divide_generation:
    # true) -- ONE contiguous global zarr, not the 4-region dict the other
    # PATCHED_MODELS entries use, so this is a plain str path (same as
    # KNOWN_MODELS' entries; _load_pred_window handles both).
    "st-singlestage-coarse-endpoints-flat": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat/"
        "test-2023-2024-ens4-global.zarr"
    ),
    "st-singlestage-coarse-endpoints-ou": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-ou/"
        "test-2023-2024-ens4-global.zarr"
    ),
    "st-flat": {
        "mid_west": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr"
        ),
        "mid_east": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat-44to44-lon180to360.zarr"
        ),
        "north_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat44to88-lon0to360.zarr"
        ),
        "south_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat-88to-44-lon0to360.zarr"
        ),
    },
    "st-ou": {
        "mid_west": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v1/"
            "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr"
        ),
        "mid_east": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v1/"
            "test-2023-2024-ens4-region-lat-44to44-lon180to360.zarr"
        ),
        "north_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v1/"
            "test-2023-2024-ens4-region-lat44to88-lon0to360.zarr"
        ),
        "south_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v1/"
            "test-2023-2024-ens4-region-lat-88to-44-lon0to360.zarr"
        ),
    },
    # v2 retrains of st-flat/st-ou after the endpoint-only-conditioning fix
    # (fme/downscaling/video_models.py, commit 7ec7d69a8 -- v1's coarse
    # conditioning leaked LR info into every interior frame instead of just
    # the two observed endpoints). CAVEAT: both checkpoints are epoch 41/200
    # (~20% of the planned schedule) -- training crashed on an ai2/titan
    # infra incident around 2026-08-09 23:00-2026-08-10 00:30 and inference
    # was run directly against the crashed run's latest.ckpt per explicit
    # instruction, not a resumed/completed run. See
    # configs/experiments/2026-08-10-video-pmd-spatiotemporal-25km-100km-{flat,ou}-v2-test-inference-*/
    # header comments. Treat as preliminary/undertrained, not the final v2
    # comparison.
    "st-flat-v2": {
        "mid_west": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v2/"
            "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr"
        ),
        "mid_east": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v2/"
            "test-2023-2024-ens4-region-lat-44to44-lon180to360.zarr"
        ),
        "north_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v2/"
            "test-2023-2024-ens4-region-lat44to88-lon0to360.zarr"
        ),
        "south_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v2/"
            "test-2023-2024-ens4-region-lat-88to-44-lon0to360.zarr"
        ),
    },
    "st-ou-v2": {
        "mid_west": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2/"
            "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr"
        ),
        "mid_east": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2/"
            "test-2023-2024-ens4-region-lat-44to44-lon180to360.zarr"
        ),
        "north_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2/"
            "test-2023-2024-ens4-region-lat44to88-lon0to360.zarr"
        ),
        "south_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v2/"
            "test-2023-2024-ens4-region-lat-88to-44-lon0to360.zarr"
        ),
    },
    # Single-stage (joint spatial+temporal in one UNet call, no separate
    # endpoint-SR network) counterpart to st-flat/st-ou -- see
    # configs/experiments/2026-07-31-video-pmd-spatiotemporal-25km-100km-single-stage/.
    "st-singlestage-flat": {
        "mid_west": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr"
        ),
        "mid_east": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat-44to44-lon180to360.zarr"
        ),
        "north_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat44to88-lon0to360.zarr"
        ),
        "south_cap": (
            "/climate-default/2026-06-25-temporal-diffusion/inference/"
            "global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/"
            "test-2023-2024-ens4-region-lat-88to-44-lon0to360.zarr"
        ),
    },
}
# Every KNOWN_MODELS/PATCHED_MODELS label resolves to a pred spec (str path
# or a PATCHED_MODELS region dict) via this combined lookup.
ALL_MODELS = {**KNOWN_MODELS, **PATCHED_MODELS}
CHANNELS_4CH = [
    "eastward_wind_at_ten_meters",
    "northward_wind_at_ten_meters",
    "PRMSL",
    "PRATEsfc",
]
# The 5ch-* models (and both stage-2 spatiotemporal models) add
# air_temperature_at_two_meters as a 5th output channel.
CHANNELS_5CH = CHANNELS_4CH + ["air_temperature_at_two_meters"]
MODEL_CHANNELS = {
    label: (CHANNELS_5CH if label.startswith("5ch-") else CHANNELS_4CH)
    for label in KNOWN_MODELS
}
MODEL_CHANNELS.update({label: CHANNELS_5CH for label in PATCHED_MODELS})
# Stage-2 models score against the fine (25km) truth; everything else
# against the original 1-degree TRUTH_ZARR.
MODEL_TRUTH = {label: TRUTH_ZARR for label in KNOWN_MODELS}
MODEL_TRUTH.update({label: FINE_TRUTH_ZARR for label in PATCHED_MODELS})
UNITS = {
    "eastward_wind_at_ten_meters": "m/s",
    "northward_wind_at_ten_meters": "m/s",
    "PRMSL": "mb",
    "PRATEsfc": "kg/m2/s",
    "air_temperature_at_two_meters": "K",
}
# One representative window per season (3 days each), all within the test
# period (2023-01-01 to 2024-01-04). Each window starts at a real clip
# boundary (00:00) and spans a whole number of days, so the lead-time-within-
# window logic below stays valid independently in each window.
SAMPLE_WINDOWS = [
    ("winter (Jan)", cftime.DatetimeJulian(2023, 1, 1), cftime.DatetimeJulian(2023, 1, 4)),
    ("spring (Apr)", cftime.DatetimeJulian(2023, 4, 1), cftime.DatetimeJulian(2023, 4, 4)),
    ("summer (Jul)", cftime.DatetimeJulian(2023, 7, 1), cftime.DatetimeJulian(2023, 7, 4)),
    ("fall (Oct)", cftime.DatetimeJulian(2023, 10, 1), cftime.DatetimeJulian(2023, 10, 4)),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Video PMD CRPS/spread-skill eval")
    parser.add_argument(
        "--model", choices=sorted(ALL_MODELS), default="pcn-v1",
        help="Which known inference output to evaluate. Stage-2 spatiotemporal "
             "models (st-flat, st-ou) are 4-way tiled and get combined into "
             "one global-coverage Dataset automatically.",
    )
    parser.add_argument(
        "--pred-zarr", default=None,
        help="Explicit path to an inference output zarr, overriding --model "
             "for one-off runs not yet in KNOWN_MODELS. Not usable for "
             "PATCHED_MODELS entries (st-flat/st-ou) since those need 4 paths.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(ALL_MODELS), default=None,
        help="Two or more known models to compare side by side in one run "
             "(overrides --model/--pred-zarr). E.g. "
             "--models pcn-v1 bb-subset-cons10.",
    )
    parser.add_argument(
        "--outdir", default=".", help="Where to save the PNG figures.",
    )
    parser.add_argument(
        "--inflate", action="store_true",
        help="Apply post-hoc ensemble spread inflation, per channel, "
             "calibrated from each model's own measured overall spread/skill "
             "ratio (factor = max(1, 1/ratio), so already-calibrated or "
             "overdispersive channels are left alone). Rescales each "
             "member's deviation from the ensemble mean -- does not change "
             "the ensemble mean or its RMSE, only spread/CRPS. This is a "
             "statistical correction on top of the existing ensemble, not a "
             "fix to the generative model itself; see COMPARISON_REPORT.md "
             "for why brownian-bridge noise underdisperses in the first "
             "place. Output labels/filenames get an '-inflated' suffix.",
    )
    return parser.parse_args()


ARGS = parse_args()
PRED_ZARR = ARGS.pred_zarr or ALL_MODELS[ARGS.model]
OUTDIR = ARGS.outdir
# Label used in plot titles/filenames so different models' outputs don't
# collide when writing to the same OUTDIR.
LABEL = ARGS.model if ARGS.pred_zarr is None else "custom"
COMPARE = ARGS.models is not None and len(ARGS.models) > 1

TIME_STEP_HOURS = 3
N_TIMESTEPS = 9  # clip length; matches config.model.n_timesteps
CLIP_STRIDE = N_TIMESTEPS - 1
LEAD_HOURS = [3, 6, 9, 12, 15, 18, 21]


def load_model_data(pred_spec, truth_raw):
    """No-op passthrough now -- see load_window for why pred is no longer
    opened/concatenated here. Kept so main()'s call sites don't change."""
    return pred_spec, truth_raw


def _load_pred_window(pred_spec, t0, t1):
    """Load one time window of prediction data, given either a single zarr
    path (str) or a 4-region dict (see PATCHED_MODELS).

    For the dict case, this opens+time-slices+``.load()``s each of the 4
    regions *separately* (small: one region x one ~24-timestep window is
    ~150-600MB) and only concatenates them into one global-coverage array
    *after* they're already in memory -- lon-concat the west/east
    mid-latitude halves into a full-longitude mid band, then lat-concat
    [south_cap, mid_band, north_cap]. Not xr.combine_by_coords -- the tiling
    (lon-split mid band + full-lon polar caps) isn't a regular lat x lon
    hypercube grid that combine_by_coords can auto-detect.

    This replaced an earlier version that built one lazy
    ``xr.concat([...])`` across all 4 regions ONCE and then repeatedly
    ``.sel(time=slice(...)).load()``d narrow windows from it. That seemed
    reasonable (each individual window is small) but reliably OOM'd at every
    memory budget tried (96/120/220/64/96GiB, both combined-model and
    split-per-model, with and without a separate truth-alignment fix) --
    each attempt died *faster* with *more* memory available, which is the
    signature of unbounded work scaling with wall-clock rather than a fixed
    large allocation. The likely cause: repeatedly slicing a *narrow* window
    out of a dask graph built from a *lazy 4-way concat spanning the full
    multi-year time axis* forced dask to reason about the whole graph's
    chunk structure on every call, and that overhead did not release
    cleanly between calls. Slicing each region's *own* zarr (a plain,
    un-concatenated dask array) by time first sidesteps that graph
    entirely -- only ever small, single-source, chunk-aligned reads.
    """
    if isinstance(pred_spec, dict):
        parts = {}
        for region, path in pred_spec.items():
            parts[region] = xr.open_zarr(path).sel(time=slice(t0, t1)).load()
        mid_band = xr.concat([parts["mid_west"], parts["mid_east"]], dim="longitude")
        return xr.concat([parts["south_cap"], mid_band, parts["north_cap"]], dim="latitude")
    return xr.open_zarr(pred_spec).sel(time=slice(t0, t1)).load()


def load_window(pred_spec, truth_raw, t0, t1):
    """(pred, truth, interior_mask, lead_hour_per_step) for one time window.

    Lead-hour-within-clip is computed *locally* to this window (position mod
    CLIP_STRIDE) -- valid as long as the window starts at a clip boundary
    (00:00) and spans whole days, which all SAMPLE_WINDOWS do.

    Truth's lat/lon nearest-neighbor alignment happens here too, against an
    already time-sliced small chunk -- same reasoning as the pred side, see
    _load_pred_window's docstring: reindexing against truth's full-year time
    axis (even lazily) was part of the same OOM pattern.
    """
    p = _load_pred_window(pred_spec, t0, t1)
    t = truth_raw.sel(time=slice(t0, t1)).sel(
        latitude=p.latitude, longitude=p.longitude, method="nearest"
    ).load()
    interior_mask = p["frame_source"].values == 1
    lead_hour = (np.arange(p.sizes["time"]) % CLIP_STRIDE) * TIME_STEP_HOURS
    return p, t, interior_mask, lead_hour


def inflate_ensemble(ens, factor, member_axis=-1):
    """Rescale each member's deviation from the ensemble mean by ``factor``
    (post-hoc spread inflation). Leaves the ensemble mean (and therefore its
    RMSE against truth) unchanged -- only spread and CRPS are affected."""
    if factor == 1.0:
        return ens
    mean = ens.mean(axis=member_axis, keepdims=True)
    return mean + factor * (ens - mean)


def compute_model_scores(pred_full, truth_full, channels, inflation_factors=None):
    """Run the full scoring pipeline for one model's already-opened pred
    dataset and raw (not yet lat/lon-aligned -- see load_window) truth
    dataset. Returns (summary_df, lead_df, crps_map, lat, lon).

    ``channels``: this model's output channel list (4ch or 5ch models differ).
    ``inflation_factors``: optional {channel: factor} to apply post-hoc
    spread inflation (see ``inflate_ensemble``) before scoring each channel.
    """
    windows = [
        (name, *load_window(pred_full, truth_full, t0, t1))
        for name, t0, t1 in SAMPLE_WINDOWS
    ]
    # lat/lon come from the first window's own pred data (pred_full is now
    # just the zarr path/region-dict spec, not a pre-opened Dataset -- see
    # load_window/_load_pred_window).
    lat = windows[0][1]["latitude"].values
    lon = windows[0][1]["longitude"].values
    area_weight = np.cos(np.radians(lat))  # (lat,), broadcasts against (..., lat, lon)

    for name, p, t, interior_mask, _ in windows:
        print(f"{name:14s} {p.sizes['time']:3d} timesteps, {int(interior_mask.sum()):2d} interior")

    # ---- Global per-channel scores (all interior frames, all windows) ----
    rows = []
    for name in channels:
        p_parts, t_parts = [], []
        for _, p_ds, t_ds, interior_mask, _ in windows:
            p_parts.append(p_ds[name].isel(time=interior_mask).transpose(
                "time", "latitude", "longitude", "ensemble").values)
            t_parts.append(t_ds[name].isel(time=interior_mask).transpose(
                "time", "latitude", "longitude").values)
        p = np.concatenate(p_parts, axis=0)
        t = np.concatenate(t_parts, axis=0)
        if inflation_factors:
            p = inflate_ensemble(p, inflation_factors.get(name, 1.0))

        crps_val = area_weighted_mean(crps_fair(p, t), area_weight, lat_axis=1)
        spread, rmse, ratio = spread_skill(p, t, area_weight, lat_axis=1)

        rows.append({
            "channel": name,
            "units": UNITS[name],
            "n_frames": p.shape[0],
            "CRPS": crps_val,
            "spread": spread,
            "MSE (ens mean)": rmse ** 2,
            "RMSE (ens mean)": rmse,
            "spread/skill ratio": ratio,
        })
        # Each channel's concatenated (time, lat, lon, ensemble) array is
        # ~1.4GB at the stage-2 25km global grid (16x the stage-1 1-degree
        # grid's footprint) -- explicit collect between channels rather than
        # relying on refcounting alone, since a run at this grid size OOM'd
        # after accumulating well past any single channel's own need.
        del p, t, p_parts, t_parts
        gc.collect()

    summary = pd.DataFrame(rows).set_index("channel")

    # ---- Skill vs. lead time within the 24h interpolation window ----
    lead_rows = []
    for name in channels:
        for lead in LEAD_HOURS:
            p_parts, t_parts = [], []
            for _, p_ds, t_ds, interior_mask, lead_hour_per_step in windows:
                sel = interior_mask & (lead_hour_per_step == lead)
                if not sel.any():
                    continue
                p_parts.append(p_ds[name].isel(time=sel).transpose(
                    "time", "latitude", "longitude", "ensemble").values)
                t_parts.append(t_ds[name].isel(time=sel).transpose(
                    "time", "latitude", "longitude").values)
            p = np.concatenate(p_parts, axis=0)
            t = np.concatenate(t_parts, axis=0)
            if inflation_factors:
                p = inflate_ensemble(p, inflation_factors.get(name, 1.0))
            crps_val = area_weighted_mean(crps_fair(p, t), area_weight, lat_axis=1)
            spread, rmse, ratio = spread_skill(p, t, area_weight, lat_axis=1)
            lead_rows.append({
                "channel": name, "lead_hour": lead, "n_frames": p.shape[0],
                "CRPS": crps_val, "spread": spread, "RMSE": rmse, "ratio": ratio,
            })
            del p, t, p_parts, t_parts
        gc.collect()

    lead_df = pd.DataFrame(lead_rows)

    # ---- Spatial map: CRPS at the hardest lead time (12h), PRMSL ----
    name = "PRMSL"
    p_parts, t_parts = [], []
    for _, p_ds, t_ds, interior_mask, lead_hour_per_step in windows:
        sel = interior_mask & (lead_hour_per_step == 12)
        p_parts.append(p_ds[name].isel(time=sel).transpose(
            "time", "latitude", "longitude", "ensemble").values)
        t_parts.append(t_ds[name].isel(time=sel).transpose(
            "time", "latitude", "longitude").values)
    p = np.concatenate(p_parts, axis=0)
    t = np.concatenate(t_parts, axis=0)
    if inflation_factors:
        p = inflate_ensemble(p, inflation_factors.get(name, 1.0))
    crps_map = crps_fair(p, t).mean(axis=0)  # (lat, lon)

    return summary, lead_df, crps_map, lat, lon


def plot_single_model(label, summary, lead_df, crps_map, lat, lon):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in summary.index:
        sub = lead_df[lead_df["channel"] == name]
        axes[0].plot(sub["lead_hour"], sub["CRPS"], marker="o", label=name)
        axes[1].plot(sub["lead_hour"], sub["ratio"], marker="o", label=name)
    axes[0].set_title("CRPS vs. lead time")
    axes[0].set_ylabel("CRPS (native units)")
    axes[1].set_title("Spread/skill ratio vs. lead time")
    axes[1].set_ylabel("ratio (1.0 = reliable)")
    axes[1].axhline(1.0, color="gray", lw=0.8, ls="--")
    for ax in axes:
        ax.set_xlabel("lead time within 24h window (hr)")
        ax.set_xticks(LEAD_HOURS)
        ax.axvline(12, color="gray", lw=0.6, alpha=0.5)
        ax.legend(fontsize=7)
    fig.suptitle(f"{label}: skill by lead time (4 seasonal windows, 2023)")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/crps_lead_time_{label}.png", dpi=150)
    plt.close(fig)

    name = "PRMSL"
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.pcolormesh(lon, lat, crps_map, cmap="viridis")
    fig.colorbar(im, ax=ax, label=f"CRPS ({UNITS[name]})")
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude")
    ax.set_title(f"{label}: {name} CRPS at 12h lead (hardest interior frame), 4-season mean")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/crps_map_{label}.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved {OUTDIR}/crps_lead_time_{label}.png and {OUTDIR}/crps_map_{label}.png")


def plot_comparison(results):
    """results: dict label -> (summary_df, lead_df, crps_map, lat, lon)."""
    labels = list(results)
    tag = "-".join(labels)
    linestyles = ["-", "--", ":", "-."]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    present = {name for label in labels for name in results[label][0].index}
    all_channels = [name for name in CHANNELS_5CH if name in present]
    channel_color = {name: colors[i % len(colors)] for i, name in enumerate(all_channels)}

    # ---- Combined lead-time plot: color = channel, linestyle = model ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for li, label in enumerate(labels):
        summary, lead_df, _, _, _ = results[label]
        ls = linestyles[li % len(linestyles)]
        for name in summary.index:
            sub = lead_df[lead_df["channel"] == name]
            leg = f"{name} ({label})"
            axes[0].plot(sub["lead_hour"], sub["CRPS"], marker="o", ls=ls,
                         color=channel_color[name], label=leg)
            axes[1].plot(sub["lead_hour"], sub["ratio"], marker="o", ls=ls,
                         color=channel_color[name], label=leg)
    axes[0].set_title("CRPS vs. lead time")
    axes[0].set_ylabel("CRPS (native units)")
    axes[1].set_title("Spread/skill ratio vs. lead time")
    axes[1].set_ylabel("ratio (1.0 = reliable)")
    axes[1].axhline(1.0, color="gray", lw=0.8, ls="--")
    for ax in axes:
        ax.set_xlabel("lead time within 24h window (hr)")
        ax.set_xticks(LEAD_HOURS)
        ax.axvline(12, color="gray", lw=0.6, alpha=0.5)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"Model comparison ({' vs. '.join(labels)}): skill by lead time")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/crps_lead_time_compare_{tag}.png", dpi=150)
    plt.close(fig)

    # ---- Combined spatial map: one panel per model, shared color scale ----
    name = "PRMSL"
    maps = {label: results[label][2] for label in labels}
    vmin = min(m.min() for m in maps.values())
    vmax = max(m.max() for m in maps.values())
    fig, axes = plt.subplots(1, len(labels), figsize=(9 * len(labels), 4.2), squeeze=False)
    axes = axes[0]
    im = None
    for ax, label in zip(axes, labels):
        _, _, crps_map, lat, lon = results[label]
        im = ax.pcolormesh(lon, lat, crps_map, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("longitude (deg E)")
        ax.set_ylabel("latitude")
        ax.set_title(label)
    fig.colorbar(im, ax=list(axes), label=f"CRPS ({UNITS[name]})")
    fig.suptitle(f"{name} CRPS at 12h lead (hardest interior frame), 4-season mean")
    fig.savefig(f"{OUTDIR}/crps_map_compare_{tag}.png", dpi=150)
    plt.close(fig)

    # ---- Combined report: side-by-side summary table ----
    combined = pd.concat(
        {label: results[label][0] for label in labels}, names=["model"]
    ).reset_index()
    combined = combined[["model", "channel", "units", "n_frames", "CRPS",
                          "spread", "MSE (ens mean)", "RMSE (ens mean)",
                          "spread/skill ratio"]]
    print(f"\n=== Combined comparison ({' vs. '.join(labels)}) ===")
    print(combined.set_index(["channel", "model"]).sort_index())
    csv_path = f"{OUTDIR}/comparison_summary_{tag}.csv"
    combined.to_csv(csv_path, index=False)

    print(
        f"\nSaved {OUTDIR}/crps_lead_time_compare_{tag}.png, "
        f"{OUTDIR}/crps_map_compare_{tag}.png, and {csv_path}"
    )


def main():
    if COMPARE:
        labels = ARGS.models
        paths = {label: ALL_MODELS[label] for label in labels}
    else:
        labels = [LABEL]
        paths = {LABEL: PRED_ZARR}

    # Cache opened truth datasets by path -- stage-1 models share TRUTH_ZARR,
    # stage-2 models share FINE_TRUTH_ZARR, so a comparison run rarely needs
    # more than one of each open at a time.
    truth_cache = {}

    results = {}
    for label in labels:
        print(f"\n=== {label} ===")
        print(f"Pred zarr: {paths[label]}")
        channels = MODEL_CHANNELS.get(label, CHANNELS_4CH)
        truth_zarr = MODEL_TRUTH.get(label, TRUTH_ZARR)
        if truth_zarr not in truth_cache:
            truth_cache[truth_zarr] = xr.open_zarr(truth_zarr)
        truth_raw = truth_cache[truth_zarr]
        pred_full, truth_full = load_model_data(paths[label], truth_raw)
        summary, lead_df, crps_map, lat, lon = compute_model_scores(
            pred_full, truth_full, channels
        )
        print("\nOverall (all interior frames, 4 seasonal windows):")
        print(summary)
        print("\nBy lead time (pooled across 4 seasonal windows):")
        print(lead_df.round(4).set_index(["channel", "lead_hour"]))

        display_label = label
        if ARGS.inflate:
            inflation_factors = {
                ch: max(1.0, 1.0 / summary.loc[ch, "spread/skill ratio"])
                for ch in channels
            }
            print(f"\nSpread inflation factors (from this model's own "
                  f"measured ratio, capped at >=1): {inflation_factors}")
            summary, lead_df, crps_map, lat, lon = compute_model_scores(
                pred_full, truth_full, channels, inflation_factors=inflation_factors
            )
            print("\n[inflated] Overall (all interior frames, 4 seasonal windows):")
            print(summary)
            print("\n[inflated] By lead time (pooled across 4 seasonal windows):")
            print(lead_df.round(4).set_index(["channel", "lead_hour"]))
            display_label = f"{label}-inflated"

        results[display_label] = (summary, lead_df, crps_map, lat, lon)

        # For the stage-2 (25km, tiled) models pred_full/truth_full are ~16x
        # the pixel count of the stage-1 1-degree grid, and each label's
        # open_pred_zarr() call opens 4 fresh zarr stores + 2 concats --
        # without an explicit collect the dask/zarr chunk-cache references
        # accumulate across labels in a --models comparison run rather than
        # being freed between iterations (confirmed: a 120GiB run OOM'd
        # after ~73 minutes, well past what any single label's data
        # should need on its own).
        del pred_full, truth_full
        gc.collect()

    if COMPARE:
        plot_comparison(results)
    else:
        display_label = list(results)[0]
        summary, lead_df, crps_map, lat, lon = results[display_label]
        plot_single_model(display_label, summary, lead_df, crps_map, lat, lon)


def crps_fair(ens, truth_arr):
    """Fair (finite-ensemble-unbiased) CRPS. ``ens``: (..., M); ``truth_arr``:
    (...) broadcastable against ``ens[..., 0]``. Returns (...)."""
    M = ens.shape[-1]
    sorted_ens = np.sort(ens, axis=-1)
    k = np.arange(1, M + 1)
    weighted_sum = np.tensordot(sorted_ens, (2 * k - M - 1), axes=([-1], [0]))
    term2 = weighted_sum / (M * (M - 1))
    term1 = np.abs(ens - truth_arr[..., None]).mean(axis=-1)
    return term1 - term2


def area_weighted_mean(arr, area_weight, lat_axis):
    """Weight by cos(lat) along ``lat_axis`` of an otherwise-arbitrary array."""
    shape = [1] * arr.ndim
    shape[lat_axis] = len(area_weight)
    w = area_weight.reshape(shape)
    w = np.broadcast_to(w, arr.shape)
    return np.sum(arr * w) / np.sum(w)


def spread_skill(ens, truth_arr, area_weight, lat_axis, member_axis=-1):
    """(spread, rmse, ratio), area-weighted along ``lat_axis``, with the
    Fortin et al. (2014) finite-ensemble correction on spread
    (``sqrt((M+1)/M)``) so it's directly comparable to ensemble-mean RMSE for
    a reliable ensemble (ratio approx 1)."""
    M = ens.shape[member_axis]
    ens_mean = ens.mean(axis=member_axis)
    rmse = np.sqrt(area_weighted_mean((ens_mean - truth_arr) ** 2, area_weight, lat_axis))
    var = ens.var(axis=member_axis, ddof=1)
    spread = np.sqrt(area_weighted_mean(var, area_weight, lat_axis)) * np.sqrt((M + 1) / M)
    return spread, rmse, spread / rmse


if __name__ == "__main__":
    main()
