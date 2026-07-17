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
}
TRUTH_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2025-07-25-X-SHiELD-AMIP-FME-3h.zarr"
)
CHANNELS = [
    "eastward_wind_at_ten_meters",
    "northward_wind_at_ten_meters",
    "PRMSL",
    "PRATEsfc",
]
UNITS = {
    "eastward_wind_at_ten_meters": "m/s",
    "northward_wind_at_ten_meters": "m/s",
    "PRMSL": "mb",
    "PRATEsfc": "kg/m2/s",
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
        "--model", choices=sorted(KNOWN_MODELS), default="pcn-v1",
        help="Which known inference output to evaluate.",
    )
    parser.add_argument(
        "--pred-zarr", default=None,
        help="Explicit path to an inference output zarr, overriding --model "
             "for one-off runs not yet in KNOWN_MODELS.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(KNOWN_MODELS), default=None,
        help="Two or more known models to compare side by side in one run "
             "(overrides --model/--pred-zarr). E.g. "
             "--models pcn-v1 bb-subset-cons10.",
    )
    parser.add_argument(
        "--outdir", default=".", help="Where to save the PNG figures.",
    )
    return parser.parse_args()


ARGS = parse_args()
PRED_ZARR = ARGS.pred_zarr or KNOWN_MODELS[ARGS.model]
OUTDIR = ARGS.outdir
# Label used in plot titles/filenames so different models' outputs don't
# collide when writing to the same OUTDIR.
LABEL = ARGS.model if ARGS.pred_zarr is None else "custom"
COMPARE = ARGS.models is not None and len(ARGS.models) > 1

TIME_STEP_HOURS = 3
N_TIMESTEPS = 9  # clip length; matches config.model.n_timesteps
CLIP_STRIDE = N_TIMESTEPS - 1
LEAD_HOURS = [3, 6, 9, 12, 15, 18, 21]


def load_model_data(pred_zarr, truth_raw):
    pred_full = xr.open_zarr(pred_zarr)
    # Predictions are on a lat_extent-cropped grid (176 of 180 raw latitudes);
    # align truth to the exact same grid before comparing.
    truth_full = truth_raw.sel(
        latitude=pred_full.latitude, longitude=pred_full.longitude, method="nearest"
    )
    return pred_full, truth_full


def load_window(pred_full, truth_full, t0, t1):
    """(pred, truth, interior_mask, lead_hour_per_step) for one time window.

    Lead-hour-within-clip is computed *locally* to this window (position mod
    CLIP_STRIDE) -- valid as long as the window starts at a clip boundary
    (00:00) and spans whole days, which all SAMPLE_WINDOWS do.
    """
    p = pred_full.sel(time=slice(t0, t1))
    t = truth_full.sel(time=slice(t0, t1))
    interior_mask = p["frame_source"].values == 1
    lead_hour = (np.arange(p.sizes["time"]) % CLIP_STRIDE) * TIME_STEP_HOURS
    return p, t, interior_mask, lead_hour


def compute_model_scores(pred_full, truth_full):
    """Run the full scoring pipeline for one model's already-opened,
    truth-aligned datasets. Returns (summary_df, lead_df, crps_map, lat, lon)."""
    lat = pred_full["latitude"].values
    lon = pred_full["longitude"].values
    area_weight = np.cos(np.radians(lat))  # (lat,), broadcasts against (..., lat, lon)

    windows = [
        (name, *load_window(pred_full, truth_full, t0, t1))
        for name, t0, t1 in SAMPLE_WINDOWS
    ]
    for name, p, t, interior_mask, _ in windows:
        print(f"{name:14s} {p.sizes['time']:3d} timesteps, {int(interior_mask.sum()):2d} interior")

    # ---- Global per-channel scores (all interior frames, all windows) ----
    rows = []
    for name in CHANNELS:
        p_parts, t_parts = [], []
        for _, p_ds, t_ds, interior_mask, _ in windows:
            p_parts.append(p_ds[name].isel(time=interior_mask).transpose(
                "time", "latitude", "longitude", "ensemble").values)
            t_parts.append(t_ds[name].isel(time=interior_mask).transpose(
                "time", "latitude", "longitude").values)
        p = np.concatenate(p_parts, axis=0)
        t = np.concatenate(t_parts, axis=0)

        crps_val = area_weighted_mean(crps_fair(p, t), area_weight, lat_axis=1)
        spread, rmse, ratio = spread_skill(p, t, area_weight, lat_axis=1)

        rows.append({
            "channel": name,
            "units": UNITS[name],
            "n_frames": p.shape[0],
            "CRPS": crps_val,
            "spread": spread,
            "RMSE (ens mean)": rmse,
            "spread/skill ratio": ratio,
        })

    summary = pd.DataFrame(rows).set_index("channel")

    # ---- Skill vs. lead time within the 24h interpolation window ----
    lead_rows = []
    for name in CHANNELS:
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
            crps_val = area_weighted_mean(crps_fair(p, t), area_weight, lat_axis=1)
            spread, rmse, ratio = spread_skill(p, t, area_weight, lat_axis=1)
            lead_rows.append({
                "channel": name, "lead_hour": lead, "n_frames": p.shape[0],
                "CRPS": crps_val, "spread": spread, "RMSE": rmse, "ratio": ratio,
            })

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
    crps_map = crps_fair(p, t).mean(axis=0)  # (lat, lon)

    return summary, lead_df, crps_map, lat, lon


def plot_single_model(label, summary, lead_df, crps_map, lat, lon):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in CHANNELS:
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
    channel_color = {name: colors[i % len(colors)] for i, name in enumerate(CHANNELS)}

    # ---- Combined lead-time plot: color = channel, linestyle = model ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for li, label in enumerate(labels):
        _, lead_df, _, _, _ = results[label]
        ls = linestyles[li % len(linestyles)]
        for name in CHANNELS:
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
                          "spread", "RMSE (ens mean)", "spread/skill ratio"]]
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
        paths = {label: KNOWN_MODELS[label] for label in labels}
    else:
        labels = [LABEL]
        paths = {LABEL: PRED_ZARR}

    truth_raw = xr.open_zarr(TRUTH_ZARR)

    results = {}
    for label in labels:
        print(f"\n=== {label} ===")
        print(f"Pred zarr: {paths[label]}")
        pred_full, truth_full = load_model_data(paths[label], truth_raw)
        summary, lead_df, crps_map, lat, lon = compute_model_scores(pred_full, truth_full)
        print("\nOverall (all interior frames, 4 seasonal windows):")
        print(summary)
        print("\nBy lead time (pooled across 4 seasonal windows):")
        print(lead_df.round(4).set_index(["channel", "lead_hour"]))
        results[label] = (summary, lead_df, crps_map, lat, lon)

    if COMPARE:
        plot_comparison(results)
    else:
        summary, lead_df, crps_map, lat, lon = results[labels[0]]
        plot_single_model(labels[0], summary, lead_df, crps_map, lat, lon)


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
