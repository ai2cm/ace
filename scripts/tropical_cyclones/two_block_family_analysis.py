"""Two-Block vs the Joint-Generation family: TC min-pressure lifecycle,
mid-latitude cyclone trajectory, and zoomed animations. Companion to the
crps_eval / diurnal_cycle_eval jobs for the two-block comparison doc.

Models (all coarse-endpoints-only, 25 km fine output, member 0):
  25 km truth
  Cascaded Generation            (cascade-infill-then-sr)
  Two Block                      (two-block-flat: pinned r + unpinned d)
  Joint Gen - flat               (ce-flat)
  Joint Gen w/o Temp Attn        (ce-flat-nta)

Run as a Beaker session with weka climate-default mounted.
"""
import datetime
import json
import re

import cftime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

BASE = "/climate-default/2026-06-25-temporal-diffusion/"
INF = BASE + "inference/"
MODELS = {
    "25 km truth": (BASE + "2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr", None),
    "Cascaded Generation": (
        INF + "hiro-downscaling-25km-100km-global-5ch-v6-cascade-infill-then-sr/"
        "test-2023-2024-ens4.zarr", 0),
    "Two Block": (
        INF + "video-pmd-spatiotemporal-25km-100km-global-5ch-two-block-"
        "coarse-endpoints-flat/test-2023-2024-ens4-global.zarr", 0),
    "Joint Gen - flat": (
        INF + "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-"
        "coarse-endpoints-flat/test-2023-2024-ens4-global.zarr", 0),
    "Joint Gen w/o Temp Attn": (
        INF + "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-"
        "coarse-endpoints-flat-no-temporal-attn/test-2023-2024-ens4-global.zarr", 0),
}
COLORS = {
    "25 km truth": "black",
    "Cascaded Generation": "#e377c2",
    "Two Block": "#2ca02c",
    "Joint Gen - flat": "#d62728",
    "Joint Gen w/o Temp Attn": "#8c564b",
}
GEN = [k for k in MODELS if k != "25 km truth"]
WINDOW_DEG = 2.5     # half-width -> 5x5 deg box (repo TC-verification convention)
FPS = 5

T789 = [
    ('2023-05-24 06:00:00', 9.1, 145.125), ('2023-05-24 09:00:00', 9.1, 145.375),
    ('2023-05-24 12:00:00', 9.349, 145.125), ('2023-05-24 15:00:00', 9.349, 145.125),
    ('2023-05-24 18:00:00', 9.349, 144.875), ('2023-05-24 21:00:00', 9.598, 144.875),
    ('2023-05-25 00:00:00', 9.598, 144.875), ('2023-05-25 03:00:00', 9.848, 144.875),
    ('2023-05-25 06:00:00', 9.848, 144.875), ('2023-05-25 09:00:00', 9.848, 144.875),
    ('2023-05-25 12:00:00', 10.097, 144.875), ('2023-05-25 15:00:00', 10.097, 144.875),
    ('2023-05-25 18:00:00', 10.346, 144.875), ('2023-05-25 21:00:00', 10.596, 144.875),
    ('2023-05-26 00:00:00', 10.845, 144.625), ('2023-05-26 03:00:00', 10.845, 144.375),
    ('2023-05-26 06:00:00', 11.094, 144.375), ('2023-05-26 09:00:00', 11.094, 144.375),
    ('2023-05-26 12:00:00', 11.343, 144.375), ('2023-05-26 15:00:00', 11.343, 144.375),
    ('2023-05-26 18:00:00', 11.593, 144.375), ('2023-05-26 21:00:00', 11.842, 144.625),
    ('2023-05-27 00:00:00', 12.091, 144.625), ('2023-05-27 03:00:00', 12.341, 144.625),
    ('2023-05-27 06:00:00', 12.59, 144.625), ('2023-05-27 09:00:00', 12.839, 144.875),
    ('2023-05-27 12:00:00', 13.338, 144.875), ('2023-05-27 15:00:00', 13.837, 144.875),
    ('2023-05-27 18:00:00', 14.086, 144.875), ('2023-05-27 21:00:00', 14.584, 144.625),
    ('2023-05-28 00:00:00', 14.834, 144.625), ('2023-05-28 03:00:00', 15.332, 144.625),
    ('2023-05-28 06:00:00', 15.582, 144.625), ('2023-05-28 09:00:00', 15.831, 144.625),
    ('2023-05-28 12:00:00', 16.08, 144.625), ('2023-05-28 15:00:00', 16.33, 144.375),
    ('2023-05-28 18:00:00', 16.579, 144.375), ('2023-05-28 21:00:00', 16.828, 144.625),
    ('2023-05-29 00:00:00', 17.077, 144.625), ('2023-05-29 03:00:00', 17.327, 144.875),
    ('2023-05-29 06:00:00', 17.576, 145.125), ('2023-05-29 09:00:00', 18.075, 145.375),
    ('2023-05-29 12:00:00', 18.324, 145.625), ('2023-05-29 15:00:00', 18.573, 146.125),
    ('2023-05-29 18:00:00', 18.823, 146.375), ('2023-05-29 21:00:00', 19.321, 146.875),
    ('2023-05-30 00:00:00', 19.82, 147.375), ('2023-05-30 03:00:00', 20.568, 147.875),
    ('2023-05-30 06:00:00', 21.066, 148.125), ('2023-05-30 09:00:00', 21.814, 148.625),
    ('2023-05-30 12:00:00', 22.812, 148.875), ('2023-05-30 15:00:00', 23.559, 149.125),
    ('2023-05-30 18:00:00', 24.557, 149.375), ('2023-05-30 21:00:00', 25.803, 149.625),
    ('2023-05-31 00:00:00', 27.05, 149.375), ('2023-05-31 03:00:00', 28.546, 149.125),
    ('2023-05-31 06:00:00', 29.543, 148.625), ('2023-05-31 09:00:00', 30.54, 148.125),
    ('2023-05-31 12:00:00', 31.288, 147.625), ('2023-05-31 15:00:00', 32.036, 147.125),
    ('2023-05-31 18:00:00', 32.784, 146.625), ('2023-05-31 21:00:00', 33.532, 145.875),
    ('2023-06-01 00:00:00', 34.03, 145.375),
]
T795 = [
    ('2023-06-30 06:00:00', 11.842, 149.125), ('2023-06-30 09:00:00', 11.842, 148.875),
    ('2023-06-30 12:00:00', 11.842, 148.375), ('2023-06-30 15:00:00', 12.091, 148.375),
    ('2023-06-30 18:00:00', 12.341, 148.125), ('2023-06-30 21:00:00', 12.59, 147.875),
    ('2023-07-01 00:00:00', 13.089, 147.375), ('2023-07-01 03:00:00', 13.338, 146.875),
    ('2023-07-01 06:00:00', 13.587, 146.375), ('2023-07-01 09:00:00', 13.587, 145.875),
    ('2023-07-01 12:00:00', 13.837, 145.625), ('2023-07-01 15:00:00', 14.086, 145.125),
    ('2023-07-01 18:00:00', 14.584, 144.625), ('2023-07-01 21:00:00', 14.834, 144.125),
    ('2023-07-02 00:00:00', 15.083, 143.625), ('2023-07-02 03:00:00', 15.582, 143.125),
    ('2023-07-02 06:00:00', 15.831, 142.625), ('2023-07-02 09:00:00', 16.08, 141.875),
    ('2023-07-02 12:00:00', 16.08, 141.375), ('2023-07-02 15:00:00', 16.33, 141.125),
    ('2023-07-02 18:00:00', 16.828, 140.375), ('2023-07-02 21:00:00', 16.828, 139.875),
    ('2023-07-03 00:00:00', 16.828, 139.375), ('2023-07-03 03:00:00', 17.327, 139.375),
    ('2023-07-03 06:00:00', 17.327, 138.625), ('2023-07-03 09:00:00', 17.576, 138.375),
    ('2023-07-03 12:00:00', 18.075, 137.875), ('2023-07-03 15:00:00', 18.075, 137.125),
    ('2023-07-03 18:00:00', 18.075, 136.625), ('2023-07-03 21:00:00', 18.075, 136.375),
    ('2023-07-04 00:00:00', 18.573, 135.625), ('2023-07-04 03:00:00', 18.573, 135.375),
    ('2023-07-04 06:00:00', 18.823, 134.625), ('2023-07-04 09:00:00', 18.823, 134.125),
    ('2023-07-04 12:00:00', 18.573, 133.625), ('2023-07-04 15:00:00', 18.823, 133.125),
    ('2023-07-04 18:00:00', 18.823, 132.375), ('2023-07-04 21:00:00', 19.072, 132.125),
    ('2023-07-05 00:00:00', 19.072, 131.625), ('2023-07-05 03:00:00', 19.072, 130.875),
    ('2023-07-05 06:00:00', 19.321, 130.625), ('2023-07-05 09:00:00', 19.321, 130.125),
    ('2023-07-05 12:00:00', 19.571, 129.625), ('2023-07-05 15:00:00', 19.82, 129.125),
    ('2023-07-05 18:00:00', 19.82, 128.625), ('2023-07-05 21:00:00', 20.069, 128.375),
    ('2023-07-06 00:00:00', 20.318, 127.875), ('2023-07-06 03:00:00', 20.318, 127.375),
    ('2023-07-06 06:00:00', 20.568, 127.125), ('2023-07-06 09:00:00', 21.066, 126.625),
    ('2023-07-06 12:00:00', 21.066, 126.125), ('2023-07-06 15:00:00', 21.316, 125.875),
    ('2023-07-06 18:00:00', 21.565, 125.625), ('2023-07-06 21:00:00', 21.565, 125.125),
    ('2023-07-07 00:00:00', 21.814, 125.125), ('2023-07-07 03:00:00', 22.313, 124.625),
    ('2023-07-07 06:00:00', 22.313, 124.375), ('2023-07-07 09:00:00', 22.562, 124.125),
    ('2023-07-07 12:00:00', 22.812, 123.875), ('2023-07-07 15:00:00', 23.31, 123.375),
    ('2023-07-07 18:00:00', 23.559, 123.125), ('2023-07-07 21:00:00', 24.058, 122.875),
    ('2023-07-08 00:00:00', 24.307, 122.125), ('2023-07-08 03:00:00', 23.809, 121.625),
    ('2023-07-08 06:00:00', 24.058, 121.875), ('2023-07-08 09:00:00', 24.806, 121.125),
    ('2023-07-08 12:00:00', 24.806, 120.375), ('2023-07-08 15:00:00', 24.557, 120.125),
    ('2023-07-08 18:00:00', 24.806, 120.125), ('2023-07-08 21:00:00', 25.055, 119.875),
    ('2023-07-09 00:00:00', 25.305, 119.625), ('2023-07-09 03:00:00', 25.554, 119.625),
    ('2023-07-09 06:00:00', 25.803, 119.375), ('2023-07-09 18:00:00', 27.798, 116.625),
]
T829 = [
    ('2023-11-03 12:00:00', 9.349, 136.625), ('2023-11-03 15:00:00', 9.349, 136.875),
    ('2023-11-03 18:00:00', 9.598, 137.125), ('2023-11-03 21:00:00', 9.598, 137.375),
    ('2023-11-04 00:00:00', 9.598, 137.625), ('2023-11-04 03:00:00', 9.848, 138.125),
    ('2023-11-04 06:00:00', 10.097, 138.625), ('2023-11-04 09:00:00', 10.097, 139.125),
    ('2023-11-04 12:00:00', 10.596, 139.625), ('2023-11-04 15:00:00', 11.094, 139.625),
    ('2023-11-04 18:00:00', 11.094, 139.375), ('2023-11-04 21:00:00', 11.593, 139.625),
    ('2023-11-05 00:00:00', 11.593, 139.125), ('2023-11-05 03:00:00', 11.593, 138.875),
    ('2023-11-05 06:00:00', 11.593, 138.875), ('2023-11-05 09:00:00', 11.842, 139.125),
    ('2023-11-05 12:00:00', 12.091, 138.875), ('2023-11-05 15:00:00', 12.341, 138.875),
    ('2023-11-05 18:00:00', 12.341, 138.375), ('2023-11-05 21:00:00', 12.839, 138.375),
    ('2023-11-06 00:00:00', 13.089, 137.625), ('2023-11-06 03:00:00', 13.089, 137.125),
    ('2023-11-06 06:00:00', 13.587, 137.125), ('2023-11-06 09:00:00', 14.086, 136.625),
    ('2023-11-06 12:00:00', 14.335, 135.625), ('2023-11-06 15:00:00', 14.335, 135.375),
    ('2023-11-06 18:00:00', 14.834, 134.625), ('2023-11-06 21:00:00', 15.083, 133.875),
    ('2023-11-07 00:00:00', 15.332, 133.125), ('2023-11-07 03:00:00', 15.332, 132.625),
    ('2023-11-07 06:00:00', 15.582, 131.875), ('2023-11-07 09:00:00', 15.582, 131.375),
    ('2023-11-07 12:00:00', 15.582, 130.625), ('2023-11-07 15:00:00', 15.332, 130.125),
    ('2023-11-07 18:00:00', 15.332, 129.625), ('2023-11-07 21:00:00', 15.332, 129.125),
    ('2023-11-08 00:00:00', 15.083, 128.625), ('2023-11-08 03:00:00', 15.083, 128.375),
    ('2023-11-08 06:00:00', 15.083, 128.375), ('2023-11-08 09:00:00', 15.332, 128.375),
    ('2023-11-08 12:00:00', 15.332, 128.125), ('2023-11-08 15:00:00', 15.582, 128.125),
    ('2023-11-08 18:00:00', 15.831, 127.875), ('2023-11-08 21:00:00', 16.08, 127.625),
    ('2023-11-09 00:00:00', 16.33, 127.375), ('2023-11-09 03:00:00', 16.33, 127.125),
    ('2023-11-09 06:00:00', 16.579, 126.875), ('2023-11-09 09:00:00', 16.579, 127.125),
    ('2023-11-09 12:00:00', 16.579, 126.875), ('2023-11-09 15:00:00', 16.579, 126.625),
    ('2023-11-09 18:00:00', 16.828, 126.625), ('2023-11-09 21:00:00', 16.828, 126.625),
    ('2023-11-10 00:00:00', 16.828, 126.375), ('2023-11-10 03:00:00', 16.828, 126.375),
    ('2023-11-10 06:00:00', 17.077, 126.375), ('2023-11-10 09:00:00', 17.327, 126.375),
    ('2023-11-10 12:00:00', 17.327, 126.375), ('2023-11-10 15:00:00', 17.327, 126.125),
    ('2023-11-10 18:00:00', 17.576, 126.125), ('2023-11-10 21:00:00', 17.825, 125.875),
    ('2023-11-11 00:00:00', 18.075, 125.625), ('2023-11-11 03:00:00', 18.075, 125.625),
    ('2023-11-11 06:00:00', 18.324, 125.375), ('2023-11-11 09:00:00', 18.573, 125.125),
    ('2023-11-11 12:00:00', 18.573, 124.875),
]
TC_TRACKS = {789: ("intense", T789), 795: ("moderate", T795), 829: ("mild", T829)}

# mid-latitude cyclone auto-detect basins (winter 2023) -- same as
# etc_trajectory_cascade_vs_baseline.py
ETC_BASINS = [
    ("N-Pacific", "2023-01-05", "2023-02-20", (36, 60), (150, 235)),
    ("N-Atlantic", "2023-01-05", "2023-02-20", (38, 62), (295, 355)),
]
ETC_WIN_TRACK = 6.0
ETC_MAX_STEP = 4.0
ETC_WIN_MIN = 3.0
ETC_FILL_MB = 1004.0
ETC_MAX_HOURS = 132


def parse_ct(s):
    m = re.match(r"(\d+)-(\d+)-(\d+)[ T](\d+):(\d+):(\d+)", s)
    return cftime.DatetimeJulian(*(int(g) for g in m.groups()))


def to_mb(v):
    v = np.asarray(v, dtype="float64")
    return v / 100.0 if (v.size and np.nanmean(v) > 2000) else v


def prmsl_da(path, ens):
    da = xr.open_zarr(path)["PRMSL"]
    if ens is not None and "ensemble" in da.dims:
        da = da.isel(ensemble=ens)
    return da


def win_min(da, t, lat0, lon0, half):
    sub = da.sel(time=t, method="nearest").sel(
        latitude=slice(lat0 - half, lat0 + half),
        longitude=slice(lon0 - half, lon0 + half))
    v = to_mb(sub.values)
    return float(np.nanmin(v)) if np.isfinite(v).any() else np.nan


def crop(vals, lat, lon, lat0, lon0, half):
    ilat = np.where((lat >= lat0 - half) & (lat <= lat0 + half))[0]
    ilon = np.where((lon >= lon0 - half) & (lon <= lon0 + half))[0]
    return vals[np.ix_(ilat, ilon)], lat[ilat], lon[ilon]


# ============================================================ A: TC lifecycle
def tc_lifecycle(fields):
    print("=== A: TC min-pressure lifecycle ===", flush=True)
    summary = {}
    for tid, (label, pts) in TC_TRACKS.items():
        times = [p[0] for p in pts]
        is00 = np.array([s.endswith("00:00:00") for s in times])
        rec = {n: [] for n in MODELS}
        for t_str, la, lo in pts:
            t = parse_ct(t_str)
            for n in MODELS:
                rec[n].append(win_min(fields[n], t, la, lo, WINDOW_DEG))
        df = pd.DataFrame({"time": times, **rec})
        df.to_csv(f"/results/tc_twoblock_track{tid}.csv", index=False)

        tr = np.array(rec["25 km truth"])
        row = {}
        for n in GEN:
            e = np.array(rec[n]) - tr
            row[n] = dict(
                mae=float(np.nanmean(np.abs(e))),
                mae_00z=float(np.nanmean(np.abs(e[is00]))),
                mae_int=float(np.nanmean(np.abs(e[~is00]))),
                bias=float(np.nanmean(e)),
                corr=float(np.corrcoef(np.array(rec[n]), tr)[0, 1]),
                min_slp=float(np.nanmin(rec[n])),
            )
        summary[tid] = dict(label=label, truth_min=float(np.nanmin(tr)), models=row)
        print(f"  track {tid} ({label}): truth min {np.nanmin(tr):.0f} mb", flush=True)
        for n in GEN:
            r = row[n]
            print(f"    {n:26s} MAE {r['mae']:5.1f} (00Z {r['mae_00z']:.1f} / "
                  f"int {r['mae_int']:.1f})  bias {r['bias']:+.1f}  corr {r['corr']:.2f}  "
                  f"min {r['min_slp']:.0f}", flush=True)

        # timeseries figure
        x = np.arange(len(times))
        i00 = np.where(is00)[0]
        fig, ax = plt.subplots(figsize=(12, 5))
        for i in i00:
            ax.axvline(i, color="0.85", lw=0.8, zorder=0)
        for n in ["Joint Gen w/o Temp Attn", "Joint Gen - flat", "Cascaded Generation",
                  "Two Block", "25 km truth"]:
            ax.plot(x, rec[n], "-", color=COLORS[n], marker="o", ms=3,
                    lw=3 if n == "25 km truth" else 1.7, label=n,
                    zorder=10 if n == "25 km truth" else 5)
        ax.scatter(i00, tr[i00], s=60, facecolors="none", edgecolors="0.3", zorder=11,
                   label="00Z (coarse-conditioned frame)")
        tk = list(range(0, len(times), 8))
        ax.set_xticks(tk)
        ax.set_xticklabels([times[i][5:16] for i in tk], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"min PRMSL in {2*WINDOW_DEG:.0f}°×{2*WINDOW_DEG:.0f}° window (mb)")
        ax.set_title(f"Track {tid} ({label}) — TC min-pressure lifecycle")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"/results/fig_tc_twoblock_track{tid}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    json.dump(summary, open("/results/tc_twoblock_summary.json", "w"), indent=2)


# ============================================================ B: mid-lat cyclone
def argmin2d(sub):
    v = to_mb(sub.values)
    if not np.isfinite(v).any():
        return None
    j, i = np.unravel_index(np.nanargmin(v), v.shape)
    return (float(sub["latitude"].values[j]), float(sub["longitude"].values[i]),
            float(v[j, i]))


def detect_track(truth, name, t0, t1, latb, lonb):
    scan = truth.sel(time=slice(t0, t1))["time"].values[::4]
    best = None
    for t in scan:
        sub = truth.sel(time=t).sel(latitude=slice(*latb), longitude=slice(*lonb))
        r = argmin2d(sub)
        if r and (best is None or r[2] < best[3]):
            best = (t, r[0], r[1], r[2])
    seed_t, lat0, lon0, seed_p = best
    print(f"  [{name}] seed {seed_t} ({lat0:.1f}N {lon0:.1f}E) {seed_p:.1f} mb", flush=True)
    all_t = truth["time"].values
    si = int(np.where(all_t == seed_t)[0][0])
    step_h = (all_t[si + 1] - all_t[si]).total_seconds() / 3600.0
    nmax = int(ETC_MAX_HOURS / step_h)

    def walk(d):
        out, lat, lon = [], lat0, lon0
        for k in range(1, nmax + 1):
            idx = si + d * k
            if idx < 0 or idx >= len(all_t):
                break
            t = all_t[idx]
            r = argmin2d(truth.sel(time=t, method="nearest").sel(
                latitude=slice(lat - ETC_WIN_TRACK, lat + ETC_WIN_TRACK),
                longitude=slice(lon - ETC_WIN_TRACK, lon + ETC_WIN_TRACK)))
            if r is None:
                break
            nlat, nlon, p = r
            if abs(nlat - lat) > ETC_MAX_STEP or abs(nlon - lon) > ETC_MAX_STEP or p > ETC_FILL_MB:
                break
            lat, lon = nlat, nlon
            out.append((str(t), lat, lon))
        return out

    return walk(-1)[::-1] + [(str(all_t[si]), lat0, lon0)] + walk(1)


def etc_analysis(fields):
    print("=== B: mid-latitude cyclone trajectory ===", flush=True)
    truth = fields["25 km truth"]
    lat, lon = truth["latitude"].values, truth["longitude"].values
    summary = {}
    for name, t0, t1, latb, lonb in ETC_BASINS:
        track = detect_track(truth, name, t0, t1, latb, lonb)
        times = [p[0] for p in track]
        is00 = np.array([s.endswith("00:00:00") for s in times])
        rec = {n: [] for n in MODELS}
        for t_str, la, lo in track:
            t = parse_ct(t_str)
            for n in MODELS:
                rec[n].append(win_min(fields[n], t, la, lo, ETC_WIN_MIN))
        df = pd.DataFrame({"time": times, **rec})
        df.to_csv(f"/results/etc_twoblock_{name}.csv", index=False)
        tr = np.array(rec["25 km truth"])
        row = {}
        for n in GEN:
            e = np.array(rec[n]) - tr
            row[n] = dict(mae=float(np.nanmean(np.abs(e))),
                          mae_00z=float(np.nanmean(np.abs(e[is00]))),
                          mae_int=float(np.nanmean(np.abs(e[~is00]))),
                          bias=float(np.nanmean(e)), min_slp=float(np.nanmin(rec[n])))
        summary[name] = dict(truth_min=float(np.nanmin(tr)), n_frames=len(times), models=row)
        print(f"  [{name}] truth min {np.nanmin(tr):.0f} mb, {len(times)} frames", flush=True)
        for n in GEN:
            r = row[n]
            print(f"    {n:26s} 00Z {r['mae_00z']:.1f} / int {r['mae_int']:.1f} mb  "
                  f"bias {r['bias']:+.1f}  min {r['min_slp']:.0f}", flush=True)

        x = np.arange(len(times))
        i00 = np.where(is00)[0]
        fig, ax = plt.subplots(figsize=(12, 4.8))
        for i in i00:
            ax.axvline(i, color="0.85", lw=0.9, zorder=0)
        for n in ["Joint Gen w/o Temp Attn", "Joint Gen - flat", "Cascaded Generation",
                  "Two Block", "25 km truth"]:
            ax.plot(x, rec[n], "-", color=COLORS[n], marker="o", ms=3,
                    lw=3 if n == "25 km truth" else 1.7, label=n,
                    zorder=10 if n == "25 km truth" else 5)
        ax.scatter(i00, tr[i00], s=55, facecolors="none", edgecolors="0.3", zorder=11,
                   label="00Z coarse-conditioned frame")
        tk = list(range(0, len(times), 4))
        ax.set_xticks(tk)
        ax.set_xticklabels([times[i][5:16] for i in tk], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"min PRMSL in {2*ETC_WIN_MIN:.0f}°×{2*ETC_WIN_MIN:.0f}° window (mb)")
        ax.set_title(f"{name} extratropical cyclone — min-pressure trajectory")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"/results/fig_etc_twoblock_{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    json.dump(summary, open("/results/etc_twoblock_summary.json", "w"), indent=2)


# ============================================================ C: animations
def animations(fields):
    print("=== C: zoomed animations ===", flush=True)
    lat = fields["25 km truth"]["latitude"].values
    lon = fields["25 km truth"]["longitude"].values
    names = list(MODELS)
    for tid, (label, pts) in TC_TRACKS.items():
        frames = []
        for i, (t_str, la, lo) in enumerate(pts):
            t = parse_ct(t_str)
            panels, mn, mx = {}, [], []
            for n in names:
                full = to_mb(fields[n].sel(time=t, method="nearest").values)
                v, cla, clo = crop(full, lat, lon, la, lo, ETC_WIN_MIN)
                panels[n] = (v, cla, clo)
                if np.isfinite(v).any():
                    mn.append(np.nanmin(v)); mx.append(np.nanmax(v))
            vmin, vmax = min(mn), max(mx)
            fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
            for ax, n in zip(axes, names):
                v, cla, clo = panels[n]
                ax.pcolormesh(clo, cla, v, cmap="turbo_r", vmin=vmin, vmax=vmax, shading="auto")
                ax.scatter([lo], [la], marker="x", color="red", s=35)
                ax.set_title(n, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"Track {tid} ({label}) — {t_str}  [{vmin:.0f}-{vmax:.0f} mb]  "
                         f"frame {i+1}/{len(pts)}", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
            plt.close(fig)
            if (i + 1) % 15 == 0 or i == len(pts) - 1:
                print(f"  track {tid} frame {i+1}/{len(pts)}", flush=True)
        out = f"/results/tc_twoblock_animation_track{tid}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / FPS), loop=0)
        print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    fields = {n: prmsl_da(p, e) for n, (p, e) in MODELS.items()}
    for n, (p, e) in MODELS.items():
        tv = xr.open_zarr(p)["time"].values
        print(f"{n}: {tv[0]} .. {tv[-1]} ({len(tv)})", flush=True)
    tc_lifecycle(fields)
    etc_analysis(fields)
    animations(fields)
    print("ALL_DONE", flush=True)
