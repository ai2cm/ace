# Static line-plot companion to tc_zoomed_animation.py: minimum SLP within
# each frame's window (25km truth + all 8 models, ensemble member 0) over
# track 789's genesis-to-peak evolution -- same 63 timesteps, same 5x5 deg
# window centered on the storm's own known-track position each frame, same
# time range (2023-05-24 06:00 through 2023-06-01 00:00) as the animation.
# A quick static read of temporal coherence: a model tracking the storm
# smoothly traces a curve close to truth's; a model losing the storm
# between conditioning points shows visible noise/discontinuities.
import re

import cftime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

FINE_TRUTH = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr"
)
KNOWN_TRACKS_CSV = "/known_tracks/known_tracks_2023_filtered_25km.csv"
TRACK_ID = 789
END_TIME = "2023-06-01 00:00:00"

MODELS = {
    "25km truth": (FINE_TRUTH, None),
    "st-flat": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-flat-v1/"
        "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr",
        0,
    ),
    "st-ou": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-ou-v1/"
        "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr",
        0,
    ),
    "st-singlestage-flat": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "global-25km-100km-spatiotemporal-singlestage-5ch-flat-v1/"
        "test-2023-2024-ens4-region-lat-44to44-lon0to180.zarr",
        0,
    ),
    "hiro": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "hiro-downscaling-25km-100km-global-5ch-v6-copied/test-2023-2024-ens4.zarr",
        0,
    ),
    "cascade-infill-then-sr": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "hiro-downscaling-25km-100km-global-5ch-v6-cascade-infill-then-sr/"
        "test-2023-2024-ens4.zarr",
        0,
    ),
    "ce-flat": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat/"
        "test-2023-2024-ens4-global.zarr",
        0,
    ),
    "ce-ou": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-ou/"
        "test-2023-2024-ens4-global.zarr",
        0,
    ),
    "ce-flat-nta": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat-no-temporal-attn/"
        "test-2023-2024-ens4-global.zarr",
        0,
    ),
}

COLORS = {
    "25km truth": "black",
    "st-flat": "#1f77b4",
    "st-ou": "#17becf",
    "st-singlestage-flat": "#2ca02c",
    "hiro": "#9467bd",
    "cascade-infill-then-sr": "#e377c2",
    "ce-flat": "#d62728",
    "ce-ou": "#ff7f0e",
    "ce-flat-nta": "#8c564b",
}

WINDOW_DEG = 2.5
OUT_PNG = "/results/tc_min_pressure_timeseries_track789.png"
OUT_CSV = "/results/tc_min_pressure_timeseries_track789.csv"


def parse_cftime(time_str: str) -> cftime.DatetimeJulian:
    m = re.match(r"(\d+)-(\d+)-(\d+)[ T](\d+):(\d+):(\d+)", time_str)
    assert m is not None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    return cftime.DatetimeJulian(y, mo, d, h, mi, s)


def load_prmsl(path, ens_idx):
    ds = xr.open_zarr(path)
    da = ds["PRMSL"]
    if ens_idx is not None and "ensemble" in da.dims:
        da = da.isel(ensemble=ens_idx)
    return da


def main():
    df = pd.read_csv(KNOWN_TRACKS_CSV)
    track = df[df.track_id == TRACK_ID].sort_values("time")
    track = track[track["time"] <= END_TIME]
    times = track["time"].tolist()
    lats = track["lat"].tolist()
    lons = track["lon"].tolist()
    print(f"{len(times)} frames: {times[0]} -> {times[-1]}")

    fields = {name: load_prmsl(path, ens) for name, (path, ens) in MODELS.items()}
    row_names = list(MODELS.keys())

    records: dict[str, list[float]] = {name: [] for name in row_names}
    for i, (t_str, lat0, lon0) in enumerate(zip(times, lats, lons)):
        t = parse_cftime(t_str)
        lat_win = slice(lat0 - WINDOW_DEG, lat0 + WINDOW_DEG)
        lon_win = slice(lon0 - WINDOW_DEG, lon0 + WINDOW_DEG)
        for name in row_names:
            da = (
                fields[name]
                .sel(time=t, method="nearest")
                .sel(latitude=lat_win, longitude=lon_win)
            )
            vals = da.values
            if np.nanmean(vals) > 2000:
                vals = vals / 100.0
            records[name].append(np.nanmin(vals) if vals.size else np.nan)
        if (i + 1) % 10 == 0 or i == len(times) - 1:
            print(f"frame {i + 1}/{len(times)} done")

    out = pd.DataFrame({"time": times, **records})
    out.to_csv(OUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(times))
    for name in row_names:
        style = dict(color=COLORS[name], linewidth=2.6 if name == "25km truth" else 1.6)
        if name == "25km truth":
            style["zorder"] = 10
        elif name == "hiro":
            style["linestyle"] = "--"
        elif name == "cascade-infill-then-sr":
            style["linestyle"] = ":"
        ax.plot(x, records[name], label=name, **style)

    tick_idx = list(range(0, len(times), 8))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [times[i][5:16] for i in tick_idx], rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("min SLP in 5x5 deg window (mb)")
    ax.set_title(
        "track 789: min-pressure time series, 25km truth vs. all 8 models\n"
        "(same window/timesteps as tc_zoomed_animation_track789.gif)"
    )
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
