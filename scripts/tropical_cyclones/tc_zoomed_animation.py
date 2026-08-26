# Animated (GIF), tightly-zoomed view of TC track 789's genesis-to-peak
# evolution: one frame per known 3-hourly timestep (2023-05-24 06:00
# through 2023-06-01 00:00, matching the static grid's endpoint -- the
# later extratropical-decay tail at high latitude is dropped since it's no
# longer a compact, visually interesting cyclone). Camera follows the
# storm's own known-track position every frame; crop is much tighter than
# the static grid (2.5 deg vs 6 deg half-width) to actually resolve
# structure. Panels: 25km truth + all 8 models, ensemble member 0, 3x3
# grid -- row 1: truth + the two true-endpoint two-stage models
# (`st-flat`/`st-ou`); row 2: `st-singlestage-flat` + the two models built
# on the same "hiro" checkpoint (`hiro` itself, real dense truth input;
# `cascade-infill-then-sr`, reconstructed input); row 3: the three
# coarse-endpoints (`ce-*`) models.
#
# To reuse for a different storm/window: pull the track's full timeline
# from known_tracks_2023_filtered_25km.csv (`df[df.track_id==<id>]`) and
# adjust TRACK_ID / END_TIME / WINDOW_DEG below.
import re

import cftime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

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

WINDOW_DEG = 2.5
FPS = 5
OUT_PATH = "/results/tc_zoomed_animation_track789.gif"


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

    frames = []
    for i, (t_str, lat0, lon0) in enumerate(zip(times, lats, lons)):
        t = parse_cftime(t_str)
        lat_win = slice(lat0 - WINDOW_DEG, lat0 + WINDOW_DEG)
        lon_win = slice(lon0 - WINDOW_DEG, lon0 + WINDOW_DEG)

        panel_data = {}
        vmins, vmaxs = [], []
        for name in row_names:
            da = (
                fields[name]
                .sel(time=t, method="nearest")
                .sel(latitude=lat_win, longitude=lon_win)
            )
            vals = da.values
            # video-PMD model zarrs store PRMSL in Pa; hiro/cascade and the
            # truth zarr store it natively in hPa -- detect which.
            if np.nanmean(vals) > 2000:
                vals = vals / 100.0
            panel_data[name] = (vals, da["latitude"].values, da["longitude"].values)
            if vals.size:
                vmins.append(np.nanmin(vals))
                vmaxs.append(np.nanmax(vals))
        vmin, vmax = min(vmins), max(vmaxs)

        fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5))
        for idx, name in enumerate(row_names):
            ax = axes[idx // 3, idx % 3]
            vals, lat, lon = panel_data[name]
            ax.pcolormesh(
                lon, lat, vals, cmap="turbo_r", vmin=vmin, vmax=vmax, shading="auto"
            )
            ax.scatter([lon0], [lat0], marker="x", color="red", s=40)
            ax.set_title(name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"track 789 -- {t_str}  [{vmin:.0f}-{vmax:.0f} mb]  "
            f"(frame {i + 1}/{len(times)})",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("RGB"))
        plt.close(fig)
        if (i + 1) % 10 == 0 or i == len(times) - 1:
            print(f"frame {i + 1}/{len(times)} done")

    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
    )
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
