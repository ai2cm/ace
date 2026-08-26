# Time-series grid for one TC's full lifecycle: columns = timesteps
# (genesis -> peak intensity -> extratropical decay), rows = 25km truth +
# every model (ensemble member 0), all showing the SLP field in a window
# centered on the storm's own position at that time. Visual complement to
# the aggregate numbers in toy/tc_verification_analysis.md.
#
# To reuse for a different storm: pull its full timeline from
# known_tracks_2023_filtered_25km.csv (`df[df.track_id==<id>]`), pick a
# representative subset of (time, lat, lon) rows spanning the lifecycle,
# and replace TIMESTEPS below. This is a one-off illustrative figure, not
# a reusable pipeline component, so track/timestep selection is a plain
# hardcoded list rather than a CLI arg.
#
# Color scale is normalized PER COLUMN (per timestep), not globally --
# a storm's ambient/peak pressure legitimately drifts a lot over its life
# (e.g. genesis ~1005mb ambient vs. ~920mb near peak for track 789), so one
# fixed scale across very different snapshots washes out the truth-vs-model
# contrast within each timestep, which is the actual comparison of
# interest.
import re

import cftime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

FINE_TRUTH = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr"
)

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

# (time, lat, lon) for track 789's lifecycle -- genesis through extratropical decay
TIMESTEPS = [
    ("2023-05-24T12:00:00", 9.348994, 145.125),
    ("2023-05-26T00:00:00", 10.844833, 144.625),
    ("2023-05-27T12:00:00", 13.337898, 144.875),
    ("2023-05-28T21:00:00", 16.828188, 144.625),
    ("2023-05-29T18:00:00", 18.822639, 146.375),
    ("2023-05-29T21:00:00", 19.321253, 146.875),
    ("2023-05-30T12:00:00", 22.811543, 148.875),
    ("2023-06-01T00:00:00", 34.030323, 145.375),
]
WINDOW_DEG = 6.0
OUT_PATH = "/results/tc_timeseries_grid_track789.png"


def parse_cftime(time_str: str) -> cftime.DatetimeJulian:
    m = re.match(r"(\d+)-(\d+)-(\d+)T(\d+):(\d+):(\d+)", time_str)
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
    fields = {name: load_prmsl(path, ens) for name, (path, ens) in MODELS.items()}
    row_names = list(MODELS.keys())

    n_rows, n_cols = len(row_names), len(TIMESTEPS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows), sharex=False, sharey=False
    )

    cache = {}
    col_vmin, col_vmax = {}, {}
    for col, (t_str, lat0, lon0) in enumerate(TIMESTEPS):
        t = parse_cftime(t_str)
        lat_win = slice(lat0 - WINDOW_DEG, lat0 + WINDOW_DEG)
        lon_win = slice(lon0 - WINDOW_DEG, lon0 + WINDOW_DEG)
        col_mins, col_maxs = [], []
        for row, name in enumerate(row_names):
            da = (
                fields[name]
                .sel(time=t, method="nearest")
                .sel(latitude=lat_win, longitude=lon_win)
            )
            vals = da.values
            # video-PMD model zarrs store PRMSL in Pa; hiro and the truth
            # zarr store it natively in hPa -- detect which.
            if np.nanmean(vals) > 2000:
                vals = vals / 100.0
            cache[(row, col)] = (vals, da["latitude"].values, da["longitude"].values)
            if vals.size:
                col_mins.append(np.nanmin(vals))
                col_maxs.append(np.nanmax(vals))
        col_vmin[col], col_vmax[col] = min(col_mins), max(col_maxs)

    for col, (t_str, lat0, lon0) in enumerate(TIMESTEPS):
        vmin, vmax = col_vmin[col], col_vmax[col]
        for row, name in enumerate(row_names):
            ax = axes[row, col]
            vals, lat, lon = cache[(row, col)]
            ax.pcolormesh(
                lon, lat, vals, cmap="viridis_r", vmin=vmin, vmax=vmax, shading="auto"
            )
            ax.scatter([lon0], [lat0], marker="x", color="red", s=30)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    t_str[:10] + "\n" + t_str[11:16] + f"\n[{vmin:.0f}-{vmax:.0f}mb]",
                    fontsize=7,
                )
            if col == 0:
                ax.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")

    fig.suptitle(
        "track 789 (2023-05-24 to 06-01): 25km truth vs. every model, "
        "ensemble member 0\n"
        "(color scale normalized PER COLUMN -- each timestep's own min/max "
        "shown in its column header)",
        fontsize=11,
    )
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
