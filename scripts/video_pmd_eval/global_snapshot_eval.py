# %% [markdown]
# # Global snapshot + one-day temporal series for a stage-2 video-PMD model
#
# Matches the two plot types in `crps_eval_results_stage2_st-flat-st-ou/global_snapshots/`
# (built for st-flat/st-ou; reproduced here for any model in
# `PATCHED_MODELS`, defaulting to `st-singlestage-flat`):
#
# - **Global snapshot**: one interior (generated) timestep, ensemble mean,
#   all 5 channels, truth vs. model -- sanity-checks the 4-region tiling is
#   seamless and shows what the generated fields actually look like.
# - **Temporal series**: the same day's full 9-frame clip (00h/24h observed
#   endpoints + 7 generated interior frames), rows = LR coarse input (shown
#   at the 00h/24h columns only), model, HR truth.

# %%
import argparse

import cftime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

plt.rcParams["figure.dpi"] = 100

# %%
FINE_TRUTH_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr"
)
COARSE_TRUTH_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr"
)
PATCHED_MODELS = {
    # No-temporal-attention ablation of st-singlestage-coarse-endpoints-flat
    # (see crps_eval.py's PATCHED_MODELS comment for the full provenance
    # note). Global patch-tiled inference, ONE contiguous global zarr on
    # weka -- plain str path.
    "st-singlestage-coarse-endpoints-flat-no-temporal-attn": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat-no-temporal-attn/"
        "test-2023-2024-ens4-global.zarr"
    ),
    # Single-stage coarse-endpoints (v2 of the single-stage architecture),
    # global patch-tiled inference -- ONE contiguous global zarr, so a
    # plain str path (see crps_eval.py's PATCHED_MODELS comment for the
    # full architecture/provenance note).
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
    # Cascade: 100km temporal infill -> 25km spatial SR (same "hiro"
    # checkpoint, conditioned on the infill output instead of real dense
    # truth). See crps_eval.py's PATCHED_MODELS comment for the full
    # pipeline rationale.
    "cascade-infill-then-sr": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "hiro-downscaling-25km-100km-global-5ch-v6-cascade-infill-then-sr/"
        "test-2023-2024-ens4.zarr"
    ),
    # v2 retrains of st-flat/st-ou after the endpoint-only-conditioning fix
    # (see crps_eval.py's PATCHED_MODELS comment for the full caveat --
    # epoch 41/200, preliminary/undertrained).
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
CHANNELS = [
    "eastward_wind_at_ten_meters",
    "northward_wind_at_ten_meters",
    "PRMSL",
    "PRATEsfc",
    "air_temperature_at_two_meters",
]
UNITS = {
    "eastward_wind_at_ten_meters": "m/s",
    "northward_wind_at_ten_meters": "m/s",
    "PRMSL": "mb",
    "PRATEsfc": "kg/m2/s",
    "air_temperature_at_two_meters": "K",
}
CMAPS = {
    "eastward_wind_at_ten_meters": "RdBu_r",
    "northward_wind_at_ten_meters": "RdBu_r",
    "PRMSL": "viridis",
    "PRATEsfc": "Blues",
    "air_temperature_at_two_meters": "inferno",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global snapshot + temporal series for a stage-2 model"
    )
    parser.add_argument(
        "--model", default="st-singlestage-flat", choices=sorted(PATCHED_MODELS)
    )
    parser.add_argument(
        "--date", default="2023-07-15", help="Clip start date (00h UTC)."
    )
    parser.add_argument("--outdir", default=".")
    return parser.parse_args()


ARGS = parse_args()
OUTDIR = ARGS.outdir


def _load_region_tiled(pred_spec, t0, t1) -> xr.Dataset:
    # pred_spec is either a 4-region dict (see PATCHED_MODELS) or a single
    # str zarr path (already-global output) -- matches crps_eval.py's
    # _load_pred_window dual-mode handling.
    if not isinstance(pred_spec, dict):
        return xr.open_zarr(pred_spec).sel(time=slice(t0, t1)).load()
    parts = {}
    for region, path in pred_spec.items():
        parts[region] = xr.open_zarr(path).sel(time=slice(t0, t1)).load()
    mid_band = xr.concat([parts["mid_west"], parts["mid_east"]], dim="longitude")
    return xr.concat([parts["south_cap"], mid_band, parts["north_cap"]], dim="latitude")


def main():
    y, m, d = (int(x) for x in ARGS.date.split("-"))
    t0 = cftime.DatetimeJulian(y, m, d, 0)
    t1 = cftime.DatetimeJulian(y, m, d, 23)  # inclusive slice -> full 00h..24h clip

    print(f"Loading {ARGS.model} clip starting {ARGS.date}...")
    pred = _load_region_tiled(PATCHED_MODELS[ARGS.model], t0, t1)
    lat = pred["latitude"].values
    lon = pred["longitude"].values
    n_times = pred.sizes["time"]
    interior_mask = pred["frame_source"].values == 1

    print("Loading HR truth...")
    truth = (
        xr.open_zarr(FINE_TRUTH_ZARR)
        .sel(time=slice(t0, t1))
        .sel(latitude=lat, longitude=lon, method="nearest")
        .load()
    )

    print("Loading LR coarse input (endpoints only)...")
    coarse_full = xr.open_zarr(COARSE_TRUTH_ZARR).sel(time=slice(t0, t1))
    coarse_endpoints = coarse_full.isel(time=[0, -1]).load()

    # ---- Global snapshot: 12h-lead interior frame (the hardest one, same
    # convention as crps_eval.py's spatial CRPS map), ensemble mean ----
    lead_hours = np.array([3 * i for i in range(n_times)])
    twelve_h_idx = int(np.where(lead_hours == 12)[0][0])
    for varname in CHANNELS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
        truth_frame = truth[varname].isel(time=twelve_h_idx).values
        pred_frame = pred[varname].isel(time=twelve_h_idx).mean(dim="ensemble").values
        vmin = min(truth_frame.min(), pred_frame.min())
        vmax = max(truth_frame.max(), pred_frame.max())
        cmap = CMAPS[varname]
        axes[0].pcolormesh(lon, lat, truth_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title("truth")
        im = axes[1].pcolormesh(lon, lat, pred_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(ARGS.model)
        for ax in axes:
            ax.set_xlabel("longitude (deg E)")
            ax.set_ylabel("latitude")
        fig.colorbar(im, ax=list(axes), label=f"{varname} ({UNITS[varname]})")
        fig.suptitle(f"{varname}, {ARGS.date} 12h lead, ensemble mean")
        fig.savefig(f"{OUTDIR}/global_snapshot_{varname}_{ARGS.model}.png", dpi=150)
        plt.close(fig)
        print(f"wrote global_snapshot_{varname}_{ARGS.model}.png")

    # ---- Temporal series: all n_times frames, rows = [LR coarse (endpoints
    # only), model ensemble mean, HR truth] ----
    for varname in CHANNELS:
        fig, axes = plt.subplots(3, n_times, figsize=(2.0 * n_times, 6.0))
        cmap = CMAPS[varname]
        truth_all = truth[varname].values
        pred_all = pred[varname].mean(dim="ensemble").values
        vmin = min(truth_all.min(), pred_all.min())
        vmax = max(truth_all.max(), pred_all.max())
        coarse_vals = coarse_endpoints[varname].values  # (2, H_c, W_c)

        for t in range(n_times):
            ax = axes[0][t]
            if t == 0 or t == n_times - 1:
                idx = 0 if t == 0 else 1
                ax.pcolormesh(
                    coarse_endpoints["longitude"].values,
                    coarse_endpoints["latitude"].values,
                    coarse_vals[idx],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            obs = "\n(obs)" if t in (0, n_times - 1) else ""
            ax.set_title(f"{3 * t}h{obs}", fontsize=8)
            if t == 0:
                ax.set_ylabel("LR input", fontsize=9)

            ax = axes[1][t]
            im = ax.pcolormesh(lon, lat, pred_all[t], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(ARGS.model, fontsize=9)

            ax = axes[2][t]
            ax.pcolormesh(lon, lat, truth_all[t], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel("HR truth", fontsize=9)

        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            label=f"{varname} ({UNITS[varname]})",
            shrink=0.6,
        )
        fig.suptitle(
            f"{varname} -- {ARGS.date} full day, LR input / {ARGS.model} / HR truth"
        )
        fig.savefig(f"{OUTDIR}/temporal_series_{varname}_{ARGS.model}.png", dpi=150)
        plt.close(fig)
        print(f"wrote temporal_series_{varname}_{ARGS.model}.png")

    print("done.")


if __name__ == "__main__":
    main()
