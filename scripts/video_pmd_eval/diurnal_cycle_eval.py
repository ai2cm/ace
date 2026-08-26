# %% [markdown]
# # Diurnal cycle of T2m/precipitation for a stage-2 spatiotemporal video-PMD model
#
# Same local-solar-time compositing methodology as
# `scripts/tropical_cyclones/diurnal_cycle_walkthrough.ipynb` (the stage-1,
# 100km/1deg version) and the stage-2 two-stage comparison in
# `crps_eval_results_stage2_st-flat-st-ou/COMPARISON_REPORT.md` -- adapted
# here to run against any model in `crps_eval.py`'s `PATCHED_MODELS` (4-region
# tiled 25km output), defaulting to `st-singlestage-flat` (the one model in
# that family this hasn't been run for yet).
#
# `local_hour = (utc_hour + lon/15) mod 24`, per-column time-mean subtracted
# before binning (isolates diurnal *shape* from each bin's particular mix of
# climate zones -- see the stage-1 notebook's markdown for why), 8 bins
# (matching the data's native 3-hourly cadence), area-weighted by cos(lat).
# Only `frame_source==1` (generated-interior) frames are composited for the
# model; truth uses every frame in the window.
#
# Land/ocean masking uses the 100km coarse truth store's `land_fraction`
# (the 25km fine store has none), nearest-neighbor regridded onto the 25km
# grid -- same approach and same minor coastline-precision caveat as the
# two-stage report.

# %%
import argparse
import gc
import json

import cftime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
# Same 4-region tiling as crps_eval.py's PATCHED_MODELS -- copied rather than
# imported since crps_eval.py executes argparse at import time.
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
    # HiRO-ACE-style spatial downscaling baseline: plain single-frame SR,
    # zero temporal conditioning. n_ens=4 test-inference (see
    # crps_eval.py's PATCHED_MODELS comment for the full provenance note).
    # Not on weka -- mounted at /hiro_result only when "hiro" is requested.
    "hiro": "/hiro_result/test-2023-2024-ens4.zarr",
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
}
NBINS = 8
BIN_EDGES = np.linspace(0.0, 24.0, NBINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diurnal cycle of T2m/precip for a 4-region-tiled stage-2 model"
    )
    parser.add_argument(
        "--model", default="st-singlestage-flat", choices=sorted(PATCHED_MODELS)
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--outdir", default=".")
    return parser.parse_args()


ARGS = parse_args()
OUTDIR = ARGS.outdir


def _load_region_tiled(
    pred_spec, t0, t1, variables: list[str] | None = None
) -> xr.Dataset:
    """Same manual per-region-then-concat loading as crps_eval.py's
    _load_pred_window -- see that function's docstring for why a lazy
    4-way concat over the full multi-year time axis reliably OOMs and this
    doesn't. ``variables``, if given, subsets BEFORE .load() -- loading all
    5 channels when only 2 are needed for a whole JJA season is what OOM'd a
    96GiB job on the first attempt at this script.

    ``pred_spec`` is either a 4-region dict (see PATCHED_MODELS) or a single
    str zarr path (already-global output, e.g. from a divide_generation
    patch-tiled inference run) -- matches crps_eval.py's _load_pred_window
    dual-mode handling."""
    if not isinstance(pred_spec, dict):
        ds = xr.open_zarr(pred_spec).sel(time=slice(t0, t1))
        if variables is not None:
            ds = ds[variables]
        return ds.load()
    parts = {}
    for region, path in pred_spec.items():
        ds = xr.open_zarr(path).sel(time=slice(t0, t1))
        if variables is not None:
            ds = ds[variables]
        parts[region] = ds.load()
    mid_band = xr.concat([parts["mid_west"], parts["mid_east"]], dim="longitude")
    return xr.concat([parts["south_cap"], mid_band, parts["north_cap"]], dim="latitude")


def region_mask(lat, lon, land_fraction, lat_bounds, lon_bounds, land=None):
    lat_ok = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_ok = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
    mask = lat_ok[:, None] & lon_ok[None, :]
    if land is True:
        mask = mask & (land_fraction > 0.5)
    elif land is False:
        mask = mask & (land_fraction <= 0.5)
    return mask


def diurnal_composite(field_anom, local_hour, lat, mask):
    """Area-weighted mean anomaly per local-hour bin, (ntime, nlat, nlon)
    field_anom under ``mask`` -- identical logic to the stage-1 notebook's
    function of the same name."""
    ilat, ilon = np.where(mask)
    weights = np.cos(np.radians(lat[ilat]))
    bin_idx = np.clip(np.digitize(local_hour[:, ilon], BIN_EDGES) - 1, 0, NBINS - 1)
    vals = field_anom[:, ilat, ilon]
    w2d = np.broadcast_to(weights, vals.shape)
    sums = np.array(
        [(vals[bin_idx == b] * w2d[bin_idx == b]).sum() for b in range(NBINS)]
    )
    wsums = np.array([w2d[bin_idx == b].sum() for b in range(NBINS)])
    # A bin can end up with zero weight for small regions: the model side
    # only composites interior (generated) frames -- excluding the
    # clip-boundary frames, which are always exactly UTC 00:00 since clips
    # tile every 24h -- so a narrow-longitude region can lose the one
    # local-hour bin that UTC 00:00 would have populated for it (truth has
    # no such gap, since it isn't frame_source-filtered). np.errstate +
    # nan-result is intentional here; peak_trough below uses nanmax/nanmin
    # so an isolated empty bin doesn't propagate to a fully-NaN amplitude.
    with np.errstate(invalid="ignore"):
        return BIN_CENTERS, sums / wsums


def peak_trough(hours, values):
    # nan-safe: see diurnal_composite's comment on why a small region can
    # have an isolated empty (NaN) bin on the model side.
    return (
        float(hours[np.nanargmax(values)]),
        float(hours[np.nanargmin(values)]),
        float(np.nanmax(values) - np.nanmin(values)),
    )


def main():
    t0 = cftime.DatetimeJulian(ARGS.year, 6, 1)
    t1 = cftime.DatetimeJulian(ARGS.year, 9, 1)

    print(f"Loading {ARGS.model} JJA {ARGS.year}...")
    # hiro (and any other non-video, frame-by-frame-independent model) has
    # no frame_source -- every frame is scoreable, no interior/endpoint
    # distinction, so don't request a variable that doesn't exist.
    pred_spec = PATCHED_MODELS[ARGS.model]
    probe_path = (
        pred_spec if isinstance(pred_spec, str) else next(iter(pred_spec.values()))
    )
    has_frame_source = "frame_source" in xr.open_zarr(probe_path).data_vars
    variables = ["air_temperature_at_two_meters", "PRATEsfc"]
    if has_frame_source:
        variables.append("frame_source")
    pred = _load_region_tiled(pred_spec, t0, t1, variables=variables)
    lat = pred["latitude"].values
    lon = pred["longitude"].values

    print("Loading truth...")
    truth_raw = xr.open_zarr(FINE_TRUTH_ZARR)[
        ["air_temperature_at_two_meters", "PRATEsfc"]
    ]
    truth = (
        truth_raw.sel(time=slice(t0, t1))
        .sel(latitude=lat, longitude=lon, method="nearest")
        .load()
    )

    print("Loading + regridding land_fraction...")
    coarse = xr.open_zarr(COARSE_TRUTH_ZARR)
    land_fraction = (
        coarse["land_fraction"]
        .sel(latitude=lat, longitude=lon, method="nearest")
        .load()
        .values
    )

    if has_frame_source:
        interior_mask = pred["frame_source"].values == 1
    else:
        interior_mask = np.ones(pred.sizes["time"], dtype=bool)
    print(
        f"{pred.sizes['time']} timesteps in window, "
        f"{int(interior_mask.sum())} generated-interior"
    )

    # cftime objects (Julian calendar here), not pandas Timestamps -- read
    # .hour/.minute directly rather than going through pandas.DatetimeIndex.
    utc_hour = np.array([t.hour + t.minute / 60.0 for t in pred["time"].values])
    local_hour = (utc_hour[:, None] + (lon / 15.0)[None, :]) % 24.0

    REGIONS = {
        "land_global": region_mask(
            lat, lon, land_fraction, (-60, 60), (0, 360), land=True
        ),
        "ocean_global": region_mask(
            lat, lon, land_fraction, (-60, 60), (0, 360), land=False
        ),
        "ne_pacific_stratus": region_mask(
            lat, lon, land_fraction, (15, 30), (225, 250), land=False
        ),
        "se_us": region_mask(lat, lon, land_fraction, (28, 35), (270, 280), land=True),
    }
    print("region pixel counts:", {k: int(v.sum()) for k, v in REGIONS.items()})

    results = {}
    for varname, unit, scale, offset in [
        ("air_temperature_at_two_meters", "K", 1.0, 0.0),
        ("PRATEsfc", "mm/day", 86400.0, 0.0),
    ]:
        truth_field = truth[varname].values * scale + offset
        pred_field = (
            pred[varname].isel(time=interior_mask).mean(dim="ensemble").values * scale
            + offset
        )
        pred_local_hour = local_hour[interior_mask]

        truth_anom = truth_field - truth_field.mean(axis=0, keepdims=True)
        pred_anom = pred_field - pred_field.mean(axis=0, keepdims=True)

        results[varname] = {}
        for region_name, mask in REGIONS.items():
            t_hours, t_vals = diurnal_composite(truth_anom, local_hour, lat, mask)
            p_hours, p_vals = diurnal_composite(pred_anom, pred_local_hour, lat, mask)
            t_peak, t_trough, t_amp = peak_trough(t_hours, t_vals)
            p_peak, p_trough, p_amp = peak_trough(p_hours, p_vals)
            results[varname][region_name] = {
                "truth_hours": t_hours.tolist(),
                "truth_values": t_vals.tolist(),
                "model_hours": p_hours.tolist(),
                "model_values": p_vals.tolist(),
                "truth_peak_lst": t_peak,
                "truth_trough_lst": t_trough,
                "truth_amplitude": t_amp,
                "model_peak_lst": p_peak,
                "model_trough_lst": p_trough,
                "model_amplitude": p_amp,
                "amplitude_pct_err": (p_amp - t_amp) / t_amp * 100.0,
                "unit": unit,
            }
        del truth_field, pred_field, truth_anom, pred_anom
        gc.collect()

    with open(f"{OUTDIR}/diurnal_results_{ARGS.model}.json", "w") as f:
        json.dump(results, f, indent=2)

    for varname, tag in [
        ("air_temperature_at_two_meters", "temp"),
        ("PRATEsfc", "precip"),
    ]:
        rows = []
        for region_name, r in results[varname].items():
            rows.append(
                {
                    "region": region_name,
                    "truth_peak_lst": r["truth_peak_lst"],
                    "truth_trough_lst": r["truth_trough_lst"],
                    "truth_amplitude": round(r["truth_amplitude"], 3),
                    f"{ARGS.model}_amplitude": round(r["model_amplitude"], 3),
                    "amplitude_pct_err": round(r["amplitude_pct_err"], 1),
                }
            )
        df = pd.DataFrame(rows).set_index("region")
        df.to_csv(f"{OUTDIR}/diurnal_{tag}_summary_{ARGS.model}.csv")
        print(f"\n=== {varname} ===")
        print(df)

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for region_name, r in results[varname].items():
            ax.plot(
                r["truth_hours"],
                r["truth_values"],
                ls="--",
                marker="o",
                label=f"{region_name} (truth)",
            )
            ax.plot(
                r["model_hours"],
                r["model_values"],
                ls="-",
                marker="s",
                label=f"{region_name} ({ARGS.model})",
            )
        ax.set_xlabel("local solar time (hr)")
        ax.set_ylabel(
            f"anomaly ({results[varname][next(iter(results[varname]))]['unit']})"
        )
        ax.set_xticks(range(0, 25, 3))
        ax.axhline(0, color="gray", lw=0.6)
        ax.legend(fontsize=6, ncol=2)
        ax.set_title(f"Diurnal {tag} cycle, {ARGS.model} vs. truth (JJA {ARGS.year})")
        fig.tight_layout()
        fig.savefig(f"{OUTDIR}/diurnal_{tag}_comparison_{ARGS.model}.png", dpi=150)
        plt.close(fig)

    print(
        f"\nSaved diurnal_results_{ARGS.model}.json, diurnal_{{temp,precip}}_summary_{ARGS.model}.csv, "
        f"diurnal_{{temp,precip}}_comparison_{ARGS.model}.png"
    )


if __name__ == "__main__":
    main()
