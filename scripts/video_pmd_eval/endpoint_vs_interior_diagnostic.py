# %% [markdown]
# # Day-1 go/no-go diagnostic for the two-block (r, d) proposal
#
# See `idea/spatiotemoral/twoblock_theory.md` (repo root, sibling of `ace/`).
# Prop 1 (endpoint mismatch of any pinned kernel) predicts that for a
# `coarse_endpoints_only` single-stage model (`endpoints_observed=False`,
# `coarse_endpoints_only=True` -- e.g. `st-singlestage-coarse-endpoints-flat`
# / `-ou`), the two clip-boundary ("endpoint") frames are diffused exactly
# like every interior frame (nothing is pinned in this mode), yet the
# single-stage noise model treats them identically to interior frames
# despite the residual's true endpoint variance coming ENTIRELY from the
# unpinned fine-detail component. The falsifiable prediction: **endpoint-time
# error should be comparable to or worse than interior-time error**, not
# dramatically better -- if a pinned-style kernel (or a model that implicitly
# learned to treat endpoints as "easier") were doing something structurally
# different there, we'd instead see endpoints noticeably outperforming the
# interior.
#
# This is a go/no-go check, not a full report: it does NOT distinguish a
# pinned-kernel model (like `st-flat`, where this comparison is meaningless
# -- endpoints are trivially exact by construction there) from the
# `coarse_endpoints_only` models this diagnostic actually targets. Only run
# this against `PATCHED_MODELS` entries below.
#
# Note: `frame_source` in these models' output zarrs is uniformly 1 (no
# pinned frame exists in `coarse_endpoints_only` mode, see
# `fme/downscaling/video_inference.py`'s `frame_source` comment) -- it does
# NOT mark endpoint vs. interior here, unlike in `crps_eval.py`'s usage for
# `endpoints_observed=True` models. This script instead uses clip-boundary
# POSITION (`lead_hour == 0`, i.e. every `clip_stride`-th frame), the same
# computation `crps_eval.py` uses for its lead-time breakdown, just without
# discarding the `lead_hour == 0` rows the way that script's `interior_mask`
# implicitly does.

# %%
import argparse

import cftime
import numpy as np
import pandas as pd
import xarray as xr

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %%
FINE_TRUTH_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr"
)
# Only coarse_endpoints_only (endpoints_observed=False) models belong here --
# see the module docstring for why this comparison is meaningless for
# pinned-endpoint models (st-flat/st-ou/st-singlestage-flat).
PATCHED_MODELS = {
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
    "st-singlestage-coarse-endpoints-flat-no-temporal-attn": (
        "/climate-default/2026-06-25-temporal-diffusion/inference/"
        "video-pmd-spatiotemporal-25km-100km-global-5ch-singlestage-coarse-endpoints-flat-no-temporal-attn/"
        "test-2023-2024-ens4-global.zarr"
    ),
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
# Same 4-season sampling as crps_eval.py's SAMPLE_WINDOWS, for a
# representative-of-the-annual-cycle estimate without reading the full
# 2023-01-01..2024-01-04 test period.
SAMPLE_WINDOWS = [
    (
        "winter (Jan)",
        cftime.DatetimeJulian(2023, 1, 1),
        cftime.DatetimeJulian(2023, 1, 4),
    ),
    (
        "spring (Apr)",
        cftime.DatetimeJulian(2023, 4, 1),
        cftime.DatetimeJulian(2023, 4, 4),
    ),
    (
        "summer (Jul)",
        cftime.DatetimeJulian(2023, 7, 1),
        cftime.DatetimeJulian(2023, 7, 4),
    ),
    (
        "fall (Oct)",
        cftime.DatetimeJulian(2023, 10, 1),
        cftime.DatetimeJulian(2023, 10, 4),
    ),
]
TIME_STEP_HOURS = 3
N_TIMESTEPS = 9  # clip length; matches config.model.n_timesteps
CLIP_STRIDE = N_TIMESTEPS - 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Day-1 two-block diagnostic: endpoint-time vs "
        "interior-time error for a coarse_endpoints_only model."
    )
    parser.add_argument(
        "--model",
        default="st-singlestage-coarse-endpoints-flat",
        choices=sorted(PATCHED_MODELS),
    )
    parser.add_argument("--outdir", default=".")
    return parser.parse_args()


def _load_pred_window(pred_spec, t0, t1) -> xr.Dataset:
    # Dual-mode (single zarr path or 4-region dict) for consistency with
    # crps_eval.py's _load_pred_window, even though every PATCHED_MODELS
    # entry above is currently a single global zarr.
    if isinstance(pred_spec, dict):
        parts = {
            r: xr.open_zarr(p).sel(time=slice(t0, t1)).load()
            for r, p in pred_spec.items()
        }
        mid_band = xr.concat([parts["mid_west"], parts["mid_east"]], dim="longitude")
        return xr.concat(
            [parts["south_cap"], mid_band, parts["north_cap"]], dim="latitude"
        )
    return xr.open_zarr(pred_spec).sel(time=slice(t0, t1)).load()


def load_window(pred_spec, truth_raw, t0, t1):
    """(pred, truth, is_endpoint) for one time window. ``is_endpoint`` marks
    clip-boundary POSITIONS (every ``CLIP_STRIDE``-th frame), regardless of
    ``frame_source`` -- see the module docstring for why."""
    p = _load_pred_window(pred_spec, t0, t1)
    t = (
        truth_raw.sel(time=slice(t0, t1))
        .sel(latitude=p.latitude, longitude=p.longitude, method="nearest")
        .load()
    )
    lead_hour = (np.arange(p.sizes["time"]) % CLIP_STRIDE) * TIME_STEP_HOURS
    is_endpoint = lead_hour == 0
    return p, t, is_endpoint


def crps_fair(ens: np.ndarray, truth_arr: np.ndarray) -> np.ndarray:
    """Fair CRPS estimator (Ferro et al. 2008), ensemble axis last. Same
    formula as crps_eval.py's crps_fair, duplicated here to keep this script
    self-contained (see that module's docstring for the derivation)."""
    m = ens.shape[-1]
    mae = np.abs(ens - truth_arr[..., None]).mean(axis=-1)
    sorted_ens = np.sort(ens, axis=-1)
    weights = 2 * np.arange(1, m + 1) - m - 1
    spread_term = (weights * sorted_ens).sum(axis=-1) / (m * (m - 1))
    return mae - spread_term


def score_split(pred: xr.Dataset, truth: xr.Dataset, mask: np.ndarray, varname: str):
    """(mse, crps) of the ensemble-mean/ensemble against truth, pooled over
    every (time in mask, lat, lon) grid point."""
    ens = pred[varname].isel(time=mask).transpose(..., "ensemble").values
    truth_arr = truth[varname].isel(time=mask).values
    ens_mean = ens.mean(axis=-1)
    mse = float(np.mean((ens_mean - truth_arr) ** 2))
    crps = float(np.mean(crps_fair(ens, truth_arr)))
    return mse, crps


def main():
    args = parse_args()
    pred_spec = PATCHED_MODELS[args.model]
    truth_raw = xr.open_zarr(FINE_TRUTH_ZARR)

    rows = []
    for season, t0, t1 in SAMPLE_WINDOWS:
        print(f"loading {args.model} / {season}...")
        pred, truth, is_endpoint = load_window(pred_spec, truth_raw, t0, t1)
        for varname in CHANNELS:
            mse_end, crps_end = score_split(pred, truth, is_endpoint, varname)
            mse_int, crps_int = score_split(pred, truth, ~is_endpoint, varname)
            rows.append(
                {
                    "season": season,
                    "channel": varname,
                    "mse_endpoint": mse_end,
                    "mse_interior": mse_int,
                    "mse_ratio (endpoint/interior)": mse_end / mse_int,
                    "crps_endpoint": crps_end,
                    "crps_interior": crps_int,
                    "crps_ratio (endpoint/interior)": crps_end / crps_int,
                }
            )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("channel")[
            [
                "mse_endpoint",
                "mse_interior",
                "crps_endpoint",
                "crps_interior",
            ]
        ]
        .mean()
        .assign(
            mse_ratio=lambda d: d["mse_endpoint"] / d["mse_interior"],
            crps_ratio=lambda d: d["crps_endpoint"] / d["crps_interior"],
        )
    )
    print(
        f"\n=== {args.model}: endpoint-time vs interior-time error "
        f"(pooled over {len(SAMPLE_WINDOWS)} seasonal windows) ==="
    )
    print(summary)
    out_csv = f"{args.outdir}/endpoint_vs_interior_{args.model}.csv"
    summary.to_csv(out_csv)
    print(f"\nwrote {out_csv}")

    go = bool((summary["mse_ratio"] >= 0.9).all())
    verdict = "GO" if go else "NO-GO (needs review)"
    explanation = (
        "endpoints are NOT easier than interior, consistent with the "
        "unpinned-detail-residual diagnosis"
        if go
        else "endpoints score meaningfully better than interior, which the "
        "theory does not predict -- investigate before proceeding."
    )
    print(
        f"\nProp 1 prediction: endpoint-time error should be comparable to "
        f"or worse than interior-time error (ratio >= ~1). "
        f"mse_ratio per channel:\n{summary['mse_ratio']}\n"
        f"Verdict: {verdict} -- {explanation}"
    )


if __name__ == "__main__":
    main()
