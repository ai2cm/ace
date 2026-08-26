"""d-block residual characterization: temporal correlation of
``d(tau) = Pi(x_f(tau) - U(x_c(tau)))`` across the FULL 9-frame window
(0..24h, ENDPOINTS INCLUDED).

Item 2 of idea/spatiotemoral/twoblock_theory.md's Section 5 empirical
program: "the existing coarse-residual report covers the r block. Repeat
for d: temporal correlation of Pi(x_f - Ux_c) -- prediction: approximately
stationary (no endpoint pinning), OU-like decay, per-channel length
scales." Mirrors toy/_bridge_analysis_full.py's methodology (deterministic
vs. stochastic split, pooled covariance, kernel-fit search) but:
  - over all 9 frames (tau = 0, 3, ..., 24h), not just the 7 interior
    frames -- d is NOT pinned, so the endpoints are exactly what's being
    tested (Prop 1's claim: Var(d(0)) > 0, comparable to interior).
  - against a Pi(x_f - U(x_c)) target instead of a linear-interpolation
    residual, and with NO Brownian-bridge candidate in the kernel-fit
    search (bridge assumes pinning, which doesn't apply here).

D (conservative box-average downsample) / U (block-replicate upsample) /
Pi = I - U@D are reimplemented here in plain numpy (factor=4, this
codebase's 25km/100km ratio) rather than imported from
fme.downscaling.twoblock, since this runs as a bare python3 script inside
a CPU-only Beaker session without the `ace` package installed -- same
self-contained-script convention as crps_eval.py /
endpoint_vs_interior_diagnostic.py.
"""

import datetime
import json
from typing import Any

import cftime
import numpy as np
import xarray as xr

FINE_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr"
)
COARSE_ZARR = (
    "/climate-default/2026-06-25-temporal-diffusion/"
    "2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr"
)
FACTOR = 4
N_TIMESTEPS = 9  # matches config.model.n_timesteps
TIME_STEP_HOURS = 3
TAU = np.arange(N_TIMESTEPS) * TIME_STEP_HOURS  # [0, 3, ..., 24]
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
# Seasonally-stratified, phase-aligned (00:00 UTC start) 24h windows from the
# TRAIN split (2013-2021) -- same rationale as crps_eval.py's SAMPLE_WINDOWS
# (representative of the annual cycle without streaming the full 9-year
# period, which at 25km global resolution is far more data per frame than
# the 1-degree store toy/_bridge_analysis_full.py streamed). Two different
# years per season for a little more than one year's idiosyncrasies.
SAMPLE_DATES = [
    (2015, 1, 15),
    (2019, 1, 15),
    (2015, 4, 15),
    (2019, 4, 15),
    (2015, 7, 15),
    (2019, 7, 15),
    (2015, 10, 15),
    (2019, 10, 15),
]
OUT_JSON = "/results/d_block_results.json"


def conservative_downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """x: (..., H, W) -> (..., H//factor, W//factor), box average."""
    *lead, h, w = x.shape
    x = x.reshape(*lead, h // factor, factor, w // factor, factor)
    return x.mean(axis=(-3, -1))


def block_replicate_upsample(x: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(x, factor, axis=-2), factor, axis=-1)


def null_space_projector(x: np.ndarray, factor: int) -> np.ndarray:
    return x - block_replicate_upsample(conservative_downsample(x, factor), factor)


def cov_to_corr(c: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(c), 1e-30, None))
    return c / np.outer(d, d)


def fit_kernel_corr(c_hat: np.ndarray, tau: np.ndarray, kind: str):
    """Grid-search length scale to best match the empirical correlation
    matrix -- same method as toy/_bridge_analysis_full.py's fit_kernel_corr,
    minus the "BB" (Brownian bridge) option: d has no pinning, so a bridge
    (which forces zero variance at the endpoints) is not a candidate here.
    """
    diffs = np.abs(tau[:, None] - tau[None, :])
    best = (np.inf, None)
    for ell in np.linspace(1.0, 60.0, 120):
        if kind == "OU":
            k = np.exp(-diffs / ell)
        elif kind == "RBF":
            k = np.exp(-(diffs**2) / (2 * ell**2))
        else:
            raise ValueError(kind)
        cc = cov_to_corr(k)
        err = np.linalg.norm(cc - c_hat) / np.linalg.norm(c_hat)
        if err < best[0]:
            best = (err, ell)
    return best


def load_window_d(fine_ds, coarse_ds, t0, t1, channel):
    """d(tau) for one 24h window, ALL 9 frames, global grid: (9, Hf, Wf).

    Uses each store's OWN native grid (no lat/lon nearest-neighbor
    alignment) -- coarse.sel(..., method="nearest") on a coarser grid would
    just repeat each coarse cell's value at every fine lat/lon that snaps to
    it, which is not conservative regridding and would silently break
    D@U=I's exactness.
    """
    fine = fine_ds[channel].sel(time=slice(t0, t1)).values.astype(np.float64)
    coarse = coarse_ds[channel].sel(time=slice(t0, t1)).values.astype(np.float64)
    fine_h, fine_w = fine.shape[-2:]
    coarse_h, coarse_w = coarse.shape[-2:]
    if fine_h != coarse_h * FACTOR or fine_w != coarse_w * FACTOR:
        # Crop to the largest FACTOR-compatible common domain (global grids
        # should already match by construction, but guard against off-by-one
        # boundary handling between the two stores).
        fine_h = (fine_h // FACTOR) * FACTOR
        fine_w = (fine_w // FACTOR) * FACTOR
        fine = fine[..., :fine_h, :fine_w]
        coarse = coarse[..., : fine_h // FACTOR, : fine_w // FACTOR]
    upsampled_coarse = block_replicate_upsample(coarse, FACTOR)
    return null_space_projector(fine - upsampled_coarse, FACTOR)


def characterize_channel(fine_ds, coarse_ds, channel: str) -> dict[str, Any]:
    """Collect d(tau) for every usable window, then compute all pooled
    moments directly from the stacked (n_windows, nT, H, W) array -- small
    enough to hold in memory (a handful of windows, global grid), so this
    skips the running-sufficient-statistics bookkeeping
    toy/_bridge_analysis_full.py needs for its much larger (9-year) stream.
    """
    windows: list[np.ndarray] = []
    for y, m, day in SAMPLE_DATES:
        t0 = cftime.DatetimeJulian(y, m, day, 0)
        # +24h (not 23h): the window's last frame IS the next day's 00:00 --
        # an inclusive .sel(time=slice(t0, t1)) needs t1 to land exactly on
        # that frame, not one step short of it.
        t1 = t0 + datetime.timedelta(hours=24)
        try:
            d = load_window_d(fine_ds, coarse_ds, t0, t1, channel)
        except Exception as e:  # noqa: BLE001 - a missing/short window shouldn't kill the run
            print(f"  skip {y}-{m:02d}-{day:02d}: {type(e).__name__}: {e}", flush=True)
            continue
        if d.shape[0] != N_TIMESTEPS:
            print(
                f"  skip {y}-{m:02d}-{day:02d}: got {d.shape[0]} frames, "
                f"expected {N_TIMESTEPS}",
                flush=True,
            )
            continue
        if windows and d.shape[-2:] != windows[0].shape[-2:]:
            print(f"  skip {y}-{m:02d}-{day:02d}: grid shape mismatch", flush=True)
            continue
        windows.append(d)
        print(
            f"  window {y}-{m:02d}-{day:02d} done ({len(windows)}/{len(SAMPLE_DATES)})",
            flush=True,
        )

    if not windows:
        print(f"  {channel}: no usable windows, skipping", flush=True)
        return {"degenerate": True}

    d_all = np.stack(windows, axis=0)  # (n_windows, nT, H, W)
    n_windows = d_all.shape[0]
    mean_d = d_all.mean(axis=0)  # (nT, H, W): deterministic (window-invariant) part
    fluct = d_all - mean_d  # (n_windows, nT, H, W): stochastic fluctuation

    if fluct.var() < 1e-20:
        return {"degenerate": True}

    sigma = np.einsum("kahw,kbhw->ab", fluct, fluct) / (
        n_windows * fluct.shape[-2] * fluct.shape[-1]
    )
    var_prof = np.diag(sigma).copy()
    std_prof = np.sqrt(np.clip(var_prof, 1e-30, None))
    det_rms = np.sqrt((mean_d**2).mean(axis=(-2, -1)))
    det_frac = det_rms / np.sqrt(det_rms**2 + var_prof)

    c_hat = cov_to_corr(sigma)
    e_ou, ell_ou = fit_kernel_corr(c_hat, TAU.astype(np.float64), "OU")
    e_rbf, ell_rbf = fit_kernel_corr(c_hat, TAU.astype(np.float64), "RBF")
    e_indep = float(np.linalg.norm(c_hat - np.eye(N_TIMESTEPS)) / np.linalg.norm(c_hat))

    # endpoint-vs-interior variance ratio -- the direct test of Prop 1
    # ("d is NOT pinned: Var(d(0)) should be comparable to interior").
    endpoint_var = 0.5 * (var_prof[0] + var_prof[-1])
    interior_var = var_prof[1:-1].mean()

    standardized = fluct / std_prof.reshape(1, -1, 1, 1)
    skew = float(np.mean((standardized**3).mean(axis=(0, -2, -1))))
    exkurt = float(np.mean((standardized**4).mean(axis=(0, -2, -1))) - 3.0)

    result = {
        "degenerate": False,
        "n_windows": n_windows,
        "var_prof": var_prof.tolist(),
        "std_prof": std_prof.tolist(),
        "C_hat": c_hat.tolist(),
        "det_frac": det_frac.tolist(),
        "e_indep": e_indep,
        "e_OU": float(e_ou),
        "ell_OU": float(ell_ou),
        "e_RBF": float(e_rbf),
        "ell_RBF": float(ell_rbf),
        "endpoint_var": float(endpoint_var),
        "interior_var": float(interior_var),
        "endpoint_interior_ratio": float(endpoint_var / interior_var),
        "skew": skew,
        "exkurt": exkurt,
        "units": UNITS[channel],
    }
    print(
        f"  {channel}: e_indep={e_indep:.3f} e_OU={e_ou:.3f} e_RBF={e_rbf:.3f} "
        f"endpoint/interior var ratio={endpoint_var / interior_var:.3f} "
        f"kurt={exkurt:.1f}",
        flush=True,
    )
    return result


def main():
    print(f"Opening {FINE_ZARR} and {COARSE_ZARR} ...", flush=True)
    fine_ds = xr.open_zarr(FINE_ZARR)
    coarse_ds = xr.open_zarr(COARSE_ZARR)

    results: dict[str, dict[str, Any]] = {}
    for channel in CHANNELS:
        print(
            f"{channel}: accumulating over {len(SAMPLE_DATES)} windows...", flush=True
        )
        results[channel] = characterize_channel(fine_ds, coarse_ds, channel)

    meta = {"tau": TAU.tolist(), "n_timesteps": N_TIMESTEPS, "factor": FACTOR}
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    print(f"Wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
