"""Two-block residual calibration + store-consistency diagnostic.

Combines two related checks that both need the same paired 25km/100km
data pass, computed in the model's own normalized/log-transformed units
(matching `VideoDiffusionModel._pack_normalized`, so the numbers below are
directly usable as config values):

1. **r/d residual std (for `r_sigma_data_by_channel`/`d_sigma_data_by_channel`)**:
   `r_sigma_data_by_channel`/`d_sigma_data_by_channel` are still at their
   default (1.0) in both two-block training configs -- this computes the
   real per-channel std of `r_target`/`d_target` (in normalized units) to
   calibrate them properly, mirroring `toy/compute_residual_std.py`'s role
   for the single-block configs.

2. **Store-consistency bias `||x_c - D(x_f)||`**: `d_target`'s docstring
   notes that `coarse_clip` and `fine_clip` are independently-sourced
   stores, not a literal `D`-downsample pair -- so a perfectly-predicting
   model's assembled output has an irreducible bias term
   `U(x_c - D(x_f))`. This computes that bias directly, per channel and
   per latitude band (the 100km store is plausibly area-weighted-regridded
   from native data, while `D` here is an unweighted box mean, which
   would diverge systematically toward the poles where a lat/lon-degree
   cell's true area shrinks). Reported relative to the r/d std from (1) so
   it's clear whether this bias is negligible or would get misdiagnosed as
   model error during eval.

`D`/`U`/`Pi` are reimplemented here in plain numpy (factor=4), same
self-contained-script convention as crps_eval.py /
endpoint_vs_interior_diagnostic.py / d_block_characterization.py.
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
N_TIMESTEPS = 9
TIME_STEP_HOURS = 3
TAU = np.arange(N_TIMESTEPS) * TIME_STEP_HOURS
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
# log1p(x * scale) transform applied before normalization, matching
# VideoDiffusionModelConfig.log_transform_channels in both two-block configs.
LOG_TRANSFORM_SCALE = {"PRATEsfc": 86400.0}
# Same normalization blocks as the two-block training configs (fine ==
# `normalization`, coarse == `coarse_normalization`) -- copied verbatim so
# this script's "normalized units" match what the model actually trains on.
FINE_NORM = {
    "eastward_wind_at_ten_meters": (-0.036200, 5.627319),
    "northward_wind_at_ten_meters": (0.170470, 4.692752),
    "PRMSL": (1008.160699, 15.043594),
    "PRATEsfc": (0.484074, 0.850964),  # log1p(PRATEsfc * 86400) space
    "air_temperature_at_two_meters": (279.675977, 20.455977),
}
COARSE_NORM = {
    "eastward_wind_at_ten_meters": (-0.036135, 5.548626),
    "northward_wind_at_ten_meters": (0.186555, 4.603089),
    "PRMSL": (1008.281106, 14.908773),
    "PRATEsfc": (0.557112, 0.871331),
    "air_temperature_at_two_meters": (279.3386, 20.2363),
}
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
N_LAT_BANDS = 9  # 20-degree bands over [-90, 90]
OUT_JSON = "/results/residual_calibration_results.json"


def conservative_downsample(x: np.ndarray, factor: int) -> np.ndarray:
    *lead, h, w = x.shape
    x = x.reshape(*lead, h // factor, factor, w // factor, factor)
    return x.mean(axis=(-3, -1))


def block_replicate_upsample(x: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(x, factor, axis=-2), factor, axis=-1)


def null_space_projector(x: np.ndarray, factor: int) -> np.ndarray:
    return x - block_replicate_upsample(conservative_downsample(x, factor), factor)


def normalize(x: np.ndarray, channel: str, norm: dict) -> np.ndarray:
    """Matches VideoDiffusionModel._pack_normalized: log1p transform (if
    configured for this channel) then standardize."""
    if channel in LOG_TRANSFORM_SCALE:
        x = np.log1p(np.clip(x, 0.0, None) * LOG_TRANSFORM_SCALE[channel])
    mean, std = norm[channel]
    return (x - mean) / std


def load_window(fine_ds, coarse_ds, t0, t1, channel):
    """(fine, coarse) for one 24h window, ALL 9 frames, native grids, RAW
    (not yet normalized) physical units."""
    fine = fine_ds[channel].sel(time=slice(t0, t1)).values.astype(np.float64)
    coarse = coarse_ds[channel].sel(time=slice(t0, t1)).values.astype(np.float64)
    fine_h, fine_w = fine.shape[-2:]
    coarse_h, coarse_w = coarse.shape[-2:]
    if fine_h != coarse_h * FACTOR or fine_w != coarse_w * FACTOR:
        fine_h = (fine_h // FACTOR) * FACTOR
        fine_w = (fine_w // FACTOR) * FACTOR
        fine = fine[..., :fine_h, :fine_w]
        coarse = coarse[..., : fine_h // FACTOR, : fine_w // FACTOR]
    return fine, coarse


def characterize_channel(
    fine_ds, coarse_ds, channel: str, coarse_lat: np.ndarray
) -> dict[str, Any]:
    r_samples = []  # normalized, coarse resolution, all 9 frames
    d_samples = []  # normalized, fine resolution, all 9 frames
    bias_samples = []  # x_c - D(x_f), coarse-normalized units, all 9 frames

    for y, m, day in SAMPLE_DATES:
        t0 = cftime.DatetimeJulian(y, m, day, 0)
        t1 = t0 + datetime.timedelta(hours=24)
        try:
            fine_raw, coarse_raw = load_window(fine_ds, coarse_ds, t0, t1, channel)
        except Exception as e:  # noqa: BLE001 - a missing/short window shouldn't kill the run
            print(f"  skip {y}-{m:02d}-{day:02d}: {type(e).__name__}: {e}", flush=True)
            continue
        if fine_raw.shape[0] != N_TIMESTEPS:
            print(f"  skip {y}-{m:02d}-{day:02d}: wrong frame count", flush=True)
            continue

        fine_n = normalize(fine_raw, channel, FINE_NORM)
        coarse_n = normalize(coarse_raw, channel, COARSE_NORM)

        # r_target: coarse residual over the coarse-endpoint linear interp,
        # in COARSE-normalized units (matches r_target(coarse_clip) in
        # fme/downscaling/twoblock.py, applied post-normalization).
        w = (TAU / TAU[-1]).reshape(-1, 1, 1)
        interp_n = (1 - w) * coarse_n[0:1] + w * coarse_n[-1:]
        r_samples.append(coarse_n - interp_n)

        # d_target: Pi(fine - U(coarse)), in FINE-normalized units. Uses
        # the SAME normalization space for both terms (fine_n and
        # upsample(coarse_n)) -- this is the model's own convention
        # (residual formed post-normalization, not pre-), which matters
        # here specifically because fine/coarse have DIFFERENT
        # normalization constants.
        upsampled_coarse_n = block_replicate_upsample(coarse_n, FACTOR)
        d_samples.append(null_space_projector(fine_n - upsampled_coarse_n, FACTOR))

        # Store-consistency bias: x_c - D(x_f), in COARSE-normalized units
        # (both terms use the SAME coarse normalization, since D(x_f) is
        # meant to approximate x_c on the coarse grid).
        downsampled_fine_n = conservative_downsample(fine_n, FACTOR)
        bias_samples.append(coarse_n - downsampled_fine_n)

        print(f"  window {y}-{m:02d}-{day:02d} done", flush=True)

    if not r_samples:
        return {"degenerate": True}

    r_all = np.stack(r_samples, axis=0)  # (n_windows, 9, Hc, Wc)
    d_all = np.stack(d_samples, axis=0)  # (n_windows, 9, Hf, Wf)
    bias_all = np.stack(bias_samples, axis=0)  # (n_windows, 9, Hc, Wc)

    r_std = float(r_all.std())
    d_std = float(d_all.std())
    bias_rms = float(np.sqrt((bias_all**2).mean()))

    # Per-latitude-band bias RMS (mean over windows/frames/lon, RMS over lat
    # within each band) -- to check for the area-weighting hypothesis (bias
    # growing toward the poles).
    lat_edges = np.linspace(-90, 90, N_LAT_BANDS + 1)
    band_labels = [
        f"{int(lat_edges[i])}..{int(lat_edges[i + 1])}" for i in range(N_LAT_BANDS)
    ]
    band_rms: list[float | None] = []
    for i in range(N_LAT_BANDS):
        band_mask = (coarse_lat >= lat_edges[i]) & (coarse_lat < lat_edges[i + 1])
        if not band_mask.any():
            band_rms.append(None)
            continue
        band_rms.append(float(np.sqrt((bias_all[..., band_mask, :] ** 2).mean())))

    result = {
        "degenerate": False,
        "n_windows": r_all.shape[0],
        "r_std": r_std,
        "d_std": d_std,
        "bias_rms": bias_rms,
        "bias_over_r_std": bias_rms / r_std,
        "bias_over_d_std": bias_rms / d_std,
        "band_labels": band_labels,
        "band_bias_rms": band_rms,
        "units": UNITS[channel],
    }
    print(
        f"  {channel}: r_std={r_std:.4f} d_std={d_std:.4f} bias_rms={bias_rms:.4f} "
        f"(bias/r_std={result['bias_over_r_std']:.3f}, "
        f"bias/d_std={result['bias_over_d_std']:.3f})",
        flush=True,
    )
    return result


def main():
    print(f"Opening {FINE_ZARR} and {COARSE_ZARR} ...", flush=True)
    fine_ds = xr.open_zarr(FINE_ZARR)
    coarse_ds = xr.open_zarr(COARSE_ZARR)
    coarse_lat = coarse_ds["latitude"].values

    results: dict[str, dict[str, Any]] = {}
    for channel in CHANNELS:
        print(
            f"{channel}: accumulating over {len(SAMPLE_DATES)} windows...", flush=True
        )
        results[channel] = characterize_channel(fine_ds, coarse_ds, channel, coarse_lat)

    meta = {"tau": TAU.tolist(), "n_timesteps": N_TIMESTEPS, "factor": FACTOR}
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    print(f"Wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
