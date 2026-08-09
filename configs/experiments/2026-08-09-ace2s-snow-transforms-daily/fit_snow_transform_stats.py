"""Fit transformed-space normalization statistics for the snow variables.

For each dataset (era5, cm4) this computes, from the daily zarr:

- recipe A ("log1p"): mean/std of log1p(SWE) and logit(SCF), and the std of
  one-day differences of each, patched into copies of the source stats files.
- recipe B ("quantile"): monotone (x, z) knot tables mapping each snow
  variable through its empirical CDF onto standard normal quantiles, fit over
  land cells; patched stats hold mean 0 / std 1 and the z-space residual std.

Only the two snow entries change in the patched stats; all other variables
are byte-identical to the source files. time-mean.nc is copied unchanged.

Sampling streams consecutive-day pairs at a stride, so residual (difference)
statistics use true one-day increments. Quantile CDFs are fit on land cells
(land_fraction > 0.05); mean/std for recipe A are global-all-cells, matching
how ACE's standard stats are computed.

Usage:
  python fit_snow_transform_stats.py era5 [--dev]
  python fit_snow_transform_stats.py cm4 [--dev]

Writes to ./stats-out/<dataset>-<recipe>/ ready for `beaker dataset create`.
"""

import argparse
import dataclasses
import os
import shutil
import subprocess

import numpy as np
import xarray as xr

SWE = "surface_snow_amount"
SCF = "surface_snow_area_fraction"
LOGIT_EPSILON = 1e-4
N_KNOTS = 1000
STATS_FILENAMES = (
    "centering.nc",
    "scaling-full-field.nc",
    "scaling-residual.nc",
    "time-mean.nc",
)


@dataclasses.dataclass
class DatasetSpec:
    zarr_url: str
    stats_url: str
    start_time: str | None
    stop_time: str | None
    pair_stride_days: int
    scf_scale: float


SPECS = {
    "era5": DatasetSpec(
        zarr_url=(
            "gs://vcm-ml-intermediate/2026-08-07-era5-1deg-8layer-daily-1940-2025/"
            "2026-08-07-era5-1deg-8layer-daily-1940-2025.zarr"
        ),
        stats_url=(
            "gs://vcm-ml-intermediate/"
            "2026-08-07-era5-1deg-8layer-daily-stats-1990-2019/combined"
        ),
        start_time="1990-01-01",
        stop_time="2019-12-31",
        pair_stride_days=2,
        scf_scale=1.0,
    ),
    "cm4": DatasetSpec(
        zarr_url=(
            "gs://vcm-ml-intermediate/"
            "2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily/"
            "2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily.zarr"
        ),
        stats_url=(
            "gs://vcm-ml-intermediate/"
            "2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-stats/"
            "combined"
        ),
        start_time=None,
        stop_time=None,
        pair_stride_days=8,
        scf_scale=100.0,
    ),
}


def _gcs_token():
    from google.oauth2.credentials import Credentials

    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    ).stdout.strip()
    return Credentials(token=token)


def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0.0, None))


def _logit(x: np.ndarray, scale: float) -> np.ndarray:
    p = np.clip(x / scale, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return np.log(p / (1.0 - p))


def _load_pairs(spec: DatasetSpec, dev: bool) -> tuple[xr.Dataset, np.ndarray]:
    ds = xr.open_zarr(
        spec.zarr_url,
        consolidated=False,
        storage_options={"token": _gcs_token()},
    )
    if spec.start_time is not None:
        ds = ds.sel(time=slice(spec.start_time, spec.stop_time))
    n = ds.sizes["time"]
    starts = np.arange(0, n - 1, spec.pair_stride_days)
    if dev:
        starts = starts[:20]
    return ds, starts


def _stream(
    ds: xr.Dataset, name: str, starts: np.ndarray, block: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Return (values at pair starts, one-day increments), flattened over
    sampled times, with the full spatial field retained per sample.
    """
    values = []
    diffs = []
    for i in range(0, len(starts), block):
        chunk_starts = starts[i : i + block]
        idx = np.stack([chunk_starts, chunk_starts + 1], axis=1).ravel()
        data = ds[name].isel(time=idx).values.astype(np.float64)
        data = data.reshape(len(chunk_starts), 2, *data.shape[1:])
        values.append(data[:, 0])
        diffs.append(data[:, 1] - data[:, 0])
        done = min(i + block, len(starts))
        print(f"  {name}: {done}/{len(starts)} pairs", flush=True)
    return np.concatenate(values), np.concatenate(diffs)


def _fit_log1p_recipe(
    swe: np.ndarray,
    swe_next: np.ndarray,
    scf: np.ndarray,
    scf_next: np.ndarray,
    scf_scale: float,
) -> dict[str, dict[str, float]]:
    t_swe, t_swe_next = _log1p(swe), _log1p(swe_next)
    t_scf, t_scf_next = _logit(scf, scf_scale), _logit(scf_next, scf_scale)
    return {
        SWE: {
            "mean": float(t_swe.mean()),
            "std": float(t_swe.std()),
            "residual_std": float((t_swe_next - t_swe).std()),
        },
        SCF: {
            "mean": float(t_scf.mean()),
            "std": float(t_scf.std()),
            "residual_std": float((t_scf_next - t_scf).std()),
        },
    }


def _fit_quantile_knots(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit (x, z) knots mapping the empirical CDF onto normal quantiles.

    Ties in x (e.g. the zero atom) collapse to a single knot at the midpoint
    of their rank mass, so the knot table is strictly increasing in x.
    """
    from scipy.stats import norm

    probs = (np.arange(N_KNOTS) + 0.5) / N_KNOTS
    x = np.quantile(sample, probs)
    knots_x = []
    knots_z = []
    for value in np.unique(x):
        mask = x == value
        mid_prob = probs[mask].mean()
        knots_x.append(float(value))
        knots_z.append(float(norm.ppf(mid_prob)))
    return np.array(knots_x), np.array(knots_z)


def _z_transform(x: np.ndarray, knots_x: np.ndarray, knots_z: np.ndarray):
    return np.interp(x, knots_x, knots_z)


def _patch_stats(
    source_dir: str,
    out_dir: str,
    entries: dict[str, dict[str, float]],
) -> None:
    """Copy the four stats files, replacing only the snow entries."""
    file_keys = {
        "centering.nc": "mean",
        "scaling-full-field.nc": "std",
        "scaling-residual.nc": "residual_std",
    }
    os.makedirs(out_dir, exist_ok=True)
    for filename in STATS_FILENAMES:
        local = os.path.join(out_dir, filename)
        subprocess.run(
            ["gsutil", "-q", "cp", f"{source_dir}/{filename}", local], check=True
        )
        if filename not in file_keys:
            continue
        ds = xr.load_dataset(local)
        for name, stats in entries.items():
            ds[name] = xr.DataArray(np.float32(stats[file_keys[filename]]))
        tmp = local + ".tmp"
        ds.to_netcdf(tmp)
        shutil.move(tmp, local)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(SPECS))
    parser.add_argument("--dev", action="store_true", help="tiny sample, no upload")
    args = parser.parse_args()
    spec = SPECS[args.dataset]

    ds, starts = _load_pairs(spec, args.dev)
    land = ds["land_fraction"].values > 0.05

    swe, swe_diff = _stream(ds, SWE, starts)
    scf, scf_diff = _stream(ds, SCF, starts)
    swe_next, scf_next = swe + swe_diff, scf + scf_diff

    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats-out")

    print("fitting recipe A (log1p/logit)")
    entries_a = _fit_log1p_recipe(swe, swe_next, scf, scf_next, spec.scf_scale)
    print({k: {s: round(v, 4) for s, v in e.items()} for k, e in entries_a.items()})
    _patch_stats(
        spec.stats_url, os.path.join(out_root, f"{args.dataset}-log1p"), entries_a
    )

    print("fitting recipe B (gaussian rank, land cells)")
    entries_b = {}
    quantile_dir = os.path.join(out_root, f"{args.dataset}-quantile")
    os.makedirs(quantile_dir, exist_ok=True)
    for name, sample, sample_next in (
        (SWE, swe, swe_next),
        (SCF, scf, scf_next),
    ):
        knots_x, knots_z = _fit_quantile_knots(sample[:, land].ravel())
        z = _z_transform(sample[:, land], knots_x, knots_z)
        z_next = _z_transform(sample_next[:, land], knots_x, knots_z)
        entries_b[name] = {
            "mean": 0.0,
            "std": 1.0,
            "residual_std": float((z_next - z).std()),
        }
        table = xr.Dataset(
            {
                "x_knots": ("knot", knots_x),
                "z_knots": ("knot", knots_z),
            }
        )
        table.to_netcdf(os.path.join(quantile_dir, f"quantile-{name}.nc"))
        print(
            f"  {name}: {len(knots_x)} knots, residual_std(z) = "
            f"{entries_b[name]['residual_std']:.4f}"
        )
    _patch_stats(spec.stats_url, quantile_dir, entries_b)

    print(f"done; outputs under {out_root}/{args.dataset}-{{log1p,quantile}}")


if __name__ == "__main__":
    main()
