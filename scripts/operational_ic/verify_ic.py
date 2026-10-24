"""Regression check of an initial condition against the training stores.

For an issue day inside the training record this compares every variable of
the ocean IC with the GLO12 training store state at that stamp (the pipeline
reuses that store's code and statics, so the match must be at float32 noise)
and every variable of the atmosphere IC with the reference daily ERA5 store's
sample (the final-ERA5 path must match at float32 noise; the ERA5T path is
expected within the spectral-transform error, reported for inspection).

    python verify_ic.py --date 2025-12-24 --ic-dir /path/to/ic
"""

import argparse
import json
import logging
import pathlib

import numpy as np
import pandas as pd
import xarray as xr
from common import REFERENCE_ATMOS_STORE, REFERENCE_OCEAN_STORE, open_zarr

log = logging.getLogger("operational_ic.verify")


def compare(ic: xr.Dataset, ref: xr.Dataset, skip: set[str]) -> dict:
    rows: dict[str, dict] = {}
    for v in ic.data_vars:
        if v in skip or v not in ref:
            continue
        a = ic[v]
        b = ref[v]
        if "time" in a.dims:
            a = a.isel(time=0)
        if "time" in b.dims:
            b = b.isel(time=0)
        a = np.asarray(a.values, dtype=np.float64)
        b = np.asarray(b.values, dtype=np.float64)
        if a.shape != b.shape:
            rows[v] = {"status": f"shape {a.shape} vs {b.shape}"}
            continue
        fa, fb = np.isfinite(a), np.isfinite(b)
        both = fa & fb
        d = a[both] - b[both]
        std = float(np.nanstd(b)) if b.size > 1 else 0.0
        rows[v] = {
            "n": int(both.sum()),
            "nan_mismatch": int((fa != fb).sum()),
            "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
            "rmse": float(np.sqrt(np.mean(d**2))) if d.size else 0.0,
            "ref_std": std,
            "rel_rmse": float(np.sqrt(np.mean(d**2)) / std)
            if d.size and std > 0
            else 0.0,
        }
    return rows


def summarize(rows: dict, tol_rel: float, label: str) -> bool:
    bad = []
    for v, r in sorted(rows.items()):
        if "status" in r:
            bad.append((v, r["status"]))
            continue
        if r["nan_mismatch"] or r["rel_rmse"] > tol_rel:
            bad.append(
                (
                    v,
                    f"rel_rmse {r['rel_rmse']:.2e} max_abs {r['max_abs']:.3g} "
                    f"nan_mismatch {r['nan_mismatch']}",
                )
            )
    worst = sorted(rows.items(), key=lambda kv: -kv[1].get("rel_rmse", 0))[:8]
    print(
        f"\n{label}: {len(rows)} variables compared, {len(bad)} outside "
        f"tolerance (rel rmse > {tol_rel:g})"
    )
    print("  worst relative RMSE:")
    for v, r in worst:
        if "status" not in r:
            print(
                f"    {v:36s} rel_rmse {r['rel_rmse']:.2e}  rmse {r['rmse']:.3g}  "
                f"max_abs {r['max_abs']:.3g}  nan_mismatch {r['nan_mismatch']}"
            )
    for v, why in bad[:20]:
        print(f"  FAIL {v}: {why}")
    return not bad


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--date", required=True)
    p.add_argument("--ic-dir", required=True)
    p.add_argument("--reference-ocean-store", default=REFERENCE_OCEAN_STORE)
    p.add_argument("--reference-atmos-store", default=REFERENCE_ATMOS_STORE)
    p.add_argument(
        "--ocean-tol",
        type=float,
        default=1e-4,
        help="relative RMSE tolerance for the ocean (float32 noise)",
    )
    p.add_argument(
        "--atmos-tol",
        type=float,
        default=1e-3,
        help="relative RMSE tolerance for the atmosphere",
    )
    p.add_argument("--json", default=None, help="write the per-variable table here")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    day = pd.Timestamp(args.date)
    d = str(day.date())
    ic_dir = pathlib.Path(args.ic_dir)
    result: dict = {}
    ok = True

    ocean_path = ic_dir / f"ocean_ic_{d}.nc"
    if ocean_path.exists():
        ic = xr.open_dataset(ocean_path).load()
        ref = open_zarr(args.reference_ocean_store)
        if day not in pd.DatetimeIndex(ref.time.values):
            print(f"ocean: {d} is not in {args.reference_ocean_store}; skipping")
        else:
            ref = ref.sel(time=[day]).load()
            # placeholders are not in the GLORYS pipeline's source store
            skip = {
                "sfdsi",
                "UI",
                "VI",
                "hfds_total_area",
                "tauuo",
                "tauvo",
                "wfo",
                "unaccounted_heating",
            }
            if ic.attrs.get("initial_condition", "").startswith("mean"):
                print(
                    "ocean IC is a 5-day mean: comparison vs the day state is "
                    "informational"
                )
            rows = compare(ic, ref, skip)
            result["ocean"] = rows
            ok &= summarize(rows, args.ocean_tol, f"ocean IC {d} vs training store")

    atmos_path = ic_dir / f"atmos_ic_{d}.nc"
    if atmos_path.exists():
        ic = xr.open_dataset(atmos_path).load()
        ref = open_zarr(args.reference_atmos_store)
        stamp = day + pd.Timedelta(days=1)  # the daily store's window-end stamp
        if stamp not in pd.DatetimeIndex(ref.time.values):
            print(
                f"atmosphere: {stamp} is not in {args.reference_atmos_store}; skipping"
            )
        else:
            ref = ref.sel(time=[stamp]).load()
            ref = ref.rename(
                {
                    k: v
                    for k, v in {"latitude": "lat", "longitude": "lon"}.items()
                    if k in ref.dims
                }
            )
            skip = {
                "land_fraction",
                "sea_surface_fraction",
                "sea_ice_fraction",
                "ocean_fraction",
            }  # GLO12's, not ERA5's
            rows = compare(ic, ref, skip)
            result["atmosphere"] = rows
            src = ic.attrs.get("source", "")
            tol = args.atmos_tol if "(final)" in src else 1.0
            label = f"atmosphere IC {d} ({src.split(' processed')[0]}) vs daily store"
            ok &= summarize(rows, tol, label)
            if "(era5t)" in src:
                print(
                    "  (ERA5T path: differences are the spectral/N320 transform "
                    "error; inspect, no pass/fail)"
                )
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, indent=1))
    print("\nVERIFY", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
