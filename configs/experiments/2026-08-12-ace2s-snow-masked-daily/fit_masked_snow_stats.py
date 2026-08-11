"""Fit normalization statistics for the MASKED snow channels.

Reuses the streaming/patching machinery of the 2026-08-09 round's
fit_snow_transform_stats.py. For each dataset, applies the shared snow mask
(NaN outside the valid domain) to the streamed sample and fits two recipes:

- masked-naive: mean/std/residual-std of raw SWE and SCF over valid cells
- masked-log1p: log1p(SWE) and logit(SCF) stats over valid cells

Stats entries are keyed under the ``_masked`` variable names (normalizers and
the loss look variables up by name). The masked channels' valid-domain
time-mean maps are also written into the patched time-mean.nc so
``time_mean_reference_data`` resolves the new names.

Usage:
  python fit_masked_snow_stats.py era5 [--dev]
  python fit_masked_snow_stats.py cm4 [--dev]

Writes to ./stats-out/<dataset>-masked-{naive,log1p}/ for `beaker dataset create`.
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "2026-08-09-ace2s-snow-transforms-daily"))
from fit_snow_transform_stats import (  # noqa: E402
    SCF,
    SPECS,
    SWE,
    _load_pairs,
    _log1p,
    _logit,
    _patch_stats,
    _stream,
)


def _nan_entry(values: "np.ndarray", diffs: "np.ndarray") -> dict:
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "residual_std": float(np.nanstd(diffs)),
    }


def _masked_names(entries):
    return {f"{name}_masked": stats for name, stats in entries.items()}


def _patch_time_mean(out_dir: str, time_means: dict[str, np.ndarray]) -> None:
    path = os.path.join(out_dir, "time-mean.nc")
    ds = xr.load_dataset(path)
    dims = ds[SWE].dims
    for name, values in time_means.items():
        ds[name] = xr.DataArray(values.astype(np.float32), dims=dims)
    tmp = path + ".tmp"
    ds.to_netcdf(tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(SPECS))
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    spec = SPECS[args.dataset]

    mask = xr.load_dataset(f"{HERE}/snow_mask.nc")["mask"].values
    valid = mask > 0.5

    ds, starts = _load_pairs(spec, args.dev)
    swe, swe_diff = _stream(ds, SWE, starts)
    scf, scf_diff = _stream(ds, SCF, starts)
    for arr in (swe, swe_diff, scf, scf_diff):
        arr[:, ~valid] = np.nan
    swe_next, scf_next = swe + swe_diff, scf + scf_diff

    time_means = {
        f"{SWE}_masked": np.where(valid, np.nanmean(swe, axis=0), np.nan),
        f"{SCF}_masked": np.where(valid, np.nanmean(scf, axis=0), np.nan),
    }
    out_root = os.path.join(HERE, "stats-out")

    print("fitting masked-naive")
    entries_naive = {SWE: _nan_entry(swe, swe_diff), SCF: _nan_entry(scf, scf_diff)}
    entries_naive = _masked_names(entries_naive)
    print({k: {s: round(v, 4) for s, v in e.items()} for k, e in entries_naive.items()})
    out = os.path.join(out_root, f"{args.dataset}-masked-naive")
    _patch_stats(spec.stats_url, out, entries_naive)
    _patch_time_mean(out, time_means)

    print("fitting masked-log1p")
    entries_log1p = _masked_names(
        {
            SWE: _nan_entry(_log1p(swe), _log1p(swe_next) - _log1p(swe)),
            SCF: _nan_entry(
                _logit(scf, spec.scf_scale),
                _logit(scf_next, spec.scf_scale) - _logit(scf, spec.scf_scale),
            ),
        }
    )
    print({k: {s: round(v, 4) for s, v in e.items()} for k, e in entries_log1p.items()})
    out = os.path.join(out_root, f"{args.dataset}-masked-log1p")
    _patch_stats(spec.stats_url, out, entries_log1p)
    _patch_time_mean(out, time_means)

    print(f"done; outputs under {out_root}/{args.dataset}-masked-{{naive,log1p}}")


if __name__ == "__main__":
    main()
