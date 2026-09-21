"""Normalization statistics for the masked, per-land-area snow channels.

Copies the parent's four stats files and adds entries under the ``_masked``
names: mean and standard deviation of the field and standard deviation of the
one-day increment, over valid (mask == 1) cells and a strided sample of
consecutive-day pairs; plus valid-domain time-mean maps in ``time-mean.nc`` so
``time_mean_reference_data`` resolves the new names. Cells are unweighted, as
in ``get_stats.py``. The fields are transformed with
``masked_snow.land_snow_fields``, the same function the store builder uses.

Usage:
  python fit_masked_snow_stats.py era5 [--dev]
  python fit_masked_snow_stats.py cm4  [--dev]

Writes ./stats-out/<parent-name>-land-snow-masked-stats/ for
``beaker dataset create``.
"""

import argparse
import os
import shutil
import subprocess

import numpy as np
import xarray as xr
from masked_snow import (
    MASKED,
    PARENTS,
    SCF,
    STATS_FILENAMES,
    SWE,
    Parent,
    land_fraction,
    land_snow_fields,
    load_mask,
    open_parent,
)

BLOCK_PAIRS = 200
REOPEN_EVERY_PAIRS = 1500


def _pair_starts(ds: xr.Dataset, parent: Parent, dev: bool) -> np.ndarray:
    if parent.stats_start is not None:
        ds = ds.sel(time=slice(parent.stats_start, parent.stats_stop))
    n = ds.sizes["time"]
    starts = np.arange(0, n - 1, parent.stats_pair_stride_days)
    return starts[:20] if dev else starts


def _subset(ds: xr.Dataset, parent: Parent) -> xr.Dataset:
    if parent.stats_start is None:
        return ds
    return ds.sel(time=slice(parent.stats_start, parent.stats_stop))


def _stream(parent: Parent, starts: np.ndarray, land_frac, valid):
    """Per-land-area masked fields at the pair starts and their one-day
    increments: dict name -> (values [n, lat, lon], diffs [n, lat, lon])."""
    values: dict[str, list[np.ndarray]] = {v: [] for v in (SWE, SCF)}
    diffs: dict[str, list[np.ndarray]] = {v: [] for v in (SWE, SCF)}
    ds = _subset(open_parent(parent), parent)
    for i in range(0, len(starts), BLOCK_PAIRS):
        if i > 0 and i % REOPEN_EVERY_PAIRS == 0:
            ds = _subset(open_parent(parent), parent)
        chunk = starts[i : i + BLOCK_PAIRS]
        idx = np.stack([chunk, chunk + 1], axis=1).ravel()
        block = ds[[SWE, SCF]].isel(time=idx).load()
        swe, scf = land_snow_fields(
            block[SWE].values, block[SCF].values, parent, land_frac, valid
        )
        for v, arr in ((SWE, swe), (SCF, scf)):
            arr = arr.reshape(len(chunk), 2, *arr.shape[1:]).astype(np.float64)
            values[v].append(arr[:, 0])
            diffs[v].append(arr[:, 1] - arr[:, 0])
        print(f"  {min(i + BLOCK_PAIRS, len(starts))}/{len(starts)} pairs", flush=True)
    return {
        v: (np.concatenate(values[v]), np.concatenate(diffs[v])) for v in (SWE, SCF)
    }


def _entry(values: np.ndarray, diffs: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "residual_std": float(np.nanstd(diffs)),
    }


def _patch_stats(source_url: str, out_dir: str, entries, time_means) -> None:
    """Copy the parent's stats files and add the masked entries."""
    file_keys = {
        "centering.nc": "mean",
        "scaling-full-field.nc": "std",
        "scaling-residual.nc": "residual_std",
    }
    os.makedirs(out_dir, exist_ok=True)
    for filename in STATS_FILENAMES:
        local = os.path.join(out_dir, filename)
        subprocess.run(
            ["gsutil", "-q", "cp", f"{source_url}/{filename}", local], check=True
        )
        ds = xr.load_dataset(local)
        if filename in file_keys:
            for name, stats in entries.items():
                ds[name] = xr.DataArray(np.float32(stats[file_keys[filename]]))
        else:
            dims = ds[SWE].dims
            for name, values in time_means.items():
                ds[name] = xr.DataArray(values.astype(np.float32), dims=dims)
        tmp = local + ".tmp"
        ds.to_netcdf(tmp)
        shutil.move(tmp, local)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(PARENTS))
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    parent = PARENTS[args.dataset]

    valid = load_mask() > 0.5
    ds = open_parent(parent)
    land_frac = land_fraction(ds)
    starts = _pair_starts(ds, parent, args.dev)
    streamed = _stream(parent, starts, land_frac, valid)

    entries = {MASKED[v]: _entry(*streamed[v]) for v in (SWE, SCF)}
    time_means = {
        MASKED[v]: np.where(valid, np.nanmean(streamed[v][0], axis=0), np.nan)
        for v in (SWE, SCF)
    }
    print({k: {s: round(x, 4) for s, x in e.items()} for k, e in entries.items()})
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "stats-out", f"{parent.output_name}-stats")
    _patch_stats(parent.stats_url, out, entries, time_means)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
