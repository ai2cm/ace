"""Normalization statistics for the masked, per-land-area snow channels.

Copies the parent's four stats files and adds entries under the ``_masked``
names: mean and standard deviation of the field and standard deviation of the
one-day increment, over valid (mask == 1) cells and a strided sample of
consecutive-day pairs; plus valid-domain time-mean maps in ``time-mean.nc`` so
``time_mean_reference_data`` resolves the new names. Cells are unweighted, as
in ``get_stats.py``. The fields are transformed with
``masked_snow.land_snow_fields``, the same function the store builder uses.

The statistics are accumulated as running moments (count, sum, sum of squares
per field, plus per-cell sums for the time means), so several parents can be
pooled without holding their samples: a pooled fit weights each parent by its
number of sampled cells, as ``combine_stats.py`` weights per-store stats by
sample count.

Usage:
  python fit_masked_snow_stats.py era5 [--dev]
  python fit_masked_snow_stats.py cm4  [--dev]
  python fit_masked_snow_stats.py --pool pic-1pct --parent-stats <dir-or-gs-url> [--dev]

A single parent copies that parent's stats (``Parent.stats_url``); ``--pool``
fits over the parents of a source set from ``masked_snow.SOURCE_SETS`` and
copies the pooled per-store stats given by ``--parent-stats`` (the
``combined/`` directory written by ``pool_daily_stats.py``).

Writes <masked_snow.WORK_DIR>/stats-out/<parent-name>-land-snow-masked-stats/
(single parent) or .../stats-out/cm4-<source-set>-daily-land-snow-masked-stats/
(pooled) for ``beaker dataset create``.
"""

import argparse
import dataclasses
import os
import shutil
import subprocess

import numpy as np
import xarray as xr
from masked_snow import (
    MASKED,
    PARENTS,
    SCF,
    SOURCE_SETS,
    STATS_DIR,
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


@dataclasses.dataclass
class Moments:
    """Running sums over sampled cells of one masked field and its one-day increment."""

    count: float = 0.0
    total: float = 0.0
    total_sq: float = 0.0
    diff_count: float = 0.0
    diff_total: float = 0.0
    diff_total_sq: float = 0.0
    cell_total: np.ndarray | None = None
    cell_count: np.ndarray | None = None

    def add(self, values: np.ndarray, diffs: np.ndarray) -> None:
        finite = np.isfinite(values)
        self.count += float(finite.sum())
        self.total += float(np.nansum(values))
        self.total_sq += float(np.nansum(values**2))
        diff_finite = np.isfinite(diffs)
        self.diff_count += float(diff_finite.sum())
        self.diff_total += float(np.nansum(diffs))
        self.diff_total_sq += float(np.nansum(diffs**2))
        cell_total = np.nansum(values, axis=0)
        cell_count = finite.sum(axis=0).astype(np.float64)
        if self.cell_total is None:
            self.cell_total, self.cell_count = cell_total, cell_count
        else:
            self.cell_total = self.cell_total + cell_total
            self.cell_count = self.cell_count + cell_count

    def entry(self) -> dict[str, float]:
        mean = self.total / self.count
        diff_mean = self.diff_total / self.diff_count
        return {
            "mean": mean,
            "std": float(np.sqrt(max(self.total_sq / self.count - mean**2, 0.0))),
            "residual_std": float(
                np.sqrt(max(self.diff_total_sq / self.diff_count - diff_mean**2, 0.0))
            ),
        }

    def time_mean(self, valid: np.ndarray) -> np.ndarray:
        if self.cell_total is None or self.cell_count is None:
            raise ValueError("no samples accumulated")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self.cell_total / self.cell_count
        return np.where(valid, mean, np.nan)


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


def _accumulate(
    parent: Parent, starts: np.ndarray, land_frac, valid, moments: dict[str, Moments]
) -> None:
    """Add the parent's per-land-area masked fields at the pair starts, and
    their one-day increments, to the running moments."""
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
            moments[v].add(arr[:, 0], arr[:, 1] - arr[:, 0])
        print(
            f"  {parent.name}: {min(i + BLOCK_PAIRS, len(starts))}/{len(starts)} pairs",
            flush=True,
        )


def _fetch(source: str, filename: str, local: str) -> None:
    if source.startswith("gs://"):
        subprocess.run(
            ["gsutil", "-q", "cp", f"{source}/{filename}", local], check=True
        )
    else:
        shutil.copy(os.path.join(source, filename), local)


def _patch_stats(source: str, out_dir: str, entries, time_means) -> None:
    """Copy the parent (or pooled) stats files and add the masked entries."""
    file_keys = {
        "centering.nc": "mean",
        "scaling-full-field.nc": "std",
        "scaling-residual.nc": "residual_std",
    }
    os.makedirs(out_dir, exist_ok=True)
    for filename in STATS_FILENAMES:
        local = os.path.join(out_dir, filename)
        _fetch(source, filename, local)
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


def fit(parents: list[Parent], dev: bool) -> tuple[dict, dict]:
    valid = load_mask() > 0.5
    moments = {v: Moments() for v in (SWE, SCF)}
    for parent in parents:
        ds = open_parent(parent)
        land_frac = land_fraction(ds)
        starts = _pair_starts(ds, parent, dev)
        _accumulate(parent, starts, land_frac, valid, moments)
    entries = {MASKED[v]: moments[v].entry() for v in (SWE, SCF)}
    time_means = {MASKED[v]: moments[v].time_mean(valid) for v in (SWE, SCF)}
    return entries, time_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", choices=sorted(PARENTS))
    parser.add_argument("--pool", choices=sorted(SOURCE_SETS))
    parser.add_argument(
        "--parent-stats",
        help="pooled per-store stats directory (local or gs://) to copy and extend; "
        "required with --pool",
    )
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    if (args.dataset is None) == (args.pool is None):
        parser.error("give exactly one of a dataset or --pool")
    if args.pool is not None and args.parent_stats is None:
        parser.error("--pool needs --parent-stats")

    if args.pool is None:
        parents = [PARENTS[args.dataset]]
        source = parents[0].stats_url
        out = os.path.join(STATS_DIR, f"{parents[0].output_name}-stats")
    else:
        parents = [PARENTS[key] for key in SOURCE_SETS[args.pool]]
        source = args.parent_stats
        out = os.path.join(STATS_DIR, f"cm4-{args.pool}-daily-land-snow-masked-stats")
    entries, time_means = fit(parents, args.dev)
    print({k: {s: round(x, 4) for s, x in e.items()} for k, e in entries.items()})
    _patch_stats(source, out, entries, time_means)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
