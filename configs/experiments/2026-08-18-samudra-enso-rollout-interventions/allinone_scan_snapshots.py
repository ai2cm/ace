#!/usr/bin/env python3
"""Scan the all-in-one snapshot zarr for NaN / fill values, per variable."""

import sys

import numpy as np
import zarr

g = zarr.open_group(sys.argv[1], mode="r")
for name in sorted(g.array_keys()):
    a = g[name]
    if name in ("lat", "lon", "time"):
        continue
    n_nan = 0
    vmin, vmax = np.inf, -np.inf
    step = 1000
    if a.ndim == 0 or "time" not in (a.metadata.dimension_names or ()):
        x = a[:]
        n_nan = int(np.isnan(x).sum())
        vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
        print(f"{name:20s} static nan={n_nan} min={vmin:.6g} max={vmax:.6g}")
        continue
    worst_t = -1
    for s in range(0, a.shape[0], step):
        x = a[s : s + step]
        bad = np.isnan(x)
        if bad.any() and worst_t < 0:
            worst_t = s + int(np.argwhere(bad)[0][0])
        n_nan += int(bad.sum())
        vmin = min(vmin, float(np.nanmin(x)))
        vmax = max(vmax, float(np.nanmax(x)))
    print(
        f"{name:20s} nan={n_nan} first_nan_t={worst_t} "
        f"min={vmin:.6g} max={vmax:.6g}"
    )
