"""Band-variance criterion for the 4-degree stochastic-Samudra screen.

Reads one 200-year piControl rollout's monthly files from a mounted beaker
result dataset and writes a few KB of reduced statistics. The monthly files are
3.4 GiB of predictions plus 3.9 GiB of target, of which SST is about 1/87th, so
this runs where the data already is and only the reduction travels.

Method follows the 1-degree diagnosis
(ai2cm/reports elynnwu/2026-08-26-ocean-internal-variability-damping,
scripts/common.py) exactly: deseasonalize by removing the 12-month climatology,
remove a linear trend, Hann-taper, periodogram, and integrate power over the
2-8 yr and 8-30 yr period bands. Ratios are prediction over target.

NOTE ON THE COAST-DISTANCE AND BOUNDARY-CURRENT DEFINITIONS. The 1-degree
numbers the screen pre-registered are not reproducible: neither the diagnosis
investigation nor its report PR contains any coast-distance or
boundary-current analysis, and no committed script computes one. The bin edges
and box bounds below are therefore defined HERE, and the comparison that means
anything is within this script's own output -- the stochastic arms against the
deterministic control, all measured identically -- not against the 1-degree
tables.
"""

import argparse
import json
import pathlib

import numpy as np
import xarray as xr

MONTHLY_DT = 1.0 / 12.0
BANDS = {"interannual": (2.0, 8.0), "decadal": (8.0, 30.0)}
# distance from land in grid cells; the last bin is open-ended
COAST_BIN_EDGES = [1, 2, 4, 8, 16, 32, np.inf]
# (lat_lo, lat_hi, lon_lo, lon_hi) in degrees, lon in [0, 360)
BOXES = {
    "kuroshio_extension": (30, 40, 140, 180),
    "gulf_stream": (35, 45, 285, 315),
    "n_atlantic_subpolar": (50, 65, 300, 340),
    "acc": (-60, -45, 0, 360),
    "n_pacific_gyre": (25, 40, 180, 220),
    "e_equatorial_pacific": (-5, 5, 210, 270),
}


def _find_dim(dims, candidates):
    for c in candidates:
        if c in dims:
            return c
    raise SystemExit(f"no dim among {candidates} in {dims}")


def deseason(x, period=12):
    """Remove the mean annual cycle along the leading (time) axis."""
    n = x.shape[0]
    if n % period:
        raise ValueError(f"time length {n} is not a whole number of years")
    clim = np.nanmean(x.reshape(n // period, period, *x.shape[1:]), axis=0)
    return x - np.tile(clim, (n // period,) + (1,) * (x.ndim - 1))


def band_power(x, dt=MONTHLY_DT):
    """Periodogram power per band, computed along the leading axis.

    Detrends first: drift otherwise leaks into the low-frequency band, which is
    the decadal band this is trying to measure.
    """
    n = x.shape[0]
    t = np.arange(n, dtype=float)
    flat = x.reshape(n, -1)
    finite = np.isfinite(flat)
    filled = np.where(finite, flat, 0.0)
    # least-squares linear fit per column, then subtract it
    design = np.stack([t, np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(design, filled, rcond=None)
    detrended = filled - design @ coef
    detrended[~finite] = 0.0
    power = np.abs(np.fft.rfft(detrended * np.hanning(n)[:, None], axis=0)) ** 2
    freq = np.fft.rfftfreq(n, d=dt)
    out = {}
    for name, (p_lo, p_hi) in BANDS.items():
        m = (freq >= 1.0 / p_hi) & (freq <= 1.0 / p_lo)
        out[name] = power[m].sum(axis=0).reshape(x.shape[1:])
    return out


def coast_distance(ocean):
    """Distance from land in grid cells, by repeated 4-neighbour dilation of the
    land mask. Longitude wraps; latitude does not."""
    dist = np.full(ocean.shape, np.inf)
    frontier = ~ocean
    d = 0
    while frontier.any() and d < 200:
        d += 1
        grown = (
            np.roll(frontier, 1, axis=1)
            | np.roll(frontier, -1, axis=1)
            | np.pad(frontier, ((1, 0), (0, 0)))[:-1]
            | np.pad(frontier, ((0, 1), (0, 0)))[1:]
        )
        new = grown & ~frontier & ocean & ~np.isfinite(dist)
        newly = grown & ocean & (dist == np.inf)
        dist[newly] = d
        if not newly.any():
            break
        frontier = frontier | newly
    return dist


def area_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/results/band_variance.json")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    data = pathlib.Path(args.data)
    pred = xr.open_dataset(data / "monthly_mean_predictions.nc")
    targ = xr.open_dataset(data / "monthly_mean_target.nc")

    # Print the layout before touching it: a failed run then carries everything
    # needed to fix it, which matters when each attempt costs ~20 min of
    # dataset attach.
    print("dims:", dict(pred.sizes), flush=True)
    print("sst dims:", pred["sst"].dims if "sst" in pred else "ABSENT", flush=True)
    print("coords:", list(pred.coords), flush=True)
    if "sst" not in pred:
        raise SystemExit(f"sst absent; monthly file has {list(pred.data_vars)[:20]}")

    latname = _find_dim(pred["sst"].dims, ("lat", "latitude", "grid_yt", "y"))
    lonname = _find_dim(pred["sst"].dims, ("lon", "longitude", "grid_xt", "x"))
    # Drop any length-1 leading axis (the evaluator writes one IC per file, so
    # the sample dimension is present but degenerate) and take the time axis as
    # the longest remaining non-horizontal dim rather than the first.
    da_p = pred["sst"].squeeze(drop=True)
    da_t = targ["sst"].squeeze(drop=True)
    rest = [d for d in da_p.dims if d not in (latname, lonname)]
    if len(rest) != 1:
        raise SystemExit(f"expected one time-like dim after squeeze, got {rest} "
                         f"from {da_p.dims}")
    tdim = rest[0]
    n_time = da_p.sizes[tdim]
    if n_time != 2400:
        raise SystemExit(f"expected 2400 months on {tdim!r}, got {n_time}; "
                         f"full dims {dict(da_p.sizes)}")
    print(f"sst ok: {n_time} months on {tdim!r}, grid {latname}/{lonname}", flush=True)

    lat = pred[latname].values
    lon = pred[lonname].values
    p = da_p.transpose(tdim, latname, lonname).values.astype("float64")
    t = da_t.transpose(tdim, latname, lonname).values.astype("float64")

    ocean = np.isfinite(t).all(axis=0)
    print(f"ocean cells {ocean.sum()} of {ocean.size}", flush=True)
    dist = coast_distance(ocean)

    bp_p = band_power(deseason(p))
    bp_t = band_power(deseason(t))

    w2d = np.outer(area_weights(lat), np.ones_like(lon))
    result = {"label": args.label, "n_time": int(n_time),
              "n_ocean_cells": int(ocean.sum())}

    for band in BANDS:
        # coast-distance curve: area-weighted ratio of summed band power
        curve = []
        for lo, hi in zip(COAST_BIN_EDGES[:-1], COAST_BIN_EDGES[1:]):
            m = ocean & (dist >= lo) & (dist < hi)
            if m.sum() == 0:
                curve.append(None)
                continue
            num = float((bp_p[band][m] * w2d[m]).sum())
            den = float((bp_t[band][m] * w2d[m]).sum())
            curve.append({"lo": lo, "hi": None if hi == np.inf else hi,
                          "ratio": num / den, "n_cells": int(m.sum())})
        result[f"coast_curve_{band}"] = curve

        boxes = {}
        for name, (la, lb, lo_, hi_) in BOXES.items():
            latm = (lat >= la) & (lat <= lb)
            lonm = ((lon % 360) >= lo_) & ((lon % 360) <= hi_)
            m = ocean & np.outer(latm, lonm)
            if m.sum() == 0:
                boxes[name] = None
                continue
            num = float((bp_p[band][m] * w2d[m]).sum())
            den = float((bp_t[band][m] * w2d[m]).sum())
            boxes[name] = {"ratio": num / den, "n_cells": int(m.sum())}
        result[f"boxes_{band}"] = boxes

        # global area-weighted ratio, for context against the wandb scalars
        m = ocean
        result[f"global_{band}"] = float((bp_p[band][m] * w2d[m]).sum()
                                         / (bp_t[band][m] * w2d[m]).sum())

    # guard rails, same reduction on the fields the task named
    for var, key in [("zos", "zos"), ("ocean_heat_content", "ohc"),
                     ("thetao_18", "thetao_deep")]:
        if var not in pred:
            continue
        pv = (pred[var].squeeze(drop=True)
              .transpose(tdim, latname, lonname).values.astype("float64"))
        tv = (targ[var].squeeze(drop=True)
              .transpose(tdim, latname, lonname).values.astype("float64"))
        bpp, bpt = band_power(deseason(pv)), band_power(deseason(tv))
        om = np.isfinite(tv).all(axis=0)
        for band in BANDS:
            result[f"guard_{key}_{band}"] = float(
                (bpp[band][om] * w2d[om]).sum() / (bpt[band][om] * w2d[om]).sum())

    # Arctic SST north of 75N, the third guard rail
    am = ocean & np.outer(lat >= 75, np.ones_like(lon, dtype=bool))
    if am.sum():
        for band in BANDS:
            result[f"guard_arctic_sst_{band}"] = float(
                (bp_p[band][am] * w2d[am]).sum() / (bp_t[band][am] * w2d[am]).sum())

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
