#!/usr/bin/env python3
"""Extract instantaneous atmosphere fields onto the ocean's 5-day time axis.

The all-in-one arm needs a handful of atmosphere variables (jet-level winds,
h500, TMP850, surface pressure, DSWRFtoa, CO2) as prediction targets of a
single Samudra-style model. The 6-hourly atmosphere zarr contains every 5-day
ocean timestamp exactly (ocean stamp k = atmos index offset + stride*k), so no
time averaging is required: this is a pure strided copy of snapshots. The
alignment is computed from the two time axes at runtime and asserted, not
assumed.

Output chunking matches the ocean zarr's (time=360, full spatial) so the
training loader reads both stores with the same access pattern.
"""

import argparse

import cftime
import numpy as np
import zarr

VARS = [
    "h500",
    "TMP850",
    "eastward_wind_2",
    "northward_wind_2",
    "PRESsfc",
    "PRMSL",
    "DSWRFtoa",
    "carbon_dioxide",
]
TIME_CHUNK = 360


def decode_time(group: zarr.Group) -> np.ndarray:
    arr = group["time"]
    units = arr.attrs["units"]
    calendar = arr.attrs["calendar"]
    return np.asarray(cftime.num2date(arr[:], units=units, calendar=calendar))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atmos", required=True, help="6-hourly atmosphere zarr")
    ap.add_argument("--ocean", required=True, help="5-day ocean zarr (time reference)")
    ap.add_argument("--out", required=True, help="output zarr path")
    args = ap.parse_args()

    src = zarr.open_group(args.atmos, mode="r")
    ocean = zarr.open_group(args.ocean, mode="r")

    atmos_times = decode_time(src)
    ocean_times = decode_time(ocean)
    idx = np.searchsorted(atmos_times, ocean_times)
    if (
        not (idx < atmos_times.size).all()
        or not (atmos_times[idx] == ocean_times).all()
    ):
        raise RuntimeError("not every ocean timestamp exists in the atmosphere axis")
    strides = np.unique(np.diff(idx))
    if strides.size != 1:
        raise RuntimeError(f"non-uniform stride between ocean stamps: {strides}")
    offset, stride = int(idx[0]), int(strides[0])
    n_out = int(idx.size)
    print(f"alignment: atmos index = {offset} + {stride}*k, {n_out} output steps")

    out = zarr.open_group(args.out, mode="w")

    # coordinates: lat/lon verbatim, time from the (aligned) atmos axis
    for name in ("lat", "lon"):
        c = src[name]
        oc = out.create_array(
            name, shape=c.shape, dtype=c.dtype, dimension_names=(name,)
        )
        oc[:] = c[:]
        oc.attrs.update(dict(c.attrs))
    tsrc = src["time"]
    tout = out.create_array(
        "time", shape=(n_out,), dtype=tsrc.dtype, dimension_names=("time",)
    )
    tout[:] = tsrc[offset::stride][:n_out]
    tout.attrs.update(dict(tsrc.attrs))

    for name in VARS:
        v = src[name]
        spatial = v.shape[1:]
        chunks = (TIME_CHUNK,) + spatial
        dims = ("time",) + (("lat", "lon") if len(spatial) == 2 else ())
        ov = out.create_array(
            name,
            shape=(n_out,) + spatial,
            dtype=v.dtype,
            chunks=chunks,
            dimension_names=dims,
        )
        ov.attrs.update(dict(v.attrs))
        for start in range(0, n_out, TIME_CHUNK):
            stop = min(start + TIME_CHUNK, n_out)
            raw = slice(
                offset + start * stride, offset + (stop - 1) * stride + 1, stride
            )
            ov[start:stop] = v[raw]
        print(f"wrote {name}: {ov.shape} chunks={chunks}")

    zarr.consolidate_metadata(out.store)
    print("done; metadata consolidated")


if __name__ == "__main__":
    main()
