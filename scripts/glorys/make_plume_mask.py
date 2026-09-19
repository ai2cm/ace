"""Build the climatological river-plume mask for the GLORYS training set.

GLORYS resolves river discharge at 1/12 degree, so surface salinity at the
Rio de la Plata, the Amazon, the Congo, the Ganges-Brahmaputra and the Pearl
falls to near zero. The emulator cannot predict those cells: runoff is not one
of its inputs, so the controlling variable is absent and no amount of capacity
recovers it. They also carry outsized leverage on normalization -- dropping the
0.11% of cells below 5 psu cuts the so_0 standard deviation by 7%, and that
standard deviation sets the loss weighting of every other cell.

This writes a static mask of the cells to exclude. It is climatological by
construction: plumes migrate seasonally, so a mask taken from one state would
be wrong for most of the year. The criterion is the monthly-mean surface
salinity in *any* calendar month, so a cell is masked if it is ever fresh.

    python make_plume_mask.py gs://bucket/plume-mask.zarr \
        --start_year 2011 --end_year 2020 --threshold 20.0

Reads surface salinity only (one level, one day per month), so the cost is
roughly 35 MB per sampled month rather than the 1.8 GB a full 3-D field costs.
"""

import argparse
import importlib.util
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "glorys_pipeline", os.path.join(_HERE, "pipeline", "glorys-pipeline.py")
)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


def sample_times(start_year: int, end_year: int, day: int) -> pd.DatetimeIndex:
    """One day per calendar month across the period."""
    return pd.DatetimeIndex(
        [
            pd.Timestamp(year=y, month=m, day=day)
            for y in range(start_year, end_year + 1)
            for m in range(1, 13)
        ]
    )


def surface_salinity(times: pd.DatetimeIndex, output_grid: str, weights_path):
    """Regridded surface salinity for each sampled time, as (time, lat, lon)."""
    ocean = gp.open_cmems(gp.URL_OCEAN).rename({"latitude": "lat", "longitude": "lon"})
    surface = float(ocean[gp.VDIM].max())  # elevation is negative, deepest first
    src = gp._make_source_grid(ocean["lat"].values, ocean["lon"].values)
    frames = []
    for i, t in enumerate(times):
        da = ocean["so"].sel(time=t).sel({gp.VDIM: surface}, method="nearest").load()
        ds = gp._regrid_dataset(
            xr.Dataset({"so_0": da}), output_grid, src, weights_path
        )
        frames.append(ds["so_0"])
        if (i + 1) % 12 == 0:
            logging.info("regridded %d/%d months", i + 1, len(times))
    return xr.concat(frames, dim=pd.Index(times, name="time"))


def plume_mask(salinity: xr.DataArray, threshold: float) -> xr.Dataset:
    """1 where the cell is usable, 0 where it is river-influenced.

    A cell is masked if its climatological monthly mean falls below the
    threshold in any calendar month.
    """
    monthly = salinity.groupby("time.month").mean("time")
    lowest = monthly.min("month")
    lowest = lowest.astype(np.float32)
    # Land is NaN here, and `NaN >= threshold` is False, which would mark every
    # land cell as river-influenced. Only mask cells we actually have data for;
    # land is already excluded by mask_2d.
    mask = ((lowest >= threshold) | np.isnan(lowest)).astype(np.float32)
    mask.attrs = {
        "long_name": "river plume mask",
        "units": "0 if river-influenced, 1 if usable",
        "threshold_psu": threshold,
    }
    lowest.attrs = {
        "long_name": "lowest climatological monthly-mean surface salinity",
        "units": "psu",
    }
    out = xr.Dataset({"plume_mask": mask, "monthly_min_so_0": lowest})
    # The GLORYS source arrays are zarr v2 and carry a numcodecs Blosc filter in
    # .encoding, which xarray propagates through the computation and then
    # rejects when writing a v3 store ("Expected a BytesBytesCodec").
    for name in list(out.variables):
        out[name].encoding.clear()
    return out


def main():
    # force=True: importing the pipeline module configures logging first.
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_path", help="zarr store to write (local or gs://)")
    p.add_argument("--start_year", type=int, default=2011)
    p.add_argument("--end_year", type=int, default=2020)
    p.add_argument("--day", type=int, default=15, help="day of month to sample")
    p.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="mask cells whose lowest climatological monthly-mean surface "
        "salinity falls below this (psu)",
    )
    p.add_argument("--output_grid", default=gp.DEFAULT_OUTPUT_GRID)
    p.add_argument("--regrid_weights", default=None)
    args = p.parse_args()

    times = sample_times(args.start_year, args.end_year, args.day)
    logging.info(
        "sampling %d months, %d-%d", len(times), args.start_year, args.end_year
    )
    salinity = surface_salinity(times, args.output_grid, args.regrid_weights)
    out = plume_mask(salinity, args.threshold)
    ocean = np.isfinite(out["monthly_min_so_0"].values)
    masked = int(((out["plume_mask"].values == 0) & ocean).sum())
    finite = int(ocean.sum())
    logging.info(
        "masked %d of %d ocean cells (%.3f%%) at threshold %.1f psu",
        masked,
        finite,
        100 * masked / max(finite, 1),
        args.threshold,
    )
    out.to_zarr(gp._make_zarr_store(args.output_path, read_only=False), mode="w")
    logging.info("wrote %s", args.output_path)


if __name__ == "__main__":
    main()
