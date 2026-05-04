"""Convert CMIP6 zarr datasets to netCDF for faster local training.

NetCDF datasets can use fork-based data workers (no forkserver overhead),
which significantly improves training throughput on local storage.

Usage:
    python zarr_to_netcdf.py <input_dir> <output_dir> [--workers N]

Example:
    python zarr_to_netcdf.py ./data/cmip6-daily-pilot/v0 ./data/cmip6-daily-pilot/v0-nc
"""

import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def convert_one(zarr_path: str, nc_dir: str) -> str:
    """Convert a single zarr store to yearly netCDF files with 1-day chunks."""
    os.makedirs(nc_dir, exist_ok=True)
    ds = xr.open_zarr(zarr_path)
    ds.load()
    encoding = {
        name: {"chunksizes": (1,) + ds[name].shape[1:]}
        for name in ds.data_vars
        if "time" in ds[name].dims
    }
    for year, yearly_ds in ds.groupby("time.year"):
        nc_path = os.path.join(nc_dir, f"data.{year}.nc")
        if os.path.exists(nc_path):
            continue
        yearly_ds.to_netcdf(nc_path, encoding=encoding)
    ds.close()
    return f"ok: {nc_dir}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir", help="Root of zarr dataset tree (contains index.csv)"
    )
    parser.add_argument("output_dir", help="Root of netCDF output tree")
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel conversion workers"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(input_dir, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index.csv found at {index_path}")
    idx = pd.read_csv(index_path)

    for f in os.listdir(input_dir):
        src = os.path.join(input_dir, f)
        dst = os.path.join(output_dir, f)
        if os.path.isfile(src) and not os.path.exists(dst):
            logger.info("Copying %s -> %s", f, dst)
            shutil.copy2(src, dst)

    tasks = []
    for _, row in idx.iterrows():
        rel = os.path.join(row["source_id"], row["experiment"], row["variant_label"])
        zarr_path = os.path.join(input_dir, rel, "data.zarr")
        nc_dir = os.path.join(output_dir, rel)
        if not os.path.isdir(zarr_path):
            logger.warning("Zarr not found, skipping: %s", zarr_path)
            continue
        tasks.append((zarr_path, nc_dir))

    logger.info("Converting %d datasets with %d workers", len(tasks), args.workers)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, zp, nd): zp for zp, nd in tasks}
        for future in as_completed(futures):
            done += 1
            try:
                msg = future.result()
            except Exception as e:
                msg = f"FAILED: {futures[future]}: {e}"
            logger.info("[%d/%d] %s", done, len(tasks), msg)

    logger.info("Done. Output at %s", output_dir)


if __name__ == "__main__":
    main()
