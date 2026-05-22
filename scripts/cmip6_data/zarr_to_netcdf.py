"""Convert CMIP6 zarr datasets to netCDF for faster local training.

NetCDF datasets can use fork-based data workers (no forkserver overhead),
which significantly improves training throughput on local storage.

``input_dir`` can be a local path or a ``gs://`` URL. With a GCS source
the zarr stores stream straight into the per-dataset converter — no
intermediate local copy of the zarr tree is required, only the netCDF
output. Auxiliary top-level files (``index.csv``, ``stats.csv``,
``stats.parquet``, ``presence.*``, etc.) are downloaded next to the
netCDF tree so the local layout matches what training code expects.

Usage:
    python zarr_to_netcdf.py <input_dir> <output_dir> [--workers N]

Example:
    python zarr_to_netcdf.py ./data/cmip6-daily-pilot/v0 ./data/cmip6-daily-pilot/v0-nc
    python zarr_to_netcdf.py gs://bucket/proj/v0 /climate-default/proj/v0
"""

import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import fsspec
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

    input_dir = args.input_dir.rstrip("/")
    output_dir = args.output_dir.rstrip("/")
    os.makedirs(output_dir, exist_ok=True)

    fs, in_root = fsspec.core.url_to_fs(input_dir)

    index_url = f"{input_dir}/index.csv"
    if not fs.exists(f"{in_root}/index.csv"):
        raise FileNotFoundError(f"No index.csv found at {index_url}")
    idx = pd.read_csv(index_url)

    # Auxiliary top-level files (index.csv, stats.csv, stats.parquet,
    # presence.*, etc.). ``fs.ls(detail=True)`` lets us skip the dataset
    # sub-directories without a second probe.
    for entry in fs.ls(in_root, detail=True):
        if entry.get("type") != "file":
            continue
        name = os.path.basename(entry["name"])
        dst = os.path.join(output_dir, name)
        if os.path.exists(dst):
            continue
        logger.info("Copying %s -> %s", name, dst)
        if input_dir.startswith(("gs://", "s3://", "http://", "https://")):
            fs.get(entry["name"], dst)
        else:
            shutil.copy2(entry["name"], dst)

    tasks = []
    for _, row in idx.iterrows():
        rel = os.path.join(row["source_id"], row["experiment"], row["variant_label"])
        zarr_path = f"{input_dir}/{rel}/data.zarr"
        nc_dir = os.path.join(output_dir, rel)
        if not fs.exists(f"{in_root}/{rel}/data.zarr"):
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
