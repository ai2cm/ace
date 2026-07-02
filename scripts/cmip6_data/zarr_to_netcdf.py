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
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import fsspec
import pandas as pd
import xarray as xr

# Include process id in the log format so worker-side and master-side
# lines are distinguishable when 8 workers interleave their output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [pid=%(process)d] %(message)s",
)
logger = logging.getLogger(__name__)


def convert_one(
    zarr_path: str, nc_dir: str, years_per_file: int, force: bool = False
) -> str:
    """Convert a single zarr store to multi-year netCDF files with 1-day chunks.

    Files are grouped in spans of ``years_per_file`` consecutive years,
    named ``data.{first}-{last}.nc`` (e.g. ``data.1940-1949.nc`` for
    ``years_per_file=10``). The intra-file chunking stays at 1 day per
    chunk so per-day random access remains fast — only file granularity
    changes. ``years_per_file=1`` recovers the legacy per-year layout
    with names ``data.{year}-{year}.nc`` (consumer-side glob ``data.*.nc``
    is unaffected).

    Logs progress and timing per-group so partial failures (a worker
    dying mid-dataset, network blip, an HDF5 write error on one decade)
    show up in pod logs with which decade was in-flight rather than
    appearing as one opaque "FAILED: ..." line at the end.
    """
    # Worker subprocess logging is configured from scratch under
    # ``spawn`` — re-initialise to match the master so output isn't
    # silently dropped. ``force=True`` so re-imports don't no-op.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [pid=%(process)d] %(message)s",
        force=True,
    )
    worker_log = logging.getLogger(__name__)
    os.makedirs(nc_dir, exist_ok=True)
    t_dataset = time.monotonic()
    worker_log.info("→ %s", zarr_path)
    # Open lazy — DO NOT call ``ds.load()`` here. A full 86-year ssp
    # zarr is ~17 GB resident; with multiple workers it OOM-kills the
    # process pool. We materialise per output file instead.
    ds = xr.open_zarr(zarr_path)
    worker_log.info(
        "    opened %s: %d vars, time=%d steps",
        os.path.basename(nc_dir.rstrip("/")) or zarr_path,
        len(ds.data_vars),
        int(ds.sizes.get("time", 0)),
    )
    # zlib1 + shuffle: roughly matches zarr's blosc/lz4 default ratio
    # (~0.41 vs ~0.36 of raw on v2 sample) for ~3x the write cost
    # and only ~70% slower per-variable reads (0.19s vs 0.11s on a
    # 2-year subset — overlapped with compute via prefetching at
    # training time, so the wall-clock impact on training step time
    # is negligible). Without these settings the netCDF mirror runs
    # ~2.5x the on-disk size of the zarr source (v2 measured: 4 TB
    # zarr → 11 TB netCDF); with these, ~4.5 TB. Higher complevels
    # (3, 5) gain only 1-2% extra compression for 25-65% more write
    # time. Shuffle is the byte-shuffle filter — significant gains
    # on float climate data because it groups same-exponent bytes
    # together for the zlib coder.
    encoding = {
        name: {
            "chunksizes": (1,) + ds[name].shape[1:],
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
        }
        for name in ds.data_vars
        if "time" in ds[name].dims
    }
    # Group by the floor-divided decade-or-N-year bin. ``groupby`` on an
    # xarray-derived integer label keeps each group's time slice contiguous
    # so the resulting netCDF retains time-monotonicity. Per-group load +
    # write keeps peak RSS at one decade's worth (~2 GB per dataset).
    bin_label = (ds["time.year"] // years_per_file) * years_per_file
    n_written = 0
    n_skipped = 0
    for _, group_ds in ds.groupby(bin_label.rename("year_bin")):
        years = group_ds["time.year"].values
        first_year, last_year = int(years.min()), int(years.max())
        nc_path = os.path.join(nc_dir, f"data.{first_year}-{last_year}.nc")
        if os.path.exists(nc_path):
            if not force:
                n_skipped += 1
                continue
            os.remove(nc_path)
            worker_log.info(
                "    %d-%d: overwriting existing file", first_year, last_year
            )
        t_group = time.monotonic()
        worker_log.info(
            "    %d-%d: loading %d timesteps...",
            first_year,
            last_year,
            int(group_ds.sizes["time"]),
        )
        group_ds.load()
        worker_log.info(
            "    %d-%d: writing %s",
            first_year,
            last_year,
            os.path.basename(nc_path),
        )
        group_ds.to_netcdf(nc_path, encoding=encoding)
        group_ds.close()
        size_mb = os.path.getsize(nc_path) / (1024 * 1024)
        worker_log.info(
            "    %d-%d: %s done in %.1fs (%.0f MiB)",
            first_year,
            last_year,
            os.path.basename(nc_path),
            time.monotonic() - t_group,
            size_mb,
        )
        n_written += 1
    ds.close()
    worker_log.info(
        "← %s in %.1fs (%d written, %d skipped)",
        zarr_path,
        time.monotonic() - t_dataset,
        n_written,
        n_skipped,
    )
    return f"ok: {nc_dir} ({n_written} written, {n_skipped} skipped)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir", help="Root of zarr dataset tree (contains index.csv)"
    )
    parser.add_argument("output_dir", help="Root of netCDF output tree")
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel conversion workers"
    )
    parser.add_argument(
        "--years-per-file",
        type=int,
        default=10,
        help=(
            "Number of consecutive calendar years per output netCDF file "
            "(default 10 = decade files like ``data.1940-1949.nc``). "
            "Intra-file chunking stays at 1 day regardless. Use 1 to "
            "recover the legacy per-year layout, or 20 for half-century "
            "files."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing output netCDFs and top-level aux files "
            "(stats.csv, presence.*, etc.). Default is skip-if-exists."
        ),
    )
    parser.add_argument(
        "--dataset-keys",
        nargs="*",
        default=None,
        help=(
            "Restrict per-dataset conversion to the listed "
            "``source_id/experiment/variant_label`` triples (one per arg). "
            "Top-level aux files are still copied regardless. If omitted, "
            "convert every dataset in the index."
        ),
    )
    args = parser.parse_args()
    if args.years_per_file < 1:
        parser.error("--years-per-file must be >= 1")

    input_dir = args.input_dir.rstrip("/")
    output_dir = args.output_dir.rstrip("/")
    os.makedirs(output_dir, exist_ok=True)

    fs, in_root = fsspec.core.url_to_fs(input_dir)

    index_url = f"{input_dir}/index.csv"
    if not fs.exists(f"{in_root}/index.csv"):
        raise FileNotFoundError(f"No index.csv found at {index_url}")
    idx = pd.read_csv(index_url)

    # Auxiliary top-level files (index.csv, stats.csv, stats.parquet,
    # presence.*, etc.) and aux subdirectories that the training side
    # reads from Weka (``normalization_*`` carries cohort + per-source
    # ``centering.nc`` / ``scaling.nc`` consumed by the model's
    # normalizer). Recursive copies skip the per-dataset directories
    # (source_id/experiment/variant) — those are handled by the
    # converter pass below.
    aux_dirs_to_copy = ("normalization_full", "per_source_normalization_full")
    for entry in fs.ls(in_root, detail=True):
        name = os.path.basename(entry["name"])
        dst = os.path.join(output_dir, name)
        kind = entry.get("type")
        if kind == "file":
            if os.path.exists(dst) and not args.force:
                continue
            logger.info("Copying %s -> %s", name, dst)
            if input_dir.startswith(("gs://", "s3://", "http://", "https://")):
                fs.get(entry["name"], dst)
            else:
                shutil.copy2(entry["name"], dst)
        elif kind == "directory" and name in aux_dirs_to_copy:
            # Trees of centering / scaling / residual netCDFs. Small
            # enough (a few MB up to ~100 MB total) to copy in one
            # shot per pass.
            if os.path.exists(dst) and not args.force:
                logger.info(
                    "Skipping existing aux dir %s (use --force to overwrite)",
                    dst,
                )
                continue
            if os.path.exists(dst):
                shutil.rmtree(dst)
            logger.info("Copying aux dir %s/ -> %s/", name, dst)
            fs.get(entry["name"], dst, recursive=True)

    allow = set(args.dataset_keys) if args.dataset_keys else None
    tasks = []
    for _, row in idx.iterrows():
        rel = os.path.join(row["source_id"], row["experiment"], row["variant_label"])
        if allow is not None and rel not in allow:
            continue
        zarr_path = f"{input_dir}/{rel}/data.zarr"
        nc_dir = os.path.join(output_dir, rel)
        if not fs.exists(f"{in_root}/{rel}/data.zarr"):
            logger.warning("Zarr not found, skipping: %s", zarr_path)
            continue
        tasks.append((zarr_path, nc_dir))
    if allow is not None:
        unmatched = allow - {
            os.path.join(r["source_id"], r["experiment"], r["variant_label"])
            for _, r in idx.iterrows()
        }
        if unmatched:
            raise SystemExit(
                f"--dataset-keys had {len(unmatched)} entries not found in "
                f"index.csv: {sorted(unmatched)}"
            )

    logger.info(
        "Converting %d datasets with %d workers, %d years per file",
        len(tasks),
        args.workers,
        args.years_per_file,
    )
    if not tasks:
        logger.warning("No datasets to convert; exiting.")
        return
    t_master = time.monotonic()
    done = 0
    n_failed = 0
    # ``spawn`` (not the Linux default ``fork``) — once the main
    # process imports fsspec / gcsfs / xarray-with-dask, fork-cloning
    # the gRPC threads and async loops into a child interpreter
    # crashes the child instantly ("A process in the process pool was
    # terminated abruptly"). Same issue compute_stats.py hits; same
    # fix. See compute_stats.py for the gory background.
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(convert_one, zp, nd, args.years_per_file, args.force): zp
            for zp, nd in tasks
        }
        for future in as_completed(futures):
            done += 1
            zarr_path = futures[future]
            try:
                msg = future.result()
                logger.info("[%d/%d] %s", done, len(tasks), msg)
            except Exception as e:
                # Two flavours of failure: (1) the worker raised a
                # plain Python exception — we get a traceback; (2)
                # the worker died via signal (OOM-kill, segfault from
                # forked C libs, etc.) — ProcessPoolExecutor raises
                # BrokenProcessPool with only a generic message. Log
                # both forms so post-mortem in pod logs has enough
                # context.
                n_failed += 1
                logger.error(
                    "[%d/%d] FAILED: %s: %s: %s",
                    done,
                    len(tasks),
                    zarr_path,
                    type(e).__name__,
                    e,
                )
                tb = traceback.format_exc()
                if tb and "BrokenProcessPool" not in tb:
                    for line in tb.rstrip().splitlines():
                        logger.error("    %s", line)
            # Master-side checkpoint: every 10 datasets log a summary
            # so a Beaker tail-the-log session shows the run isn't
            # stalled even on long datasets.
            if done % 10 == 0 or done == len(tasks):
                elapsed = time.monotonic() - t_master
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(tasks) - done) / rate if rate > 0 else float("inf")
                logger.info(
                    "  progress: %d/%d (%d failed) elapsed=%.0fs eta=%.0fs",
                    done,
                    len(tasks),
                    n_failed,
                    elapsed,
                    eta,
                )

    logger.info(
        "Done. Output at %s. %d succeeded, %d failed (%.0fs total).",
        output_dir,
        len(tasks) - n_failed,
        n_failed,
        time.monotonic() - t_master,
    )
    # Exit non-zero so Beaker / CI marks the job failed if any
    # dataset's conversion didn't land. The legacy behaviour returned
    # 0 unconditionally, which silently masked a 243/243-failed run
    # earlier in this session.
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
