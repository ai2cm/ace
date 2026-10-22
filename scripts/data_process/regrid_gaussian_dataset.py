"""Conservatively regrid a zarr dataset between Gaussian grids.

This is meant for making a coarser version of an existing dataset when the raw
model output is not available, e.g. regridding a 1-degree (F90) E3SM dataset to
the 4-degree (F22.5) grid used for ERA5. The Gaussian grid definitions match
those in scripts/era5/pipeline/xr-beam-pipeline.py.

Since both grids are regular in latitude and longitude, first-order
conservative regridding is separable: the weights are the fractional overlaps
of source and target cells in sin(latitude) and in longitude, with cell
bounds at the midpoints between cell centers. Unlike xESMF, which treats cell
edges as great circles, this exactly conserves area-weighted means computed
with these latitude-longitude cell areas.

Variables with dims (time, lat, lon) are regridded in blocks of time, in
parallel, and written directly into a pre-initialized zarr store. Variables
with dims (lat, lon) are regridded once, and scalar variables are copied. The
regridding can be resumed if interrupted: completed time blocks are recorded in
a "<output>.progress" directory, which is removed once all blocks are done.

The dependencies of this script are xarray, zarr>=3 and dask, plus the
dependencies of get_stats.py if --stats-output-directory is given.

Example:
    python regrid_gaussian_dataset.py \
        /path/to/1deg-dataset.zarr /path/to/4deg-dataset.zarr \
        --input-grid F90 --output-grid F22.5 \
        --fraction-variables LANDFRAC,OCNFRAC,ICEFRAC \
        --stats-output-directory /path/to/4deg-dataset-stats
"""

import logging
import multiprocessing
import os
import shutil
import time

import click
import dask.array as da
import numpy as np
import xarray as xr
import zarr

# Gaussian grid specs: name -> N (grid number; nlat=2N, nlon=4N)
GAUSSIAN_GRID_N = {
    "F22.5": 22.5,
    "F45": 45,
    "F90": 90,
    "F360": 360,
}

# (latitude weights, longitude weights), set in the parent process before
# forking workers.
_WEIGHTS: tuple[np.ndarray, np.ndarray] | None = None


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Compute cell boundaries from centers, clipping to [lo, hi]."""
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    bounds = np.concatenate([[lo], midpoints, [hi]])
    return bounds


def _gaussian_latitudes(n: int | float) -> np.ndarray:
    """Compute Gaussian grid latitudes for grid number N (2N latitudes).

    Returns latitudes in degrees, sorted south-to-north.
    These are the roots of the Legendre polynomial P_{2N}(sin(lat)).
    """
    from numpy.polynomial.legendre import leggauss

    nlat = round(2 * n)
    x, _ = leggauss(nlat)
    lat = np.degrees(np.arcsin(x))
    return np.sort(lat)


def _make_gaussian_grid(grid: str) -> xr.Dataset:
    """Create Gaussian grid dataset with cell centers and bounds."""
    n = GAUSSIAN_GRID_N[grid]
    lat = _gaussian_latitudes(n)
    nlon = round(4 * n)
    dlon = 360.0 / nlon
    # Longitude centers offset by half a grid spacing
    lon = np.linspace(dlon / 2, 360 - dlon / 2, nlon)
    lat_b = _cell_bounds(lat, -90, 90)
    lon_b = _cell_bounds(lon, 0, 360)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], lat_b),
            "lon_b": (["lon_b"], lon_b),
        }
    )


def _overlap_weights(source_bounds: np.ndarray, target_bounds: np.ndarray):
    """Fraction of each target cell covered by each source cell, for 1D cells
    with monotonically increasing bounds."""
    overlap = np.maximum(
        0.0,
        np.minimum(target_bounds[1:, None], source_bounds[None, 1:])
        - np.maximum(target_bounds[:-1, None], source_bounds[None, :-1]),
    )
    return overlap / overlap.sum(axis=1, keepdims=True)


def _conservative_weights(
    source_grid: xr.Dataset, target_grid: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    sin_lat_b = [
        np.sin(np.radians(g["lat_b"].values)) for g in (source_grid, target_grid)
    ]
    lat_weights = _overlap_weights(*sin_lat_b)
    lon_weights = _overlap_weights(
        source_grid["lon_b"].values, target_grid["lon_b"].values
    )
    return lat_weights, lon_weights


def _regrid_array(array: np.ndarray, clip: bool) -> np.ndarray:
    """Regrid an array whose last two dimensions are (lat, lon)."""
    assert _WEIGHTS is not None
    lat_weights, lon_weights = _WEIGHTS
    regridded = lat_weights @ array.astype(np.float64) @ lon_weights.T
    if clip:
        regridded = np.clip(regridded, 0.0, 1.0)
    return regridded.astype(array.dtype)


def _progress_marker(progress_dir: str, start: int) -> str:
    return os.path.join(progress_dir, f"{start:09d}.done")


def _regrid_block(args: tuple) -> int:
    input_zarr, output_zarr, progress_dir, names, fraction_names, start, stop = args
    source = zarr.open_group(input_zarr, mode="r")
    target = zarr.open_group(output_zarr, mode="r+")
    for name in names:
        target[name][start:stop] = _regrid_array(
            source[name][start:stop], clip=name in fraction_names
        )
    open(_progress_marker(progress_dir, start), "w").close()
    return start


def _initialize_output(
    ds: xr.Dataset,
    output_zarr: str,
    output_grid: xr.Dataset,
    time_varying: list[str],
    fraction_names: set[str],
    shard_size: int,
    history: str,
):
    """Write coordinates, static and scalar variables, and metadata for the
    time-varying variables, whose data are filled in later by time block."""
    n_time = ds.sizes["time"]
    n_lat, n_lon = output_grid.sizes["lat"], output_grid.sizes["lon"]
    template = xr.Dataset(
        coords={
            "time": ds["time"],
            "lat": ("lat", output_grid["lat"].values, ds["lat"].attrs),
            "lon": ("lon", output_grid["lon"].values, ds["lon"].attrs),
        },
        attrs={**ds.attrs, "regrid_history": history},
    )
    encoding = {
        "time": {
            k: v
            for k, v in ds["time"].encoding.items()
            if k in ("units", "calendar", "dtype")
        }
    }
    for name, variable in ds.data_vars.items():
        if name in time_varying:
            template[name] = (
                variable.dims,
                da.zeros(
                    (n_time, n_lat, n_lon),
                    dtype=variable.dtype,
                    chunks=(shard_size, n_lat, n_lon),
                ),
                variable.attrs,
            )
            encoding[name] = {
                "chunks": (1, n_lat, n_lon),
                "shards": (shard_size, n_lat, n_lon),
            }
        elif variable.dims == ("lat", "lon"):
            template[name] = (
                variable.dims,
                _regrid_array(variable.values, clip=name in fraction_names),
                variable.attrs,
            )
        elif variable.dims == ():
            template[name] = variable.load()
        else:
            raise ValueError(f"Unsupported dims {variable.dims} for {name}")
    template.to_zarr(
        output_zarr, mode="w-", compute=False, encoding=encoding, zarr_format=3
    )


@click.command()
@click.argument("input_zarr", type=str)
@click.argument("output_zarr", type=str)
@click.option("--input-grid", default="F90", type=click.Choice(list(GAUSSIAN_GRID_N)))
@click.option(
    "--output-grid", default="F22.5", type=click.Choice(list(GAUSSIAN_GRID_N))
)
@click.option(
    "--fraction-variables",
    default="",
    help="Comma-separated variables to clip to [0, 1] after regridding.",
)
@click.option(
    "--time-block-size",
    default=360,
    help="Number of times regridded per task; also the output shard size.",
)
@click.option("--n-workers", default=32, help="Number of worker processes.")
@click.option(
    "--max-blocks",
    default=None,
    type=int,
    help="Only regrid this many time blocks (for testing).",
)
@click.option(
    "--stats-output-directory",
    default=None,
    help="If given, compute normalization stats with get_stats.py once done.",
)
def main(
    input_zarr: str,
    output_zarr: str,
    input_grid: str,
    output_grid: str,
    fraction_variables: str,
    time_block_size: int,
    n_workers: int,
    max_blocks: int | None,
    stats_output_directory: str | None,
):
    global _WEIGHTS
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    input_zarr = input_zarr.rstrip("/")
    output_zarr = output_zarr.rstrip("/")
    progress_dir = output_zarr + ".progress"
    fraction_names = {name for name in fraction_variables.split(",") if name}

    ds = xr.open_zarr(input_zarr, decode_timedelta=False)
    source_grid = _make_gaussian_grid(input_grid)
    target_grid = _make_gaussian_grid(output_grid)
    np.testing.assert_allclose(ds["lat"].values, source_grid["lat"].values, atol=1e-6)
    np.testing.assert_allclose(ds["lon"].values, source_grid["lon"].values, atol=1e-6)
    missing = fraction_names - set(ds.data_vars)
    if missing:
        raise ValueError(f"Fraction variables not in dataset: {sorted(missing)}")
    time_varying = [
        name for name, v in ds.data_vars.items() if v.dims == ("time", "lat", "lon")
    ]

    _WEIGHTS = _conservative_weights(source_grid, target_grid)

    if not os.path.exists(output_zarr):
        history = (
            f"Conservatively regridded from {input_grid} to {output_grid} Gaussian "
            f"grid by scripts/data_process/regrid_gaussian_dataset.py "
            f"from {input_zarr}. Clipped to [0, 1]: {sorted(fraction_names)}."
        )
        logging.info(f"Initializing {output_zarr}")
        _initialize_output(
            ds,
            output_zarr,
            target_grid,
            time_varying,
            fraction_names,
            time_block_size,
            history,
        )
        os.makedirs(progress_dir)
    elif not os.path.isdir(progress_dir):
        logging.info(f"{output_zarr} exists and is complete. Skipping regridding.")
        time_varying = []

    n_time = ds.sizes["time"]
    starts = range(0, n_time, time_block_size)
    if max_blocks is not None:
        starts = starts[:max_blocks]
    tasks = [
        (
            input_zarr,
            output_zarr,
            progress_dir,
            time_varying,
            fraction_names,
            start,
            min(start + time_block_size, n_time),
        )
        for start in starts
        if time_varying and not os.path.exists(_progress_marker(progress_dir, start))
    ]
    logging.info(
        f"Regridding {len(tasks)} of {len(starts)} time blocks of "
        f"{len(time_varying)} variables with {n_workers} workers"
    )
    initial_time = time.time()
    with multiprocessing.get_context("fork").Pool(n_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_regrid_block, tasks), start=1):
            if i % 10 == 0 or i == len(tasks):
                elapsed = time.time() - initial_time
                logging.info(
                    f"{i}/{len(tasks)} blocks done, "
                    f"{elapsed / 60:.1f} min elapsed, "
                    f"{elapsed / i * (len(tasks) - i) / 60:.1f} min remaining"
                )

    if os.path.isdir(progress_dir):
        if len(os.listdir(progress_dir)) < len(range(0, n_time, time_block_size)):
            logging.info("Not all time blocks are done; stopping before stats.")
            return
        shutil.rmtree(progress_dir)
    zarr.consolidate_metadata(output_zarr)

    if stats_output_directory is not None:
        from get_stats import StatsConfig, get_stats

        get_stats(
            config=StatsConfig(
                output_directory=stats_output_directory, data_type="E3SMV2"
            ),
            input_zarr=output_zarr,
            out_dir=stats_output_directory,
            debug=False,
        )


if __name__ == "__main__":
    main()
