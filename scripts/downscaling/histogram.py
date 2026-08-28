import argparse
import os
import re
from datetime import datetime

import cftime
import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

WIND_SPEED = "wind_speed"
# 10m wind component name pairs, tried in order against the dataset
WIND_COMPONENT_NAMES = (
    ("UGRD10m", "VGRD10m"),
    ("eastward_wind_at_ten_meters", "northward_wind_at_ten_meters"),
)

# latitude dimension names, tried in order against the dataset
LAT_DIM_NAMES = ("lat", "latitude", "grid_yt")

# ProgressBar redraws via a bare "\r" with no newline, intended for a tty to
# overwrite in place. Log streaming (e.g. beaker/gantry) is line-buffered on
# "\n", so those redraws never surface at all until the one real newline
# ProgressBar writes at completion. LineProgressBar below writes a real "\n"
# each update instead, so periodic progress actually appears in the log.
PROGRESS_BAR_UPDATE_INTERVAL_SECONDS = 3
# gcsfs runs GCS calls through its own asyncio event loop in a background
# thread; dask's default threaded scheduler hitting that loop concurrently
# from many worker threads is a known deadlock trigger (most easily hit when
# a mid-computation credential refresh is involved). Processes each get
# their own event loop, avoiding the shared-loop deadlock.
DASK_SCHEDULER = "processes"


class LineProgressBar(ProgressBar):
    """``ProgressBar`` variant that writes a real newline per update.

    ``ProgressBar`` writes "\\r[bar] | X% Completed | elapsed" with no
    newline between updates, relying on a tty to overwrite the line in
    place. Piped through line-buffered log streaming, none of that surfaces
    until the single "\\n" ``ProgressBar`` writes at completion, so nothing
    appears in the log until the whole computation is done.
    """

    def _draw_bar(self, frac, elapsed):
        from dask.utils import format_time

        percent = int(100 * frac)
        msg = f"{percent}% Completed | {format_time(elapsed)}\n"
        if self._file is not None:
            self._file.write(msg)
            self._file.flush()


TIME_FORMAT = "%Y%m%d:%H%M"
# strptime accepts fewer digits than the format implies ("2013011:1200" parses
# as 2013-01-01 12:00, "20130101:12" as 00:12), which would silently select the
# wrong window, so the field widths are checked before parsing.
TIME_PATTERN = re.compile(r"^\d{8}:\d{4}$")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute histograms over dataset")
    parser.add_argument(
        "path",
        help="zarr dataset path",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=[
            "PRATEsfc",
            "eastward_wind_at_ten_meters",
            "northward_wind_at_ten_meters",
            "PRMSL",
            "wind_speed",
        ],
        help="Variables to compute histograms for",
    )
    parser.add_argument(
        "--output-dir",
        default="./histograms",
        help="Output directory for histogram data",
    )
    parser.add_argument(
        "--start-time",
        default=None,
    )
    parser.add_argument(
        "--stop-time",
        default=None,
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=PROGRESS_BAR_UPDATE_INTERVAL_SECONDS,
        help="Seconds between progress bar updates",
    )
    parser.add_argument(
        "--lat-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Limit the dataset to latitudes in [MIN, MAX]",
    )
    args = parser.parse_args()
    return args


def find_wind_components(ds: xr.Dataset) -> tuple[str, str]:
    """Names of the 10m eastward/northward wind components in ``ds``."""
    for eastward, northward in WIND_COMPONENT_NAMES:
        if eastward in ds and northward in ds:
            return eastward, northward
    raise ValueError(
        f"cannot derive {WIND_SPEED!r}: no 10m wind component pair in dataset, "
        f"looked for {' and '.join(map(str, WIND_COMPONENT_NAMES))}"
    )


def find_lat_dim(ds: xr.Dataset) -> str:
    """Name of the latitude dimension in ``ds``."""
    for name in LAT_DIM_NAMES:
        if name in ds.dims:
            return name
    raise ValueError(f"no latitude dimension in dataset, looked for {LAT_DIM_NAMES}")


def select_lat_range(ds: xr.Dataset, lat_min: float, lat_max: float) -> xr.Dataset:
    dim = find_lat_dim(ds)
    return ds.sel({dim: slice(lat_min, lat_max)})


def add_wind_speed(ds: xr.Dataset) -> tuple[xr.Dataset, tuple[str, str]]:
    """``ds`` with a lazy ``wind_speed`` variable, plus its component names."""
    eastward, northward = find_wind_components(ds)
    wind_speed = np.sqrt(ds[eastward] ** 2 + ds[northward] ** 2)
    wind_speed.attrs = {
        "units": ds[eastward].attrs.get("units", "m/s"),
        "long_name": "wind speed at ten meters",
    }
    return ds.assign({WIND_SPEED: wind_speed}), (eastward, northward)


def compute_histograms(
    ds: xr.Dataset,
    variables: list[str],
    bins: dict[str, np.ndarray],
    progress_interval: float = PROGRESS_BAR_UPDATE_INTERVAL_SECONDS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Histograms for ``variables`` in a single pass over the data.

    The histograms are submitted to dask as one graph, so inputs shared between
    them (e.g. the wind components read by both ``wind_speed`` and its own
    component histograms) are read once rather than once per variable.
    """
    counts = {var: da.histogram(ds[var].data, bins=bins[var])[0] for var in variables}
    with LineProgressBar(dt=progress_interval):
        (counts,) = dask.compute(counts, scheduler=DASK_SCHEDULER)
    # bins are explicit edges, so they are the edges da.histogram would return
    return {var: (counts[var], bins[var]) for var in variables}


def trim_empty_bins(
    counts: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``counts`` and ``edges`` without their all-zero leading and trailing bins.

    Only the contiguous runs of empty bins at the two ends are dropped; empty
    bins that have populated bins on both sides are kept, so gaps within the
    distribution stay visible. Returned unchanged if every bin is empty.
    """
    (populated,) = np.nonzero(counts)
    if len(populated) == 0:
        return counts, edges
    first, last = populated[0], populated[-1]
    # bin i spans edges[i] to edges[i + 1], so keeping bins first..last needs
    # one more edge than bins
    return counts[first : last + 1], edges[first : last + 2]


def save_histogram(
    counts: np.ndarray, edges: np.ndarray, variable: str, output_dir: str, units: str
) -> None:
    nc = xr.Dataset(
        data_vars={
            "counts": (("bin",), counts),
            "edges": (("bin_edge",), edges),
        },
        coords={
            "bin": np.arange(len(counts)),
            "bin_edge": np.arange(len(edges)),
        },
    )
    nc.to_netcdf(f"{output_dir}/{variable}_histogram.nc")
    # plot histogram as step plot and save to png, over the populated range only
    # (the netCDF above keeps the full bin range)
    plot_counts, plot_edges = trim_empty_bins(counts, edges)
    frequency = plot_counts / plot_counts.sum()
    plt.step(plot_edges[:-1], frequency, where="post")
    plt.yscale("log")
    plt.xlabel(f"{units}")
    plt.ylabel("Frequency")
    plt.title(f"{variable}")
    plt.savefig(f"{output_dir}/{variable}_histogram.png")
    plt.close()


def estimate_bins(
    data: xr.DataArray, nbins: int = 500, stretch_factor: float = 6
) -> np.ndarray:
    data_0 = data.isel(time=0)
    data_min, data_max = dask.compute(
        data_0.min(), data_0.max(), scheduler=DASK_SCHEDULER
    )
    center = (float(data_min) + float(data_max)) / 2
    half_width = (float(data_max) - float(data_min)) / 2 * stretch_factor
    return np.linspace(center - half_width, center + half_width, nbins + 1)


def str_to_datetime(time_str: str) -> datetime:
    """Parse a "YYYYMMDD:HHMM" timestamp, e.g. "20130101:1200" -> 2013-01-01 12:00."""
    if not TIME_PATTERN.match(time_str):
        raise ValueError(
            f"{time_str!r} is not a YYYYMMDD:HHMM timestamp, e.g. 20130101:1200"
        )
    return datetime.strptime(time_str, TIME_FORMAT)


def on_dataset_calendar(time: datetime, ds: xr.Dataset) -> datetime | cftime.datetime:
    """``time`` rebuilt on ``ds``'s own calendar, so it can index ``ds.time``."""
    index = ds.indexes["time"]
    if not isinstance(index, xr.CFTimeIndex):
        return time
    return cftime.datetime(
        time.year,
        time.month,
        time.day,
        time.hour,
        time.minute,
        calendar=index.calendar,
    )


def main():
    args = parse_args()
    ds = xr.open_zarr(args.path)

    if args.lat_range is not None:
        lat_min, lat_max = args.lat_range
        ds = select_lat_range(ds, lat_min, lat_max)

    t_min = (
        args.start_time
        if args.start_time is None
        else on_dataset_calendar(str_to_datetime(args.start_time), ds)
    )
    t_max = (
        args.stop_time
        if args.stop_time is None
        else on_dataset_calendar(str_to_datetime(args.stop_time), ds)
    )
    time_index = ds.indexes["time"]
    ds = ds.sel(time=slice(t_min, t_max))
    if ds.sizes["time"] == 0:
        # an empty selection only surfaces as an IndexError once bins are
        # estimated, so it is reported here where the cause is still visible
        raise ValueError(
            f"no timesteps in [{t_min}, {t_max}]; {args.path} spans "
            f"[{time_index[0]}, {time_index[-1]}] ({len(time_index)} steps)"
            if len(time_index) > 0
            else f"{args.path} has no timesteps"
        )

    if len(args.variables) == 0:
        raise ValueError("No variables provided to compute histograms for.")

    os.makedirs(args.output_dir, exist_ok=True)

    variables = list(args.variables)
    # wind speed is derived from the components, so any component histograms
    # that were also requested go in the same group and share the reads
    wind_group: list[str] = []
    if WIND_SPEED in variables and WIND_SPEED not in ds:
        ds, components = add_wind_speed(ds)
        wind_group = [WIND_SPEED] + [c for c in components if c in variables]
    groups = [g for g in [wind_group] if g] + [
        [var] for var in variables if var not in wind_group
    ]

    for group in groups:
        print(f"computing bins for {', '.join(group)}...")
        bins = {var: estimate_bins(ds[var]) for var in group}
        print(f"computing histograms for {', '.join(group)}...")
        histograms = compute_histograms(
            ds, group, bins, progress_interval=args.progress_interval
        )

        for var, (var_counts, var_edges) in histograms.items():
            save_histogram(
                var_counts,
                var_edges,
                var,
                args.output_dir,
                units=ds[var].attrs.get("units", ""),
            )

    print("done")


if __name__ == "__main__":
    main()
