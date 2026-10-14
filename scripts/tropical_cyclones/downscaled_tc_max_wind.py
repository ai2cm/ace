"""
Sample per-track maximum wind speed from a downscaled dataset and histogram it.

``detect_tc_tracks.py`` detects TC tracks on a coarse-resolution ACE dataset and
writes ``tracks.csv`` (one row per track point: track_id, time, lon, lat, ...).
This script takes those coarse-grid tracks and follows them into a *downscaled*
dataset -- a higher-resolution field assumed to cover the same storms as the
coarse run the tracks came from -- to measure how strong each storm's winds are
once the finer grid resolves them.

For each track point it samples the downscaled wind within a small box around
the (lat, lon) center rather than a single nearest cell: the track center is
only located to coarse-grid accuracy, so the resolved wind maximum generally
sits somewhere in the storm's inner core, offset from that center. The box max
is taken over every track point, giving one peak wind value per track. Those
per-track maxima are then histogrammed exactly as ``plot_max_wind_histogram.py``
does (which this reuses), and also written to a ``tracks.csv``-compatible CSV so
the downscaled distribution can be overlaid against the coarse one with
``plot_max_wind_histogram.py``.

Usage examples:
    # Sample the downscaled zarr along tracks detected on the coarse run
    python downscaled_tc_max_wind.py downscaled.zarr out/tracks.csv \
        --out-dir downscaled_out/

    # Wind stored as a single speed variable rather than u/v components
    python downscaled_tc_max_wind.py downscaled.zarr out/tracks.csv \
        --out-dir downscaled_out/ --wind-var windspeed_10m

    # Overlay coarse vs. downscaled afterwards
    python plot_max_wind_histogram.py out/tracks.csv \
        downscaled_out/downscaled_tracks.csv --labels coarse downscaled \
        --out compare.png
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import xarray as xr
from detect_tc_tracks import detect_lat_lon_names
from plot_max_wind_histogram import plot_histograms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _match_track_lon(track_lon: float, ds_lon: xr.DataArray) -> float:
    """Put a track longitude into the downscaled dataset's longitude convention.

    TempestExtremes reports longitudes in [0, 360); a downscaled dataset may use
    [-180, 180). Without matching conventions the box selection around a storm
    center silently lands on empty water on the wrong side of the antimeridian.
    """
    lon_min = float(ds_lon.min())
    lon_max = float(ds_lon.max())
    if lon_max <= 180.0 and track_lon > 180.0:
        return track_lon - 360.0
    if lon_min >= 0.0 and track_lon < 0.0:
        return track_lon + 360.0
    return track_lon


def _wind_speed(ds: xr.Dataset, u_var: str, v_var: str, wind_var: str) -> xr.DataArray:
    """Return a wind-speed field, from a single speed var or from u/v components."""
    if wind_var is not None:
        if wind_var not in ds:
            raise KeyError(
                f"{wind_var!r} not found in downscaled dataset; available: "
                f"{list(ds.data_vars)}"
            )
        return ds[wind_var]
    for name in (u_var, v_var):
        if name not in ds:
            raise KeyError(
                f"{name!r} not found in downscaled dataset; available: "
                f"{list(ds.data_vars)}. Pass --wind-var if wind is stored as a "
                "single speed variable."
            )
    return (ds[u_var] ** 2 + ds[v_var] ** 2) ** 0.5


def track_max_wind(
    speed: xr.DataArray,
    track_df: pd.DataFrame,
    lat_name: str,
    lon_name: str,
    radius: float,
) -> float:
    """Max downscaled wind speed within ``radius`` deg of any point on a track.

    For each track point the nearest time slice is taken, then a lat/lon box of
    half-width ``radius`` degrees is masked out (boolean masking rather than
    ``.sel(slice)`` so it is robust to descending or unsorted coordinates), and
    the box maximum is recorded. The track's value is the max over all points.
    """
    lat = speed[lat_name]
    lon = speed[lon_name]
    peak = float("-inf")
    for row in track_df.itertuples(index=False):
        at_time = speed.sel(time=row.time, method="nearest")
        tlon = _match_track_lon(row.lon, lon)
        in_box = (
            (lat >= row.lat - radius)
            & (lat <= row.lat + radius)
            & (lon >= tlon - radius)
            & (lon <= tlon + radius)
        )
        box = at_time.where(in_box, drop=True)
        if box.size == 0:
            continue
        peak = max(peak, float(box.max()))
    return peak


def compute_downscaled_max_wind(
    ds: xr.Dataset,
    tracks: pd.DataFrame,
    lat_name: str,
    lon_name: str,
    u_var: str,
    v_var: str,
    wind_var: str,
    radius: float,
) -> pd.DataFrame:
    """One row per track: track_id and its peak downscaled wind speed.

    The output columns (``track_id``, ``wind``) match ``tracks.csv`` from
    ``detect_tc_tracks.py`` so the result is directly consumable by
    ``plot_max_wind_histogram.py``.
    """
    speed = _wind_speed(ds, u_var, v_var, wind_var)
    records = []
    for track_id, track_df in tracks.groupby("track_id"):
        peak = track_max_wind(speed, track_df, lat_name, lon_name, radius)
        if peak == float("-inf"):
            logger.warning(
                "Track %d had no downscaled grid points within %.1f deg of any "
                "track point; skipping.",
                track_id,
                radius,
            )
            continue
        records.append({"track_id": int(track_id), "wind": peak})
    out = pd.DataFrame(records)
    logger.info(
        "Sampled downscaled peak wind for %d/%d tracks",
        len(out),
        tracks["track_id"].nunique(),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("zarr", help="Path/URI of the downscaled dataset (zarr).")
    parser.add_argument(
        "tracks_csv",
        help="tracks.csv written by detect_tc_tracks.py on the coarse dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default="downscaled_out",
        help="Directory for the per-track CSV and histogram.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Half-width (deg) of the lat/lon box sampled around each track "
        "point. Covers the storm inner core offset from the coarse-grid center.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Ensemble member to select if a 'sample' dim is present.",
    )
    parser.add_argument("--time-start", default=None, help="Time subset start.")
    parser.add_argument("--time-end", default=None, help="Time subset end.")
    parser.add_argument("--u-var", default="UGRD10m", help="10m eastward wind var.")
    parser.add_argument("--v-var", default="VGRD10m", help="10m northward wind var.")
    parser.add_argument(
        "--wind-var",
        default=None,
        help="Single wind-speed var to use instead of computing |(u, v)|.",
    )
    parser.add_argument(
        "--lat-name",
        default=None,
        help="Latitude coord name (auto-detected if not set).",
    )
    parser.add_argument(
        "--lon-name",
        default=None,
        help="Longitude coord name (auto-detected if not set).",
    )
    parser.add_argument(
        "--bins", type=int, default=20, help="Number of histogram bins."
    )
    parser.add_argument(
        "--label",
        default="downscaled",
        help="Legend label for the histogram.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(args.tracks_csv, parse_dates=["time"])
    for col in ("track_id", "time", "lat", "lon"):
        if col not in tracks.columns:
            raise KeyError(
                f"{col!r} column not found in {args.tracks_csv}; available: "
                f"{list(tracks.columns)}"
            )

    logger.info("Opening downscaled dataset %s", args.zarr)
    ds = xr.open_zarr(args.zarr)
    if "sample" in ds.dims:
        logger.info("Selecting sample=%d of %d", args.sample, ds.sizes["sample"])
        ds = ds.isel(sample=args.sample)
    if args.time_start is not None or args.time_end is not None:
        ds = ds.sel(time=slice(args.time_start, args.time_end))

    lat_name, lon_name = args.lat_name, args.lon_name
    if lat_name is None or lon_name is None:
        lat_name, lon_name = detect_lat_lon_names(ds)
        logger.info("Auto-detected lat/lon coord names: %s, %s", lat_name, lon_name)

    per_track = compute_downscaled_max_wind(
        ds,
        tracks,
        lat_name=lat_name,
        lon_name=lon_name,
        u_var=args.u_var,
        v_var=args.v_var,
        wind_var=args.wind_var,
        radius=args.radius,
    )

    csv_path = out_dir / "downscaled_tracks.csv"
    per_track.to_csv(csv_path, index=False)
    logger.info("Wrote per-track downscaled max wind to %s", csv_path)

    hist_path = out_dir / "downscaled_max_wind_histogram.png"
    plot_histograms(
        [per_track.set_index("track_id")["wind"]],
        [args.label],
        bins=args.bins,
        out_path=hist_path,
    )


if __name__ == "__main__":
    main()
