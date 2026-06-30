#!/usr/bin/env python
"""
Locate tropical cyclones (TCs) in an ACE global output zarr and emit a
``fme.downscaling.predict`` config that downscales a moving 4x4 degree box
around each cyclone.

Detection is a simple filter: a grid point is a TC candidate if its sea-level
pressure is below ``--prmsl-threshold`` AND its 10m wind speed is above
``--wind-threshold``, within a tropical latitude band. Connected candidate
regions are reduced to a pressure-weighted centroid per snapshot, and centroids
are linked across time into tracks by nearest-neighbour association.

Each track becomes a named event (``<prefix><id>``). Each snapshot of a track
becomes a separate ``EventConfig`` named ``<event_name>_<snapshot_date>`` whose
lat/lon extent is a box centred on that snapshot's centroid. With
``save_generated_samples: true`` set on every event, running predict.py writes
one ``<event_name>_<snapshot_date>.nc`` per snapshot, which
``stitch_event_movie.py`` then animates.

Usage:
    python downscale_events.py <ace_output.zarr> --output-config events.yaml \\
        [--base-config template.yaml] [--experiment-dir DIR] \\
        [--prmsl-threshold 1005] [--wind-threshold 17] [--box-size 4.0] ...

The coarse data path in the generated config is set to ``<ace_output.zarr>``.
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from scipy import ndimage

# Default variable names follow the X-SHiELD / ACE2S naming convention.
DEFAULT_PRMSL_NAME = "PRMSL"
DEFAULT_U10_NAME = "eastward_wind_at_ten_meters"
DEFAULT_V10_NAME = "northward_wind_at_ten_meters"

# EventConfig.date format expected by fme.downscaling.predict.
EVENT_DATE_FORMAT = "%Y-%m-%dT%H:%M"
# Compact date stamp embedded in the event name / output filename. Must contain
# a YYYYMMDD substring so downstream tooling (utils.find_event_files,
# plot_events.filename_to_datetime) can parse it.
EVENT_NAME_DATE_FORMAT = "%Y%m%dT%H"


@dataclasses.dataclass
class Detection:
    """A single per-snapshot cyclone centre."""

    time_index: int
    timestamp: np.datetime64
    lat: float
    lon: float
    min_prmsl: float
    max_wind: float


@dataclasses.dataclass
class DetectionConfig:
    prmsl_threshold: float
    wind_threshold: float
    min_abs_lat: float
    max_abs_lat: float
    min_region_size: int


def _circular_mean_lon(lons: np.ndarray, weights: np.ndarray, period: float) -> float:
    """Weighted circular mean of longitudes, robust to the wrap boundary.

    Returns a value in the same range/convention as the input longitudes.
    """
    offset = lons.min()
    angles = (lons - offset) * (2.0 * np.pi / period)
    sin = np.average(np.sin(angles), weights=weights)
    cos = np.average(np.cos(angles), weights=weights)
    mean_angle = np.arctan2(sin, cos) % (2.0 * np.pi)
    return float(offset + mean_angle * (period / (2.0 * np.pi)))


def detect_centers(
    prmsl: np.ndarray,
    wind_speed: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    config: DetectionConfig,
) -> list[tuple[float, float, float, float]]:
    """Find cyclone centres in a single snapshot.

    Returns a list of ``(lat, lon, min_prmsl, max_wind)`` tuples, one per
    connected candidate region that exceeds the minimum size.
    """
    abs_lat = np.abs(lat)[:, None]
    lat_mask = (abs_lat >= config.min_abs_lat) & (abs_lat <= config.max_abs_lat)
    candidate = (
        (prmsl < config.prmsl_threshold)
        & (wind_speed > config.wind_threshold)
        & lat_mask
        & np.isfinite(prmsl)
        & np.isfinite(wind_speed)
    )

    labeled, n_regions = ndimage.label(candidate)
    lon_period = 360.0
    latgrid = np.broadcast_to(lat[:, None], prmsl.shape)
    longrid = np.broadcast_to(lon[None, :], prmsl.shape)

    centers: list[tuple[float, float, float, float]] = []
    for region_id in range(1, n_regions + 1):
        region = labeled == region_id
        if int(region.sum()) < config.min_region_size:
            continue
        # Weight the centroid by pressure depth below the threshold so the
        # centre sits at the strongest part of the low.
        depth = np.clip(config.prmsl_threshold - prmsl[region], 0.0, None)
        if depth.sum() <= 0:
            continue
        lat_c = float(np.average(latgrid[region], weights=depth))
        lon_c = _circular_mean_lon(longrid[region], depth, lon_period)
        centers.append(
            (lat_c, lon_c, float(prmsl[region].min()), float(wind_speed[region].max()))
        )
    return centers


def _lon_distance(lon_a: float, lon_b: float, period: float = 360.0) -> float:
    diff = abs(lon_a - lon_b) % period
    return min(diff, period - diff)


def link_tracks(
    detections_per_time: list[list[Detection]],
    max_distance_deg: float,
    max_gap: int,
) -> list[list[Detection]]:
    """Greedily link per-snapshot detections into tracks by nearest neighbour.

    A detection joins the closest active track whose last centre is within
    ``max_distance_deg`` (great-circle-free, in lat/lon degrees) and whose last
    update was no more than ``max_gap`` snapshots ago; otherwise it seeds a new
    track.
    """
    tracks: list[list[Detection]] = []

    for t, detections in enumerate(detections_per_time):
        active = [track for track in tracks if t - track[-1].time_index <= max_gap]
        claimed: set[int] = set()
        for det in detections:
            best_track = None
            best_dist = max_distance_deg
            for track in active:
                if id(track) in claimed:
                    continue
                last = track[-1]
                dlat = det.lat - last.lat
                dlon = _lon_distance(det.lon, last.lon)
                dist = float(np.hypot(dlat, dlon))
                if dist <= best_dist:
                    best_dist = dist
                    best_track = track
            if best_track is None:
                tracks.append([det])
            else:
                best_track.append(det)
                claimed.add(id(best_track))
    return tracks


def find_cyclone_tracks(
    ds: xr.Dataset,
    prmsl_name: str,
    u10_name: str,
    v10_name: str,
    config: DetectionConfig,
    max_distance_deg: float,
    max_gap: int,
    min_track_length: int,
) -> list[list[Detection]]:
    lat = ds["lat"].values
    lon = ds["lon"].values
    times = ds["time"].values

    detections_per_time: list[list[Detection]] = []
    for t in range(times.shape[0]):
        snap = ds.isel(time=t)
        prmsl = np.asarray(snap[prmsl_name].values, dtype=float)
        u10 = np.asarray(snap[u10_name].values, dtype=float)
        v10 = np.asarray(snap[v10_name].values, dtype=float)
        wind_speed = np.hypot(u10, v10)
        centers = detect_centers(prmsl, wind_speed, lat, lon, config)
        detections_per_time.append(
            [
                Detection(t, times[t], lat_c, lon_c, min_p, max_w)
                for (lat_c, lon_c, min_p, max_w) in centers
            ]
        )

    tracks = link_tracks(detections_per_time, max_distance_deg, max_gap)
    return [track for track in tracks if len(track) >= min_track_length]


def _box_extent(
    center: float, size: float, lo: float, hi: float
) -> tuple[float, float]:
    """Centred interval of width ``size`` clamped to ``[lo, hi]``."""
    half = size / 2.0
    start = center - half
    stop = center + half
    if start < lo:
        start, stop = lo, lo + size
    if stop > hi:
        start, stop = hi - size, hi
    return float(start), float(stop)


def build_events(
    tracks: list[list[Detection]],
    box_size: float,
    n_samples: int,
    name_prefix: str,
    lat_limit: float,
) -> list[dict]:
    events: list[dict] = []
    for track_idx, track in enumerate(tracks, start=1):
        event_name = f"{name_prefix}{track_idx:03d}"
        for det in track:
            timestamp = np.datetime64(det.timestamp, "s").item()
            lat_start, lat_stop = _box_extent(det.lat, box_size, -lat_limit, lat_limit)
            lon_start = det.lon - box_size / 2.0
            lon_stop = det.lon + box_size / 2.0
            name_stamp = timestamp.strftime(EVENT_NAME_DATE_FORMAT)
            events.append(
                {
                    "name": f"{event_name}_{name_stamp}",
                    "n_samples": n_samples,
                    "date": timestamp.strftime(EVENT_DATE_FORMAT),
                    "save_generated_samples": True,
                    "lat_extent": {"start": lat_start, "stop": lat_stop},
                    "lon_extent": {
                        "start": float(lon_start),
                        "stop": float(lon_stop),
                    },
                }
            )
    return events


def _default_base_config() -> dict:
    """Scaffold config with placeholders the user must fill in."""
    return {
        "experiment_dir": "FILL_IN/output",
        "model": {"checkpoint": "FILL_IN/checkpoint.tar"},
        "data": {
            "coarse": [{"data_path": "FILL_IN"}],
            "batch_size": 1,
            "num_data_workers": 0,
            "strict_ensemble": False,
        },
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": True,
            "log_to_file": True,
            "project": "ace-downscaling",
            "entity": "ai2cm",
        },
    }


def build_config(
    base_config: dict,
    coarse_data_path: str,
    events: list[dict],
    experiment_dir: str | None,
) -> dict:
    config = dict(base_config)
    config["events"] = events
    data = dict(config.get("data", {}))
    data["coarse"] = [{"data_path": coarse_data_path}]
    config["data"] = data
    if experiment_dir is not None:
        config["experiment_dir"] = experiment_dir
    return config


def _detect_pressure_threshold(ds: xr.Dataset, prmsl_name: str, hpa: float) -> float:
    """Return the threshold in the data's native pressure units.

    ACE outputs PRMSL in Pa; the CLI takes the threshold in hPa for
    convenience. If the data already looks like hPa, leave it unscaled.
    """
    sample_max = float(np.nanmax(ds[prmsl_name].isel(time=0).values))
    return hpa * 100.0 if sample_max > 2000.0 else hpa


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate tropical cyclones in an ACE global output zarr and "
        "generate a downscaling predict config."
    )
    parser.add_argument("zarr_path", help="Path to the ACE global output zarr.")
    parser.add_argument(
        "--output-config",
        default="downscale_events_config.yaml",
        help="Path to write the generated predict config "
        "(default: downscale_events_config.yaml).",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional template config to merge events into. Its model, "
        "logging, and data settings are preserved. If omitted, a scaffold with "
        "placeholders is written.",
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Override experiment_dir in the generated config.",
    )
    parser.add_argument(
        "--prmsl-name", default=DEFAULT_PRMSL_NAME, help="Sea-level pressure variable."
    )
    parser.add_argument(
        "--u10-name", default=DEFAULT_U10_NAME, help="Eastward 10m wind variable."
    )
    parser.add_argument(
        "--v10-name", default=DEFAULT_V10_NAME, help="Northward 10m wind variable."
    )
    parser.add_argument(
        "--prmsl-threshold",
        type=float,
        default=1005.0,
        help="Max sea-level pressure for a TC candidate, in hPa (default: 1005).",
    )
    parser.add_argument(
        "--wind-threshold",
        type=float,
        default=17.0,
        help="Min 10m wind speed for a TC candidate, in m/s (default: 17).",
    )
    parser.add_argument(
        "--lat-band",
        type=float,
        nargs=2,
        default=(5.0, 45.0),
        metavar=("MIN_ABS_LAT", "MAX_ABS_LAT"),
        help="Absolute latitude band to search (default: 5 45).",
    )
    parser.add_argument(
        "--min-region-size",
        type=int,
        default=4,
        help="Min connected candidate grid cells to count as a centre (default: 4).",
    )
    parser.add_argument(
        "--max-track-distance",
        type=float,
        default=8.0,
        help="Max centroid movement in degrees between snapshots to link a "
        "track (default: 8).",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=1,
        help="Max number of missed snapshots before a track is considered "
        "ended (default: 1).",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=4,
        help="Drop tracks with fewer than this many snapshots (default: 4).",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=4.0,
        help="Side length of the lat/lon bounding box in degrees (default: 4).",
    )
    parser.add_argument(
        "--lat-limit",
        type=float,
        default=88.0,
        help="Hard latitude clamp for boxes (default: 88, the downscaling limit).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Ensemble samples to generate per event (default: 8).",
    )
    parser.add_argument(
        "--name-prefix",
        default="tc_",
        help="Prefix for generated event names (default: tc_).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ds = xr.open_zarr(args.zarr_path)
    # ACE outputs may carry a leading sample/ensemble dimension; collapse it.
    if "sample" in ds.dims:
        ds = ds.isel(sample=0)

    prmsl_threshold = _detect_pressure_threshold(
        ds, args.prmsl_name, args.prmsl_threshold
    )
    config = DetectionConfig(
        prmsl_threshold=prmsl_threshold,
        wind_threshold=args.wind_threshold,
        min_abs_lat=args.lat_band[0],
        max_abs_lat=args.lat_band[1],
        min_region_size=args.min_region_size,
    )

    tracks = find_cyclone_tracks(
        ds,
        prmsl_name=args.prmsl_name,
        u10_name=args.u10_name,
        v10_name=args.v10_name,
        config=config,
        max_distance_deg=args.max_track_distance,
        max_gap=args.max_gap,
        min_track_length=args.min_track_length,
    )
    n_snapshots = sum(len(track) for track in tracks)
    print(f"Found {len(tracks)} cyclone track(s) over {n_snapshots} snapshot(s).")
    for i, track in enumerate(tracks, start=1):
        start = np.datetime64(track[0].timestamp, "s")
        end = np.datetime64(track[-1].timestamp, "s")
        print(
            f"  {args.name_prefix}{i:03d}: {len(track)} snapshots, "
            f"{start} -> {end}, min PRMSL {min(d.min_prmsl for d in track):.0f}"
        )

    events = build_events(
        tracks,
        box_size=args.box_size,
        n_samples=args.n_samples,
        name_prefix=args.name_prefix,
        lat_limit=args.lat_limit,
    )
    if not events:
        print("No cyclone events detected; no config written.")
        return

    base_config = _default_base_config()
    if args.base_config is not None:
        with open(args.base_config) as f:
            base_config = yaml.safe_load(f)

    out_config = build_config(
        base_config,
        coarse_data_path=args.zarr_path,
        events=events,
        experiment_dir=args.experiment_dir,
    )

    output_path = Path(args.output_config)
    with open(output_path, "w") as f:
        yaml.safe_dump(out_config, f, sort_keys=False)
    print(f"Wrote {len(events)} event(s) to {output_path}")
    if args.base_config is None:
        print(
            "NOTE: fill in the FILL_IN placeholders (model, experiment_dir) "
            "before running fme.downscaling.predict."
        )


if __name__ == "__main__":
    main()
