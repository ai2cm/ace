#!/usr/bin/env python
"""
Emit a ``fme.downscaling.predict`` config that downscales a moving box around
each tropical cyclone detected by
``scripts/tropical_cyclones/detect_tc_tracks.py``.

``detect_tc_tracks.py`` runs the TempestExtremes TC-detection recipe on an ACE
global output zarr and writes a ``tracks.csv`` with one row per track point
(columns: ``track_id, time, lon, lat, slp, wind``). This script reads that CSV
and turns each track into a named event (``<prefix><id>``). Each track point
becomes a separate ``EventConfig`` named ``<event_name>_<snapshot_date>`` whose
lat/lon extent is a box centred on that point. With ``save_generated_samples:
true`` set on every event, running predict.py writes one
``<event_name>_<snapshot_date>.nc`` per point, which ``stitch_event_movie.py``
then animates.

Usage:
    python downscale_events.py <ace_output.zarr> <tracks.csv> \\
        --output-config events.yaml [--base-config template.yaml] \\
        [--experiment-dir DIR] [--box-size 4.0] [--n-samples 8] ...

``<ace_output.zarr>`` is the coarse dataset ``detect_tc_tracks.py`` ran on; it is
set as the coarse data path in the generated config. ``<tracks.csv>`` is that
run's output.
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# EventConfig.date format expected by fme.downscaling.predict.
EVENT_DATE_FORMAT = "%Y-%m-%dT%H:%M"
# Compact date stamp embedded in the event name / output filename. Must contain
# a YYYYMMDD substring so downstream tooling (utils.find_event_files,
# plot_events.filename_to_datetime) can parse it.
EVENT_NAME_DATE_FORMAT = "%Y%m%dT%H"

# Columns written by detect_tc_tracks.parse_stitch_output.
REQUIRED_TRACK_COLUMNS = {"track_id", "time", "lat", "lon", "slp", "wind"}


@dataclasses.dataclass
class TrackPoint:
    """A single point along a detected cyclone track."""

    timestamp: np.datetime64
    lat: float
    lon: float
    slp: float
    wind: float


def load_tracks(csv_path: Path) -> list[tuple[int, list[TrackPoint]]]:
    """Read detect_tc_tracks.py output into ``(track_id, points)`` pairs.

    Expects the columns written by ``detect_tc_tracks.parse_stitch_output``:
    ``track_id, time, lon, lat, slp, wind``. The original ``track_id`` is
    preserved (rather than re-enumerated) so generated events trace back to the
    tracks.csv, and points within a track are ordered by time.
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])
    missing = REQUIRED_TRACK_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"{csv_path} is missing column(s) {sorted(missing)}; found "
            f"{list(df.columns)}. Expected detect_tc_tracks.py output."
        )
    tracks: list[tuple[int, list[TrackPoint]]] = []
    for track_id, track_df in df.groupby("track_id"):
        track_df = track_df.sort_values("time")
        points = [
            TrackPoint(
                timestamp=np.datetime64(row.time),
                lat=float(row.lat),
                lon=float(row.lon),
                slp=float(row.slp),
                wind=float(row.wind),
            )
            for row in track_df.itertuples(index=False)
        ]
        tracks.append((int(track_id), points))
    return tracks


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
    tracks: list[tuple[int, list[TrackPoint]]],
    box_size: float,
    n_samples: int,
    name_prefix: str,
    lat_limit: float,
) -> list[dict]:
    events: list[dict] = []
    for track_id, points in tracks:
        event_name = f"{name_prefix}{track_id:03d}"
        for pt in points:
            timestamp = np.datetime64(pt.timestamp, "s").item()
            lat_start, lat_stop = _box_extent(pt.lat, box_size, -lat_limit, lat_limit)
            lon_start = pt.lon - box_size / 2.0
            lon_stop = pt.lon + box_size / 2.0
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a downscaling predict config from "
        "detect_tc_tracks.py output."
    )
    parser.add_argument(
        "zarr_path",
        help="Path to the ACE global output zarr detect_tc_tracks ran on; "
        "written as the coarse data path in the generated config.",
    )
    parser.add_argument(
        "tracks_csv",
        help="tracks.csv written by scripts/tropical_cyclones/detect_tc_tracks.py.",
    )
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
        "--box-size",
        type=float,
        default=16,
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
    tracks = load_tracks(Path(args.tracks_csv))
    n_points = sum(len(points) for _, points in tracks)
    print(f"Loaded {len(tracks)} cyclone track(s) over {n_points} track point(s).")
    for track_id, points in tracks:
        start = np.datetime64(points[0].timestamp, "s")
        end = np.datetime64(points[-1].timestamp, "s")
        print(
            f"  {args.name_prefix}{track_id:03d}: {len(points)} points, "
            f"{start} -> {end}, min SLP {min(p.slp for p in points):.0f}"
        )

    events = build_events(
        tracks,
        box_size=args.box_size,
        n_samples=args.n_samples,
        name_prefix=args.name_prefix,
        lat_limit=args.lat_limit,
    )
    if not events:
        print("No cyclone events found in tracks.csv; no config written.")
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
