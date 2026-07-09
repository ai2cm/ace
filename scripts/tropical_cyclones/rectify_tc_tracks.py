"""Rectify high-resolution SLP-only TC tracks against coarse warm-core tracks.

The high-resolution 3h datasets (25km / 100km downscaling output) have no
upper-tropospheric temperature, so TC detection on them (``detect_tc_tracks.py
--no-warm-core``) is SLP-only and picks up *every* closed low -- extratropical
storms, polar lows, monsoon lows, terrain/equator artifacts -- not just TCs.

This script keeps only the genuine TCs by corroborating the fine SLP-only
detections against the coarse **warm-core** tracks (the 6h ~100km run, which
does apply the warm-core criterion and so identifies real TCs). For each coarse
track it rebuilds a 3h-resolution track from the fine centers:

  1. Anchor: at each coarse 6h timestamp, require a fine center within
     ``--anchor-radius`` degrees (great-circle) of the coarse position.
  2. In-between: for each fine timestamp strictly between two coarse times,
     take the nearest fine center within ``--window-radius`` degrees of the
     position linearly interpolated (in time) between the bracketing coarse
     points -- so the densified points stay on the same system.
  3. Edges: the rectified track spans exactly the coarse track's lifetime;
     in-between 3h points come from the fine detection, but the track ends when
     the coarse warm-core determination ends (no extratropical-remnant tail).

A coarse track is emitted only if at least ``--min-anchor-frac`` of its 6h
timestamps found an anchor (otherwise it is reported as unconfirmed).

Output: ``rectified_tracks.csv`` (track_id = coarse track id; one row per 3h
point with lat/lon/slp/wind from the fine data, the source fine track id, the
point type, and the distance to the coarse/interpolated position) plus a
printed coverage summary.

Usage:
    python rectify_tc_tracks.py \
        scratch/tc_full/tracks.csv \
        scratch/tc25_full/tracks.csv \
        scratch/tc25_rectified/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def great_circle_deg(
    lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in degrees from one point to an array of points.

    Uses the haversine formula so it is correct across the 0/360 longitude
    seam and at high latitude, unlike a plain lat/lon box.
    """
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    d = (
        np.sin((lat2r - lat1r) / 2) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2) ** 2
    )
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(d, 0.0, 1.0))))


def interp_position(
    lat_a: float, lon_a: float, lat_b: float, lon_b: float, frac: float
) -> tuple[float, float]:
    """Interpolate a position a fraction ``frac`` of the way from A to B.

    Longitude is interpolated along the shortest arc (wrap-aware) and returned
    in [0, 360), matching the TempestExtremes track convention.
    """
    lat = lat_a + (lat_b - lat_a) * frac
    dlon = ((lon_b - lon_a + 180.0) % 360.0) - 180.0
    lon = (lon_a + dlon * frac) % 360.0
    return lat, lon


def _index_fine_by_time(fine: pd.DataFrame) -> dict:
    """Group the fine detections by timestamp into numpy arrays for fast lookup.

    Returns ``{timestamp: dict(lat, lon, slp, wind, track_id)}`` so matching a
    coarse point is an O(centers-at-that-time) vectorized distance rather than
    a scan of the whole fine table.
    """
    index = {}
    for t, g in fine.groupby("time"):
        index[t] = {
            "lat": g["lat"].to_numpy(),
            "lon": g["lon"].to_numpy(),
            "slp": g["slp"].to_numpy(),
            "wind": g["wind"].to_numpy(),
            "track_id": g["track_id"].to_numpy(),
        }
    return index


def _nearest_center(
    index: dict, t: pd.Timestamp, lat: float, lon: float, radius: float
):
    """Nearest fine center to (lat, lon) at time ``t`` within ``radius`` degrees.

    Returns a dict row (with the distance) or None if no center is within range.
    """
    centers = index.get(t)
    if centers is None:
        return None
    dist = great_circle_deg(lat, lon, centers["lat"], centers["lon"])
    j = int(np.argmin(dist))
    if dist[j] > radius:
        return None
    return {
        "lat": float(centers["lat"][j]),
        "lon": float(centers["lon"][j]),
        "slp": float(centers["slp"][j]),
        "wind": float(centers["wind"][j]),
        "source_fine_track_id": int(centers["track_id"][j]),
        "dist_deg": float(dist[j]),
    }


def rectify_track(
    coarse_track: pd.DataFrame,
    fine_index: dict,
    fine_times: np.ndarray,
    anchor_radius: float,
    window_radius: float,
) -> tuple[list[dict], int]:
    """Build the 3h rectified points for one coarse warm-core track.

    Returns ``(rows, n_anchors_hit)``. ``rows`` are the densified track points
    (anchors + in-between), each tagged with ``point_type``; ``n_anchors_hit``
    is how many of the coarse 6h timestamps found an anchor within
    ``anchor_radius`` (used to decide whether the track is confirmed).
    """
    c = coarse_track.sort_values("time").reset_index(drop=True)
    rows: list[dict] = []
    n_anchors_hit = 0

    # Anchors: nearest fine center at each coarse timestamp, within anchor_radius.
    for _, cp in c.iterrows():
        hit = _nearest_center(
            fine_index, cp["time"], cp["lat"], cp["lon"], anchor_radius
        )
        if hit is not None:
            n_anchors_hit += 1
            rows.append({"time": cp["time"], "point_type": "anchor", **hit})

    # In-between: for each consecutive coarse pair, every fine timestamp strictly
    # between them, matched to the time-interpolated coarse position. This also
    # fills multi-step gaps (coarse maxgap can exceed 6h) automatically.
    for i in range(len(c) - 1):
        t0, t1 = c.loc[i, "time"], c.loc[i + 1, "time"]
        span = (t1 - t0).total_seconds()
        if span <= 0:
            continue
        between = fine_times[(fine_times > t0) & (fine_times < t1)]
        for t in between:
            frac = (t - t0).total_seconds() / span
            elat, elon = interp_position(
                c.loc[i, "lat"],
                c.loc[i, "lon"],
                c.loc[i + 1, "lat"],
                c.loc[i + 1, "lon"],
                frac,
            )
            hit = _nearest_center(fine_index, t, elat, elon, window_radius)
            if hit is not None:
                rows.append({"time": t, "point_type": "interp", **hit})

    rows.sort(key=lambda r: r["time"])
    return rows, n_anchors_hit


def rectify(
    coarse: pd.DataFrame,
    fine: pd.DataFrame,
    anchor_radius: float,
    window_radius: float,
    min_anchor_frac: float,
) -> tuple[pd.DataFrame, dict]:
    """Rectify all coarse tracks against the fine detections. See module docstring."""
    fine_index = _index_fine_by_time(fine)
    fine_times = np.array(sorted(fine_index.keys()))

    out_rows: list[dict] = []
    confirmed, unconfirmed = 0, 0
    anchor_fracs = []
    for tid, ctrack in coarse.groupby("track_id"):
        n_coarse = len(ctrack)
        rows, n_hit = rectify_track(
            ctrack, fine_index, fine_times, anchor_radius, window_radius
        )
        frac = n_hit / n_coarse if n_coarse else 0.0
        anchor_fracs.append(frac)
        if frac < min_anchor_frac:
            unconfirmed += 1
            continue
        confirmed += 1
        for r in rows:
            out_rows.append({"track_id": int(tid), **r})

    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df[
            [
                "track_id",
                "time",
                "lat",
                "lon",
                "slp",
                "wind",
                "point_type",
                "source_fine_track_id",
                "dist_deg",
            ]
        ]
    stats = {
        "coarse_tracks": coarse["track_id"].nunique(),
        "confirmed": confirmed,
        "unconfirmed": unconfirmed,
        "mean_anchor_frac": float(np.mean(anchor_fracs)) if anchor_fracs else 0.0,
        "rectified_points": len(df),
        "anchor_points": int((df["point_type"] == "anchor").sum())
        if not df.empty
        else 0,
        "interp_points": int((df["point_type"] == "interp").sum())
        if not df.empty
        else 0,
    }
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "coarse_csv", help="Coarse 6h warm-core tracks.csv (the TC truth)."
    )
    parser.add_argument("fine_csv", help="Fine 3h SLP-only tracks.csv to rectify.")
    parser.add_argument("out_dir", help="Directory for rectified_tracks.csv.")
    parser.add_argument(
        "--anchor-radius",
        type=float,
        default=1.0,
        help="Max great-circle degrees between a coarse 6h point and its fine "
        "center (same timestamp). Default 1.0.",
    )
    parser.add_argument(
        "--window-radius",
        type=float,
        default=2.5,
        help="Max great-circle degrees between an in-between fine center and the "
        "time-interpolated coarse position. Default 2.5.",
    )
    parser.add_argument(
        "--min-anchor-frac",
        type=float,
        default=0.5,
        help="Min fraction of a coarse track's 6h timestamps that must find an "
        "anchor for the track to be emitted. Default 0.5.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coarse = pd.read_csv(args.coarse_csv, parse_dates=["time"])
    fine = pd.read_csv(args.fine_csv, parse_dates=["time"])
    logger.info(
        "Coarse: %d tracks / %d pts | Fine: %d tracks / %d pts",
        coarse["track_id"].nunique(),
        len(coarse),
        fine["track_id"].nunique(),
        len(fine),
    )

    df, stats = rectify(
        coarse, fine, args.anchor_radius, args.window_radius, args.min_anchor_frac
    )
    csv_path = out_dir / "rectified_tracks.csv"
    df.to_csv(csv_path, index=False)

    logger.info("Wrote %s", csv_path)
    logger.info(
        "Confirmed %d / %d coarse tracks (%d unconfirmed, mean anchor match %.0f%%)",
        stats["confirmed"],
        stats["coarse_tracks"],
        stats["unconfirmed"],
        100 * stats["mean_anchor_frac"],
    )
    logger.info(
        "Rectified points: %d (%d anchor, %d in-between)",
        stats["rectified_points"],
        stats["anchor_points"],
        stats["interp_points"],
    )


if __name__ == "__main__":
    main()
