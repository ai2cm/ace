"""Demonstration of the radial TC verification metrics on real 25 km storms.

This exercises ``tc_radial_metrics`` end-to-end:

  1. Pick a few mature storm snapshots from the rectified 25 km tracks
     (``scratch/tc25_rectified_filt/rectified_tracks.csv``; lowest SLP per track).
  2. For each, read a 16 deg (~1600 km) patch centered on the storm from the
     25 km 3h zarr, and take that snapshot as the *target*.
  3. Fabricate a *synthetic ensemble* by random kernel smoothing / jitter of the
     target (explicitly NOT realistic -- this only validates the plumbing).
  4. Compute azimuthal-radial profiles for the target and every member, then
     radial CRPS(r) / RMSE(r) and the eval.md scorecard.
  5. Write a single compressed-NetCDF statistics summary (all profiles, ensemble
     mean/std, CRPS(r)/RMSE(r), ring counts, per-member scorecard, and provenance
     attrs incl. the generating command) plus per-storm figures, to
     ``scratch/tc25_eval_demo/``.

Gotcha: the zarr stores the whole globe as one chunk per timestep, so each
snapshot read fetches a global chunk (~100-160 MB/s). Keep ``--n-storms`` small.

Run (this VM):
    ~/miniconda3/bin/conda run -n fme python \
        scripts/tropical_cyclones/tc_radial_eval_demo.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cftime  # noqa: E402
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tc_radial_metrics as trm  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ZARR = (
    "gs://vcm-ml-scratch/andrep/2025-08-12-X-SHiELD-AMIP-downscaling-3h.zarr/"
)
DEFAULT_TRACKS = "scratch/tc25_rectified_filt/rectified_tracks.csv"
DEFAULT_OUT = "scratch/tc25_eval_demo"

# The three verified fields: (label, unit-for-plot, is-pressure).
FIELDS = ("wind", "pressure", "precip")
FIELD_UNITS = {"wind": "m/s", "pressure": "hPa", "precip": "mm/hr"}


def select_mature_points(tracks_csv: str, n_storms: int) -> pd.DataFrame:
    """Lowest-SLP (most mature) point of the strongest ``n_storms`` distinct tracks.

    Restricted away from the longitude seam (8..352 deg) and to |lat|<55 so a
    plain lat/lon patch slice is unambiguous for this demo.
    """
    df = pd.read_csv(tracks_csv, parse_dates=["time"])
    df = df[(df["lon"] > 8) & (df["lon"] < 352) & (df["lat"].abs() < 55)]
    peak_idx = df.groupby("track_id")["slp"].idxmin()
    peaks = df.loc[peak_idx].sort_values("slp").head(n_storms)
    return peaks.reset_index(drop=True)


def read_patch(
    ds: xr.Dataset, time, lat_c: float, lon_c: float, half_deg: float
) -> dict:
    """Read a (2*half_deg)-degree box around the center for one timestep.

    Returns 2D wind speed (m/s), SLP (Pa), precip (mm/hr) and the 1D lat/lon axes.
    The zarr time axis is a julian CFTimeIndex, so convert and select nearest.
    """
    ts = pd.Timestamp(time)
    t = cftime.DatetimeJulian(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    sub = ds.sel(time=t, method="nearest").sel(
        latitude=slice(lat_c - half_deg, lat_c + half_deg),
        longitude=slice(lon_c - half_deg, lon_c + half_deg),
    )
    u = sub["eastward_wind_at_ten_meters"].values
    v = sub["northward_wind_at_ten_meters"].values
    return {
        "wind": np.hypot(u, v),
        "pressure": sub["PRMSL"].values * 100.0,  # mb -> Pa
        "precip": sub["PRATEsfc"].values * 3600.0,  # kg/m2/s -> mm/hr
        "lats": sub["latitude"].values,
        "lons": sub["longitude"].values,
    }


def make_synthetic_ensemble(target: dict, n_members: int, seed: int) -> dict:
    """Fabricate an ensemble by random smoothing + jitter of the target fields.

    Deliberately crude (random Gaussian blur, a few-cell roll, and smoothed
    additive noise) -- enough spread to exercise CRPS/RMSE, not a real forecast.
    """
    rng = np.random.default_rng(seed)
    members: dict[str, list] = {f: [] for f in FIELDS}
    for _ in range(n_members):
        sigma = rng.uniform(0.5, 2.5)
        shift = (int(rng.integers(-3, 4)), int(rng.integers(-3, 4)))
        for f in FIELDS:
            base = target[f]
            noise = gaussian_filter(rng.standard_normal(base.shape), sigma=6.0)
            amp = 0.15 * float(np.nanstd(base))
            pert = (
                gaussian_filter(np.roll(base, shift, axis=(0, 1)), sigma) + amp * noise
            )
            if f == "precip":
                pert = np.clip(pert, 0.0, None)
            members[f].append(pert)
    return {f: np.stack(members[f]) for f in FIELDS}


def profiles_for_snapshot(
    target: dict, ensemble: dict, center: tuple[float, float], edges: np.ndarray
) -> dict:
    """Radial profiles (target + ensemble) for each field, sharing one center."""
    lats, lons = target["lats"], target["lons"]
    out: dict = {"r_km": trm.bin_centers(edges)}
    for f in FIELDS:
        out[f] = {
            "target": trm.compute_radial_profile(target[f], *center, lats, lons, edges),
            "ensemble": trm.compute_radial_profile(
                ensemble[f], *center, lats, lons, edges
            ),
        }
    return out


def compute_scorecard_stats(profs: dict) -> pd.DataFrame:
    """Per-member eval.md scorecard vs the target; return the member-wise table."""
    r = profs["r_km"]
    tgt = trm.StormProfiles(
        r_km=r,
        wind_mean=profs["wind"]["target"].mean,
        pressure_mean=profs["pressure"]["target"].mean,
        wind_var=profs["wind"]["target"].var,
        precip_mean=profs["precip"]["target"].mean,
    )
    ens_wind = profs["wind"]["ensemble"]
    ens_pres = profs["pressure"]["ensemble"]
    ens_precip = profs["precip"]["ensemble"]
    rows = []
    for m in range(ens_wind.mean.shape[0]):
        sim = trm.StormProfiles(
            r_km=r,
            wind_mean=ens_wind.mean[m],
            pressure_mean=ens_pres.mean[m],
            wind_var=ens_wind.var[m],
            precip_mean=ens_precip.mean[m],
        )
        rows.append(trm.scorecard(sim, tgt))
    return pd.DataFrame(rows)


def plot_raw_profiles(profs: dict, title: str, path: Path) -> None:
    """Raw azimuthal-mean profiles: target (bold) vs members (thin), per field."""
    r = profs["r_km"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, f in zip(axes, FIELDS):
        tgt = profs[f]["target"].mean
        ens = profs[f]["ensemble"].mean
        scale = 0.01 if f == "pressure" else 1.0  # Pa -> hPa for display
        for m in range(ens.shape[0]):
            ax.plot(r, ens[m] * scale, color="C0", alpha=0.25, lw=0.8)
        ax.plot(r, tgt * scale, color="k", lw=2.5, label="target")
        ax.plot([], [], color="C0", alpha=0.5, label="ensemble members")
        ax.set_xlabel("radius (km)")
        ax.set_ylabel(f"{f} ({FIELD_UNITS[f]})")
        ax.set_title(f)
        ax.legend(fontsize=8)
    fig.suptitle(f"Raw radial profiles -- {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_error_curves(profs: dict, title: str, path: Path) -> None:
    """CRPS(r) and RMSE(r) per field, plus target vs ensemble-mean with spread."""
    r = profs["r_km"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, f in zip(range(3), FIELDS):
        tgt = profs[f]["target"].mean
        ens = profs[f]["ensemble"].mean
        scale = 0.01 if f == "pressure" else 1.0
        scores = trm.radial_scores(tgt, ens, member_axis=0)
        ens_mean = np.nanmean(ens, axis=0)
        ens_std = np.nanstd(ens, axis=0)

        top = axes[0, j]
        top.plot(r, tgt * scale, "k", lw=2, label="target")
        top.plot(r, ens_mean * scale, "C0", lw=1.5, label="ensemble mean")
        top.fill_between(
            r,
            (ens_mean - ens_std) * scale,
            (ens_mean + ens_std) * scale,
            color="C0",
            alpha=0.2,
            label="+/-1 std",
        )
        top.set_title(f"{f} profile")
        top.set_ylabel(f"{f} ({FIELD_UNITS[f]})")
        top.legend(fontsize=8)

        bot = axes[1, j]
        bot.plot(r, scores["crps"] * scale, "C3", lw=1.8, label="CRPS(r)")
        bot.plot(r, scores["rmse"] * scale, "C4", lw=1.8, ls="--", label="RMSE(r)")
        bot.set_xlabel("radius (km)")
        bot.set_ylabel(f"error ({FIELD_UNITS[f]})")
        bot.set_title(f"{f} CRPS / RMSE")
        bot.legend(fontsize=8)
    fig.suptitle(f"Ensemble radial error -- {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# Physical units of each stored field (SLP kept in Pa, as the metrics use it).
FIELD_STORE_UNITS = {"wind": "m s-1", "pressure": "Pa", "precip": "mm hr-1"}


def collect_storm_record(row, profs: dict, center: tuple[float, float]) -> dict:
    """Gather every profile / error array for one storm into a plain dict.

    Returned arrays are later stacked across storms into the summary Dataset.
    """
    r = profs["r_km"]
    fields: dict = {}
    for f in FIELDS:
        tgt = profs[f]["target"].mean
        ens = profs[f]["ensemble"].mean  # (members, nbins)
        scores = trm.radial_scores(tgt, ens, member_axis=0)
        fields[f] = {
            "target_mean": tgt,
            "target_var": profs[f]["target"].var,
            "ensemble_mean": ens,
            "ens_mean": np.nanmean(ens, axis=0),
            "ens_std": np.nanstd(ens, axis=0),
            "crps": scores["crps"],
            "rmse": scores["rmse"],
            "count": profs[f]["target"].count.astype(float),
        }
    rmax_members = np.array(
        [
            trm.radius_of_max_wind(profs["wind"]["ensemble"].mean[m], r)[0]
            for m in range(profs["wind"]["ensemble"].mean.shape[0])
        ]
    )
    rmax_target = trm.radius_of_max_wind(profs["wind"]["target"].mean, r)[0]
    return {
        "track_id": int(row.track_id),
        "time": str(pd.Timestamp(row.time)),
        "track_lat": float(row.lat),
        "track_lon": float(row.lon),
        "center_lat": float(center[0]),
        "center_lon": float(center[1]),
        "track_slp_hpa": float(row.slp) / 100.0,
        "track_wind_ms": float(row.wind),
        "fields": fields,
        "scorecard": compute_scorecard_stats(profs),
        "crps_R_max_km": float(trm.crps_ensemble(rmax_target, rmax_members)),
    }


def build_summary_dataset(records: list, r_km: np.ndarray, attrs: dict) -> xr.Dataset:
    """Assemble all storm records into one NetCDF-ready statistics Dataset.

    Dims: ``storm``, ``radius`` (km), ``member``. Per-field profiles, ensemble
    mean/std, CRPS(r)/RMSE(r) and ring counts; per-(storm, member) scorecard
    metrics; and per-storm track metadata. ``attrs`` become global provenance.
    """
    n_members = records[0]["fields"]["wind"]["ensemble_mean"].shape[0]
    data_vars: dict = {}

    def stack(getter):
        return np.stack([getter(rec) for rec in records])

    meta = {
        "track_id": ("storm", stack(lambda r: r["track_id"])),
        "time": ("storm", stack(lambda r: r["time"])),
        "track_lat": ("storm", stack(lambda r: r["track_lat"])),
        "track_lon": ("storm", stack(lambda r: r["track_lon"])),
        "center_lat": ("storm", stack(lambda r: r["center_lat"])),
        "center_lon": ("storm", stack(lambda r: r["center_lon"])),
        "track_slp_hpa": ("storm", stack(lambda r: r["track_slp_hpa"])),
        "track_wind_ms": ("storm", stack(lambda r: r["track_wind_ms"])),
        "crps_R_max_km": ("storm", stack(lambda r: r["crps_R_max_km"])),
    }
    data_vars.update(meta)

    for f in FIELDS:
        for key in (
            "target_mean",
            "target_var",
            "ens_mean",
            "ens_std",
            "crps",
            "rmse",
            "count",
        ):
            arr = stack(lambda r, f=f, key=key: r["fields"][f][key])
            var = xr.Variable(("storm", "radius"), arr)
            if key in ("target_mean", "ens_mean", "crps", "rmse"):
                var.attrs["units"] = FIELD_STORE_UNITS[f]
            data_vars[f"{f}_{key}"] = var
        ens = stack(lambda r, f=f: r["fields"][f]["ensemble_mean"])
        data_vars[f"{f}_ensemble_mean"] = xr.Variable(
            ("storm", "member", "radius"), ens, {"units": FIELD_STORE_UNITS[f]}
        )

    for key in records[0]["scorecard"].columns:
        arr = stack(lambda r, key=key: r["scorecard"][key].to_numpy())
        data_vars[key] = xr.Variable(("storm", "member"), arr)

    ds = xr.Dataset(
        data_vars,
        coords={
            "storm": np.arange(len(records)),
            "radius": ("radius", np.asarray(r_km), {"units": "km"}),
            "member": np.arange(n_members),
        },
        attrs=attrs,
    )
    return ds


def save_summary_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Write the summary Dataset to compressed NetCDF."""
    encoding = {
        v: {"zlib": True, "complevel": 4}
        for v in ds.data_vars
        if np.issubdtype(ds[v].dtype, np.floating)
    }
    ds.to_netcdf(path, encoding=encoding)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--tracks", default=DEFAULT_TRACKS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--n-storms", type=int, default=4)
    parser.add_argument("--n-members", type=int, default=20)
    parser.add_argument("--patch-halfwidth-deg", type=float, default=8.0)
    parser.add_argument("--dr-km", type=float, default=25.0)
    parser.add_argument("--r-max-km", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = trm.radial_bin_edges(args.dr_km, args.r_max_km)

    command = "python " + " ".join(sys.argv)
    logger.info("Generating command: %s", command)
    logger.info("Tracks source: %s", args.tracks)

    peaks = select_mature_points(args.tracks, args.n_storms)
    logger.info(
        "Selected %d storms:\n%s",
        len(peaks),
        peaks[["track_id", "time", "lat", "lon", "slp", "wind"]].to_string(),
    )

    logger.info("Opening zarr %s", args.zarr)
    ds = xr.open_zarr(args.zarr)

    records = []
    for _, row in peaks.iterrows():
        tag = f"trk{int(row.track_id)}_{pd.Timestamp(row.time).strftime('%Y%m%d%H')}"
        title = (
            f"track {int(row.track_id)} @ {row.time} ({row.lat:.1f}N,{row.lon:.1f}E)"
        )
        logger.info("Reading patch for %s", title)
        target = read_patch(ds, row.time, row.lat, row.lon, args.patch_halfwidth_deg)
        ensemble = make_synthetic_ensemble(target, args.n_members, args.seed)

        center = trm.refine_center_min_slp(
            target["pressure"], target["lats"], target["lons"], (row.lat, row.lon)
        )
        profs = profiles_for_snapshot(target, ensemble, center, edges)
        plot_raw_profiles(profs, title, out_dir / f"raw_profiles_{tag}.png")
        plot_error_curves(profs, title, out_dir / f"error_curves_{tag}.png")

        record = collect_storm_record(row, profs, center)
        records.append(record)
        logger.info(
            "%s: CRPS(R_max) = %.1f km; member-mean scorecard:\n%s",
            tag,
            record["crps_R_max_km"],
            record["scorecard"].mean().to_string(),
        )

    attrs = {
        "title": "Radial TC structural-verification statistics (synthetic demo)",
        "description": (
            "Azimuthal-radial profiles, ensemble CRPS(r)/RMSE(r) and eval.md "
            "scorecard for a few mature storms. The ensemble is SYNTHETIC "
            "(random kernel smoothing of the target) -- plumbing check only."
        ),
        "command": command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_zarr": args.zarr,
        "tracks_source": args.tracks,
        "crps_estimator": "fair (matches fme/downscaling/metrics_and_maths.py)",
        "rmse_definition": "abs error of the ensemble mean, |mean_m(X_m) - x|",
        "n_members": args.n_members,
        "dr_km": args.dr_km,
        "r_max_km": args.r_max_km,
        "patch_halfwidth_deg": args.patch_halfwidth_deg,
        "seed": args.seed,
    }
    summary = build_summary_dataset(records, trm.bin_centers(edges), attrs)
    nc_path = out_dir / "tc_radial_stats.nc"
    save_summary_netcdf(summary, nc_path)
    logger.info(
        "Wrote summary NetCDF %s and figures to %s",
        nc_path.resolve(),
        out_dir.resolve(),
    )


if __name__ == "__main__":
    main()
