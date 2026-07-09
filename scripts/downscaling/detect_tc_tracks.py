"""
Detect tropical cyclone tracks in ACE output using TempestExtremes.

This script runs the standard TempestExtremes TC detection recipe
(DetectNodes -> StitchNodes) on a zarr dataset written by ACE inference. It
follows the default settings recommended for TC detection, with one
modification for ACE: because ACE does not predict geopotential height, the
warm-core-aloft criterion uses an upper-tropospheric temperature decrease
(``T3``, the ~250-400 hPa layer, saved directly as ``air_temperature_3``)
instead of a Z300-Z500 thickness decrease. The required temperature drop is
0.4 K, approximately equivalent to the 58.8 m^2 s^-2 thickness decrease of the
standard recipe under hydrostatic balance.

The pipeline is:
  1. Split the zarr's time axis into shard-sized bundles and, per bundle,
     assemble the fields TempestExtremes needs (sea-level pressure, 10m u/v
     wind, and -- for the warm-core recipe -- T3), write them to a transient
     NetCDF, run ``DetectNodes`` on it, and delete the NetCDF. Bundles are
     processed in parallel across ``--workers`` processes, so peak disk stays
     bounded (~``workers`` x bundle) instead of materializing the whole dataset.
  2. Run ``StitchNodes`` to link the candidate nodes into tracks (reading the
     per-bundle candidate files in time order, so tracks span bundles).
  3. Parse the StitchNodes output into a tidy CSV (one row per track point).
  4. Pickle a per-track xarray ``.sel(..., method="nearest")`` kwargs dict
     for pulling storm-following data out of the original ACE dataset
     (on by default; disable with ``--no-write-sel-args``).

For datasets without an upper-tropospheric temperature field (e.g. the 3h
downscaling outputs), pass ``--no-warm-core`` to run the SLP-only recipe
(closed SLP contour + wind), and set ``--timefilter`` to the data cadence.

TempestExtremes must be installed and on ``PATH``, or its binaries passed via
``--detect-exe``/``--stitch-exe``. Build it from source with
``make -C scripts/downscaling tc_deps`` (see that target for dependencies).

Usage examples:
    # Basic run on a local zarr, writing intermediates + tracks to out/
    python detect_tc_tracks.py /path/to/ace_output.zarr out/

    # Select a specific ensemble member and time range
    python detect_tc_tracks.py gs://bucket/ace_output.zarr out/ \
        --sample 2 --time-start 2020-01-01 --time-end 2020-12-31

    # 3h SLP-only run (no upper-air T), shard-aligned bundles, 6 parallel workers
    python detect_tc_tracks.py gs://bucket/downscaling_3h.zarr out/ \
        --no-warm-core --timefilter 3hr --chunk-size 128 --workers 6 \
        --u-var eastward_wind_at_ten_meters --v-var northward_wind_at_ten_meters

    # Keep the per-bundle NetCDF intermediates instead of deleting them
    python detect_tc_tracks.py /path/to/ace_output.zarr out/ --keep-netcdf

    # Skip writing per-track xarray .sel kwargs
    python detect_tc_tracks.py /path/to/ace_output.zarr out/ --no-write-sel-args
"""

import argparse
import logging
import multiprocessing
import pickle
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Default TC-detection recipe parameters -------------------------------
# These reproduce the TempestExtremes default TC settings, with the T3
# substitution for the warm-core-aloft criterion (see module docstring).
PSL_CONTOUR = "PSL,200.0,5.5,0"  # SLP rises >= 200 Pa within 5.5 deg GCD
T3_CONTOUR = "T3,-0.4,6.5,1.0"  # T3 falls >= 0.4 K within 6.5 deg (excl. inner 1 deg)
MERGE_DIST = 6.0  # merge candidates within 6.0 deg GCD
STITCH_RANGE = 8.0  # max deg between consecutive track points
STITCH_MINTIME = "54h"  # minimum track duration
STITCH_MAXGAP = "24h"  # maximum time gap within a track
# Track must have wind >= 10 m/s for >= 10 steps and stay within +/- 50 lat.
STITCH_THRESHOLD = "wind,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10"
IN_FMT = "lon,lat,slp,wind"

_HPA_UNITS = {"hpa", "mb", "millibar", "millibars"}

# Matches TC_INSTALL_PREFIX's default in scripts/downscaling/Makefile.
_TC_INSTALL_BIN = Path.home() / ".local" / "tempestextremes" / "bin"

# (lat, lon) coordinate name pairs to try, in order, when --lat-name/--lon-name
# are not given.
LAT_LON_NAME_CANDIDATES = [
    ("grid_yt", "grid_xt"),
    ("lat", "lon"),
    ("latitude", "longitude"),
]


def detect_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    """Guess the lat/lon coordinate names from a set of known conventions."""
    for lat_name, lon_name in LAT_LON_NAME_CANDIDATES:
        if lat_name in ds.variables and lon_name in ds.variables:
            return lat_name, lon_name
    raise ValueError(
        "Could not auto-detect lat/lon coordinate names from any of "
        f"{LAT_LON_NAME_CANDIDATES}; pass --lat-name/--lon-name explicitly. "
        f"Available variables: {list(ds.variables)}"
    )


def _pressure_to_pa(da: xr.DataArray) -> xr.DataArray:
    """Convert a pressure DataArray to Pascals if its units attr indicates hPa/mb.

    TempestExtremes' closed-contour thresholds (e.g. PSL_CONTOUR) are
    calibrated in Pa; ACE pressure fields like PRMSL are commonly stored in
    hPa/mb, and silently searching that data in the wrong units means the
    threshold is effectively never met.
    """
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in _HPA_UNITS:
        da = da * 100.0
    return da


def build_te_dataset(
    ds: xr.Dataset,
    psl_var: str,
    psfc_var: str,
    t3_var: str,
    u_var: str,
    v_var: str,
    sample: int,
    warm_core: bool = True,
) -> xr.Dataset:
    """Assemble the TempestExtremes input fields from an ACE dataset.

    Selects sea-level pressure (preferring ``psl_var`` if present, else falling
    back to surface pressure), the 10m winds, and (when ``warm_core``) the T3
    temperature layer, renaming them to the canonical names used in the
    DetectNodes command. Pass ``warm_core=False`` for datasets without an
    upper-tropospheric temperature field (e.g. the 3h downscaling outputs),
    which run the SLP-only recipe and so do not need ``t3_var``.
    """
    if "sample" in ds.dims:
        logger.info("Selecting sample=%d of %d", sample, ds.sizes["sample"])
        ds = ds.isel(sample=sample)

    if psl_var in ds:
        psl = ds[psl_var]
        logger.info("Using %s for sea-level pressure", psl_var)
    elif psfc_var in ds:
        psl = ds[psfc_var]
        logger.warning(
            "%s not found; using surface pressure %s as PSL "
            "(acceptable over ocean where TCs occur)",
            psl_var,
            psfc_var,
        )
    else:
        raise KeyError(
            f"Neither {psl_var!r} nor {psfc_var!r} found in dataset; "
            f"available: {list(ds.data_vars)}"
        )
    orig_units = psl.attrs.get("units")
    psl = _pressure_to_pa(psl)
    if str(orig_units).strip().lower() in _HPA_UNITS:
        logger.info("Converted PSL from %s to Pa", orig_units)

    required = (u_var, v_var) + ((t3_var,) if warm_core else ())
    for name in required:
        if name not in ds:
            raise KeyError(
                f"{name!r} not found in dataset; available: {list(ds.data_vars)}"
            )

    data = {"PSL": psl, "U10": ds[u_var], "V10": ds[v_var]}
    if warm_core:
        data["T3"] = ds[t3_var]
    out = xr.Dataset(data)
    out["PSL"].attrs["units"] = "Pa"
    out["U10"].attrs["units"] = "m/s"
    out["V10"].attrs["units"] = "m/s"
    if warm_core:
        out["T3"].attrs["units"] = "K"
    return out


# Calendars TempestExtremes' Time parser reads without shifting the date.
_TE_READABLE_CALENDARS = frozenset({"standard", "gregorian", "proleptic_gregorian"})


def _normalize_time_for_tempest(ds: xr.Dataset) -> xr.Dataset:
    """Relabel a non-standard-calendar time axis so TempestExtremes reads it right.

    TempestExtremes mis-reads a non-standard calendar (notably ``julian``, which
    ACE output zarrs carry) on a "seconds since ..." axis and stamps every
    detected node with a date offset from the true model time (~11 days for
    julian in the 2010s). That silently shifts every point in tracks.csv, so a
    downscaling box built from it lands on the wrong model timestep. Relabel the
    time coordinate onto the proleptic Gregorian calendar, preserving the
    wall-clock (year/month/day/hour/...) values, before writing the inputs.

    No-op for a datetime64 axis or one already on a TempestExtremes-readable
    calendar.
    """
    time_index = ds.indexes["time"]
    calendar = getattr(time_index, "calendar", None)
    if calendar is None or calendar in _TE_READABLE_CALENDARS:
        return ds
    gregorian = pd.DatetimeIndex(
        [
            pd.Timestamp(
                year=t.year,
                month=t.month,
                day=t.day,
                hour=t.hour,
                minute=t.minute,
                second=t.second,
            )
            for t in time_index
        ]
    )
    logger.info(
        "Relabeling %r-calendar time axis onto proleptic_gregorian for "
        "TempestExtremes (wall-clock preserved)",
        calendar,
    )
    return ds.assign_coords(time=gregorian)


def _te_time_encoding(ds: xr.Dataset) -> dict:
    """CF time units/calendar that TempestExtremes will actually accept.

    TempestExtremes' Time::FromCFCompliantUnitsOffsetInt only recognizes
    "days|hours|minutes|seconds since ..." unit prefixes (TimeObj.cpp); left to
    its own defaults, xarray can encode datetime64[ns] data as "nanoseconds
    since ...", which TempestExtremes rejects with 'Unknown "time::units"
    format'. Forcing seconds-since-epoch avoids that. The calendar is taken from
    the (already normalized, see _normalize_time_for_tempest) time axis and
    constrained to one TempestExtremes reads correctly.
    """
    calendar = ds["time"].encoding.get("calendar", "proleptic_gregorian")
    if calendar not in _TE_READABLE_CALENDARS:
        calendar = "proleptic_gregorian"
    return {
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": calendar,
        "dtype": "int64",
    }


def _check_exe(exe: str) -> None:
    """Raise a helpful error if a TempestExtremes binary is not on PATH.

    TempestExtremes is conda-only (not pip-installable); point the user at the
    make target that installs it.
    """
    if shutil.which(exe) is None and not Path(exe).is_file():
        raise FileNotFoundError(
            f"TempestExtremes executable {exe!r} not found on PATH. "
            "Build and install it from source with:\n"
            "    make -C scripts/downscaling tc_deps\n"
            "then pass its location via --detect-exe/--stitch-exe, e.g.\n"
            "    --detect-exe ~/.local/tempestextremes/bin/DetectNodes\n"
            "    --stitch-exe ~/.local/tempestextremes/bin/StitchNodes"
        )


@dataclass(frozen=True)
class BundleSpec:
    """Everything a worker process needs to detect one time-bundle.

    Passed by value to worker processes, so every field is a plain
    picklable scalar/string. The worker re-opens the zarr and re-applies the
    same deterministic time subset, so bundles are reproducible regardless of
    which process runs them.
    """

    zarr: str
    time_start: str | None
    time_end: str | None
    sample: int
    start: int
    size: int
    psl_var: str
    psfc_var: str
    t3_var: str
    u_var: str
    v_var: str
    warm_core: bool
    lat_name: str
    lon_name: str
    timefilter: str
    detect_exe: str
    nc_dir: str
    nodes_dir: str
    keep_netcdf: bool


def _detect_on_file(
    nc_path: Path,
    out_path: Path,
    exe: str,
    lat_name: str,
    lon_name: str,
    warm_core: bool,
    timefilter: str,
    logdir: Path,
) -> None:
    """Run DetectNodes on a single NetCDF bundle, writing one candidate file.

    The closed-contour command includes the T3 warm-core term only when
    ``warm_core`` is set; otherwise it is the SLP-only recipe used for datasets
    without upper-tropospheric temperature.
    """
    _check_exe(exe)
    contour = f"{PSL_CONTOUR};{T3_CONTOUR}" if warm_core else PSL_CONTOUR
    cmd = [
        exe,
        "--in_data",
        str(nc_path),
        "--out",
        str(out_path),
        "--timefilter",
        timefilter,
        "--searchbymin",
        "PSL",
        "--closedcontourcmd",
        contour,
        "--mergedist",
        str(MERGE_DIST),
        "--outputcmd",
        "PSL,min,0;_VECMAG(U10,V10),max,2.0",
        "--latname",
        lat_name,
        "--lonname",
        lon_name,
        "--logdir",
        str(logdir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"DetectNodes failed on {nc_path.name} (exit {result.returncode}):\n"
            + result.stderr[-2000:]
        )


def _process_bundle(spec: BundleSpec) -> tuple[int, str]:
    """Read one time-bundle, write NetCDF, run DetectNodes, delete NetCDF.

    Returns ``(start, candidate_path)``. The NetCDF intermediate is removed in
    a ``finally`` (unless ``keep_netcdf``) so a failure cannot orphan it, and
    peak disk stays bounded to roughly ``workers x bundle_size``.
    """
    nc_path = Path(spec.nc_dir) / f"te_input_{spec.start:06d}.nc"
    cand_path = Path(spec.nodes_dir) / f"cand_{spec.start:06d}.dat"
    try:
        if not (spec.keep_netcdf and nc_path.exists()):
            ds = xr.open_zarr(spec.zarr)
            if spec.time_start is not None or spec.time_end is not None:
                ds = ds.sel(time=slice(spec.time_start, spec.time_end))
            te = build_te_dataset(
                ds,
                psl_var=spec.psl_var,
                psfc_var=spec.psfc_var,
                t3_var=spec.t3_var,
                u_var=spec.u_var,
                v_var=spec.v_var,
                sample=spec.sample,
                warm_core=spec.warm_core,
            )
            sub = te.isel(time=slice(spec.start, spec.start + spec.size)).load()
            sub = _normalize_time_for_tempest(sub)
            sub.to_netcdf(nc_path, encoding={"time": _te_time_encoding(sub)})
        _detect_on_file(
            nc_path,
            cand_path,
            spec.detect_exe,
            spec.lat_name,
            spec.lon_name,
            spec.warm_core,
            spec.timefilter,
            Path(spec.nodes_dir),
        )
        return spec.start, str(cand_path)
    finally:
        if not spec.keep_netcdf and nc_path.exists():
            nc_path.unlink()


def run_bundled_detection(
    zarr: str,
    out_dir: Path,
    n_time: int,
    chunk_size: int,
    workers: int,
    *,
    time_start: str | None,
    time_end: str | None,
    sample: int,
    psl_var: str,
    psfc_var: str,
    t3_var: str,
    u_var: str,
    v_var: str,
    warm_core: bool,
    lat_name: str,
    lon_name: str,
    timefilter: str,
    detect_exe: str,
    keep_netcdf: bool,
) -> Path:
    """Detect candidate nodes in parallel over shard-sized time-bundles.

    Each bundle is read from the zarr, written to a transient NetCDF, consumed
    by its own DetectNodes, and deleted -- interleaved across ``workers``
    processes so peak disk is bounded and DetectNodes runs concurrently.
    Returns the path to a file listing the per-bundle candidate files in time
    order, ready for StitchNodes' ``--in_list``.
    """
    _check_exe(detect_exe)
    nc_dir = out_dir / "netcdf"
    nodes_dir = out_dir / "nodes"
    nc_dir.mkdir(parents=True, exist_ok=True)
    nodes_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        BundleSpec(
            zarr=zarr,
            time_start=time_start,
            time_end=time_end,
            sample=sample,
            start=start,
            size=chunk_size,
            psl_var=psl_var,
            psfc_var=psfc_var,
            t3_var=t3_var,
            u_var=u_var,
            v_var=v_var,
            warm_core=warm_core,
            lat_name=lat_name,
            lon_name=lon_name,
            timefilter=timefilter,
            detect_exe=detect_exe,
            nc_dir=str(nc_dir),
            nodes_dir=str(nodes_dir),
            keep_netcdf=keep_netcdf,
        )
        for start in range(0, n_time, chunk_size)
    ]
    logger.info(
        "DetectNodes over %d bundles of %d timesteps, %d worker(s), warm_core=%s",
        len(specs),
        chunk_size,
        workers,
        warm_core,
    )

    results: list[tuple[int, str]] = []
    if workers == 1:
        for spec in specs:
            results.append(_process_bundle(spec))
            logger.info("  bundle %06d done", spec.start)
    else:
        # "spawn" (not the Linux default "fork"): the parent has already
        # initialized gcsfs' asyncio/aiohttp state via xr.open_zarr, and
        # forking that into workers corrupts it (BrokenProcessPool). Spawned
        # workers start a clean interpreter and open the zarr themselves.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {ex.submit(_process_bundle, s): s.start for s in specs}
            for fut in as_completed(futures):
                results.append(fut.result())
                logger.info(
                    "  bundle %06d done (%d/%d)",
                    futures[fut],
                    len(results),
                    len(specs),
                )

    # Sort by bundle start so StitchNodes reads candidates in time order, and
    # drop any that DetectNodes left empty/absent.
    cand_files = [c for _, c in sorted(results) if Path(c).is_file()]
    if not cand_files:
        raise FileNotFoundError(
            "No candidate files were produced; check the DetectNodes output."
        )
    node_list_path = out_dir / "candidate_files.txt"
    node_list_path.write_text("\n".join(cand_files) + "\n")
    logger.info(
        "Wrote candidate file list %s (%d files)", node_list_path, len(cand_files)
    )
    return node_list_path


def run_stitch_nodes(in_list: Path, out_path: Path, exe: str) -> None:
    """Run StitchNodes to link candidate centers into tracks.

    ``in_list`` lists the per-chunk candidate files from DetectNodes;
    StitchNodes reads them as one continuous time series via ``--in_list`` so
    tracks may span chunk boundaries.
    """
    _check_exe(exe)
    cmd = [
        exe,
        "--in_list",
        str(in_list),
        "--out",
        str(out_path),
        "--in_fmt",
        IN_FMT,
        "--range",
        str(STITCH_RANGE),
        "--mintime",
        STITCH_MINTIME,
        "--maxgap",
        STITCH_MAXGAP,
        "--threshold",
        STITCH_THRESHOLD,
    ]
    logger.info("Running StitchNodes: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_stitch_output(path: Path) -> pd.DataFrame:
    """Parse a StitchNodes ASCII track file into a tidy DataFrame.

    The format is a sequence of tracks, each introduced by a ``start`` header
    line, followed by one line per track point:

        start  <npts>  <year> <month> <day> <hour>
            <i> <j> <lon> <lat> <slp> <wind> <year> <month> <day> <hour>
            ...

    Returns columns: track_id, time, lon, lat, slp (Pa), wind (m/s).
    """
    rows = []
    track_id = -1
    for line in path.read_text().splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "start":
            track_id += 1
            continue
        # Point line: i, j, lon, lat, slp, wind, year, month, day, hour
        _i, _j, lon, lat, slp, wind, year, month, day, hour = tokens[:10]
        rows.append(
            {
                "track_id": track_id,
                "time": pd.Timestamp(
                    year=int(year),
                    month=int(month),
                    day=int(day),
                    hour=int(hour),
                ),
                "lon": float(lon),
                "lat": float(lat),
                "slp": float(slp),
                "wind": float(wind),
            }
        )
    df = pd.DataFrame(rows)
    logger.info(
        "Parsed %d track points across %d tracks",
        len(df),
        df["track_id"].nunique() if not df.empty else 0,
    )
    return df


def track_to_sel_kwargs(
    track_df: pd.DataFrame,
    lat_name: str,
    lon_name: str,
    point_dim: str = "track_point",
) -> dict:
    """Build xarray ``.sel(..., method="nearest")`` kwargs following one track.

    Wraps the track's time/lat/lon values in DataArrays sharing ``point_dim``
    so that ``ds.sel(**kwargs, method="nearest")`` does pointwise selection
    along the track (one point per track timestep) instead of an outer
    product over every combination of the three coordinates.

    ``lat_name``/``lon_name`` should match the coordinate names of the
    dataset this will be applied to (e.g. the ``--lat-name``/``--lon-name``
    used to build the TempestExtremes input, such as ``grid_yt``/``grid_xt``
    for X-SHiELD output), not the ``lat``/``lon`` column names in the tracks
    DataFrame itself.
    """
    track_df = track_df.sort_values("time")
    return {
        "time": xr.DataArray(track_df["time"].values, dims=point_dim),
        lat_name: xr.DataArray(track_df["lat"].values, dims=point_dim),
        lon_name: xr.DataArray(track_df["lon"].values, dims=point_dim),
    }


def write_track_sel_args(
    df: pd.DataFrame, out_dir: Path, lat_name: str, lon_name: str
) -> Path:
    """Write one pickled xarray .sel kwargs dict per track to out_dir/sel_args.

    Each file holds the dict returned by track_to_sel_kwargs for that track,
    ready to use as ``ds.sel(**kwargs, method="nearest")`` against the
    original ACE dataset. Returns the sel_args directory.
    """
    sel_dir = out_dir / "sel_args"
    sel_dir.mkdir(parents=True, exist_ok=True)
    for track_id, track_df in df.groupby("track_id"):
        kwargs = track_to_sel_kwargs(track_df, lat_name, lon_name)
        path = sel_dir / f"track_{track_id:04d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)
    logger.info("Wrote %d track sel-arg files to %s", df["track_id"].nunique(), sel_dir)
    return sel_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("zarr", help="Path/URI of the ACE output zarr dataset.")
    parser.add_argument("out_dir", help="Directory for intermediates and tracks.")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Ensemble member to select if a 'sample' dim is present.",
    )
    parser.add_argument("--time-start", default=None, help="Time subset start.")
    parser.add_argument("--time-end", default=None, help="Time subset end.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=112,
        help="Timesteps per NetCDF file (112 = 28 days at 6-hourly).",
    )
    parser.add_argument("--psl-var", default="PRMSL", help="Sea-level pressure var.")
    parser.add_argument(
        "--psfc-var",
        default="PRESsfc",
        help="Surface pressure var (fallback if PSL var absent).",
    )
    parser.add_argument(
        "--t3-var",
        default="air_temperature_3",
        help="Upper-tropospheric temperature layer (T3) var.",
    )
    parser.add_argument("--u-var", default="UGRD10m", help="10m eastward wind var.")
    parser.add_argument("--v-var", default="VGRD10m", help="10m northward wind var.")
    parser.add_argument(
        "--lat-name",
        default=None,
        help="Latitude coord name. If not set, auto-detected from known "
        f"conventions: {LAT_LON_NAME_CANDIDATES}.",
    )
    parser.add_argument(
        "--lon-name",
        default=None,
        help="Longitude coord name. If not set, auto-detected alongside " "--lat-name.",
    )
    parser.add_argument(
        "--detect-exe",
        default=str(_TC_INSTALL_BIN / "DetectNodes"),
        help="Path to DetectNodes (default: install location of "
        "`make -C scripts/downscaling tc_deps`).",
    )
    parser.add_argument(
        "--stitch-exe",
        default=str(_TC_INSTALL_BIN / "StitchNodes"),
        help="Path to StitchNodes (default: install location of "
        "`make -C scripts/downscaling tc_deps`).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for the read->NetCDF->DetectNodes->delete "
        "pipeline (one time-bundle per task). Bound by RAM: each holds ~one "
        "bundle in memory (~1.6 GB for a 128-step 720x1440 bundle).",
    )
    parser.add_argument(
        "--warm-core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the T3 upper-tropospheric warm-core criterion (standard TC "
        "recipe). Use --no-warm-core for datasets without upper-air temperature "
        "(e.g. 3h downscaling outputs); that runs the SLP-only recipe.",
    )
    parser.add_argument(
        "--timefilter",
        default="6hr",
        help="DetectNodes --timefilter (e.g. '6hr' for 6-hourly data, '3hr' for "
        "3-hourly). Restricts detection to timesteps on this cadence.",
    )
    parser.add_argument(
        "--keep-netcdf",
        action="store_true",
        help="Keep the per-bundle NetCDF intermediates instead of deleting each "
        "after its DetectNodes finishes (default deletes, bounding peak disk).",
    )
    parser.add_argument(
        "--write-sel-args",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For each track, pickle an xarray .sel(..., method='nearest') "
            "kwargs dict (following the track's time/lat/lon) to "
            "out_dir/sel_args/track_<id>.pkl. On by default; disable with "
            "--no-write-sel-args."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_name, lon_name = args.lat_name, args.lon_name

    logger.info("Opening zarr %s", args.zarr)
    ds = xr.open_zarr(args.zarr)
    if args.time_start is not None or args.time_end is not None:
        ds = ds.sel(time=slice(args.time_start, args.time_end))
    if lat_name is None or lon_name is None:
        lat_name, lon_name = detect_lat_lon_names(ds)
        logger.info("Auto-detected lat/lon coord names: %s, %s", lat_name, lon_name)
    n_time = ds.sizes["time"]

    tracks_path = out_dir / "tracks.dat"
    csv_path = out_dir / "tracks.csv"

    node_list_path = run_bundled_detection(
        args.zarr,
        out_dir,
        n_time,
        args.chunk_size,
        args.workers,
        time_start=args.time_start,
        time_end=args.time_end,
        sample=args.sample,
        psl_var=args.psl_var,
        psfc_var=args.psfc_var,
        t3_var=args.t3_var,
        u_var=args.u_var,
        v_var=args.v_var,
        warm_core=args.warm_core,
        lat_name=lat_name,
        lon_name=lon_name,
        timefilter=args.timefilter,
        detect_exe=args.detect_exe,
        keep_netcdf=args.keep_netcdf,
    )
    run_stitch_nodes(node_list_path, tracks_path, args.stitch_exe)

    df = parse_stitch_output(tracks_path)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote tracks to %s", csv_path)

    if args.write_sel_args and not df.empty:
        write_track_sel_args(df, out_dir, lat_name, lon_name)


if __name__ == "__main__":
    main()
