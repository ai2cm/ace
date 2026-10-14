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
  1. Open the ACE zarr and assemble the fields TempestExtremes needs
     (sea-level pressure, T3, 10m u/v wind, surface height) into per-chunk
     NetCDF files.
  2. Run ``DetectNodes`` to find candidate TC centers at each timestep.
  3. Run ``StitchNodes`` to link candidates into tracks.
  4. Parse the StitchNodes output into a tidy CSV (one row per track point).
  5. Pickle a per-track xarray ``.sel(..., method="nearest")`` kwargs dict
     for pulling storm-following data out of the original ACE dataset
     (on by default; disable with ``--no-write-sel-args``).

TempestExtremes must be installed and on ``PATH``, or its binaries passed via
``--detect-exe``/``--stitch-exe``. Build it from source with
``make -C scripts/downscaling tc_deps`` (see that target for dependencies).

Usage examples:
    # Basic run on a local zarr, writing intermediates + tracks to out/
    python detect_tc_tracks.py /path/to/ace_output.zarr out/

    # Select a specific ensemble member and time range
    python detect_tc_tracks.py gs://bucket/ace_output.zarr out/ \
        --sample 2 --time-start 2020-01-01 --time-end 2020-12-31

    # Reuse already-written NetCDF intermediates (skip step 1)
    python detect_tc_tracks.py /path/to/ace_output.zarr out/ --skip-convert

    # Only build the NetCDF inputs, don't run TempestExtremes
    python detect_tc_tracks.py /path/to/ace_output.zarr out/ --convert-only

    # Remove the NetCDF intermediates once DetectNodes has consumed them
    python detect_tc_tracks.py /path/to/ace_output.zarr out/ --cleanup

    # Skip writing per-track xarray .sel kwargs
    python detect_tc_tracks.py /path/to/ace_output.zarr out/ --no-write-sel-args
"""

import argparse
import logging
import pickle
import shutil
import subprocess
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
# Track must have wind >= 10 m/s for >= 10 steps, stay within +/- 50 lat, and
# sit over surface height <= 150 m (near sea level) for >= 10 steps.
STITCH_THRESHOLD = "wind,>=,10,10;lat,<=,50.0,10;lat,>=,-50.0,10;zs,<=,150.0,10"

# DetectNodes output columns (after the leading i,j) must match IN_FMT in order;
# see OUTPUT_CMD, which produces slp (PSL), wind (|U10,V10|), and zs (ZS).
IN_FMT = "lon,lat,slp,wind,zs"
OUTPUT_CMD = "PSL,min,0;_VECMAG(U10,V10),max,2.0;ZS,min,0"

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


def _select_surface_height(
    ds: xr.Dataset,
    zs_ds: xr.Dataset | None,
    zs_var: str,
    psl: xr.DataArray,
) -> xr.DataArray:
    """Get the surface-height field for the ZS track filter.

    Prefers ``zs_var`` from the main dataset; if it's absent there, falls back
    to ``zs_ds`` (the optional ``--zs-zarr`` dataset), which must be on the same
    grid (same shape). Surface height is static, so any leading sample/time dims
    are dropped, the field is placed on ``psl``'s spatial grid, and then
    broadcast onto ``psl``'s time axis so DetectNodes sees it at every timestep.
    """
    if zs_var in ds:
        zs = ds[zs_var]
    elif zs_ds is not None and zs_var in zs_ds:
        logger.warning(
            "%s not found in main dataset; taking it from the --zs-zarr "
            "fallback dataset (must be on the same grid)",
            zs_var,
        )
        zs = zs_ds[zs_var]
    elif zs_ds is not None:
        raise KeyError(
            f"{zs_var!r} not found in the main dataset nor in the --zs-zarr "
            f"fallback; main vars: {list(ds.data_vars)}, "
            f"fallback vars: {list(zs_ds.data_vars)}"
        )
    else:
        raise KeyError(
            f"{zs_var!r} not found in the main dataset (pass a dataset "
            f"containing it via --zs-zarr); available: {list(ds.data_vars)}"
        )

    for extra in ("sample", "time"):
        if extra in zs.dims:
            zs = zs.isel({extra: 0}, drop=True)

    # Put ZS on PSL's spatial grid before broadcasting over time. A fallback
    # --zs-zarr may name its grid dims differently (e.g. lat/lon vs the main
    # grid_yt/grid_xt); broadcasting by name would then form a giant outer
    # product instead of overlaying the grids. Remap positionally after
    # checking the shapes match, and drop the source's own spatial coords so
    # xarray uses PSL's (a differently-labeled same-shape grid would otherwise
    # align to all-NaN).
    spatial_dims = tuple(d for d in psl.dims if d != "time")
    expected_shape = tuple(psl.sizes[d] for d in spatial_dims)
    if tuple(zs.dims) != spatial_dims:
        if zs.shape != expected_shape:
            raise ValueError(
                f"surface-height field {dict(zs.sizes)} does not match the main "
                f"grid {dict(zip(spatial_dims, expected_shape))}; regrid the "
                "--zs-zarr source onto the main grid first."
            )
        zs = xr.DataArray(zs.data, dims=spatial_dims)
    if "time" in psl.dims:
        zs = zs.broadcast_like(psl)
    return zs


def build_te_dataset(
    ds: xr.Dataset,
    psl_var: str,
    psfc_var: str,
    t3_var: str,
    u_var: str,
    v_var: str,
    zs_var: str,
    sample: int,
    zs_ds: xr.Dataset | None = None,
) -> xr.Dataset:
    """Assemble the TempestExtremes input fields from an ACE dataset.

    Selects sea-level pressure (preferring ``psl_var`` if present, else falling
    back to surface pressure), the T3 temperature layer, 10m winds, and surface
    height (from ``ds``, or from ``zs_ds`` if absent there), and renames them to
    the canonical names used in the DetectNodes command.
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

    for name in (t3_var, u_var, v_var):
        if name not in ds:
            raise KeyError(
                f"{name!r} not found in dataset; available: {list(ds.data_vars)}"
            )

    zs = _select_surface_height(ds, zs_ds, zs_var, psl)

    out = xr.Dataset(
        {
            "PSL": psl,
            "T3": ds[t3_var],
            "U10": ds[u_var],
            "V10": ds[v_var],
            "ZS": zs,
        }
    )
    out["PSL"].attrs["units"] = "Pa"
    out["T3"].attrs["units"] = "K"
    out["U10"].attrs["units"] = "m/s"
    out["V10"].attrs["units"] = "m/s"
    out["ZS"].attrs["units"] = "m"
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
    "days|hours|minutes|seconds since ..." unit prefixes (TimeObj.cpp); left
    to its own defaults, xarray can encode datetime64[ns] data as
    "nanoseconds since ...", which TempestExtremes rejects with
    'Unknown "time::units" format'. Forcing seconds-since-epoch avoids that.

    The calendar is taken from the (already normalized, see
    _normalize_time_for_tempest) time axis and constrained to one
    TempestExtremes reads correctly, defaulting to proleptic_gregorian.
    """
    calendar = ds["time"].encoding.get("calendar", "proleptic_gregorian")
    if calendar not in _TE_READABLE_CALENDARS:
        calendar = "proleptic_gregorian"
    return {
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": calendar,
        "dtype": "int64",
    }


def write_netcdf_chunks(ds: xr.Dataset, out_dir: Path, chunk_size: int) -> Path:
    """Write the dataset to per-chunk NetCDF files and a DetectNodes file list.

    TempestExtremes reads NetCDF (not zarr), and splitting the record dimension
    into multiple files keeps memory bounded and lets DetectNodes stream via
    ``--in_data_list``. Returns the path to the written file list.
    """
    ds = _normalize_time_for_tempest(ds)
    n_time = ds.sizes["time"]
    nc_dir = out_dir / "netcdf"
    nc_dir.mkdir(parents=True, exist_ok=True)

    time_encoding = _te_time_encoding(ds)
    paths = []
    for start in range(0, n_time, chunk_size):
        sub = ds.isel(time=slice(start, start + chunk_size))
        path = nc_dir / f"te_input_{start:06d}.nc"
        logger.info("Writing %s (%d timesteps)", path.name, sub.sizes["time"])
        sub.to_netcdf(path, encoding={"time": time_encoding})
        paths.append(path)

    list_path = out_dir / "input_files.txt"
    list_path.write_text("\n".join(str(p) for p in paths) + "\n")
    logger.info("Wrote file list %s (%d files)", list_path, len(paths))
    return list_path


def build_node_file_list(in_list_path: Path, out_dir: Path) -> Path:
    """Write a DetectNodes ``--out_file_list`` matching an ``--in_data_list``.

    TempestExtremes only honors a single ``--out`` path when there is exactly
    one input file; with multiple input files (our per-chunk NetCDF split)
    it instead invents its own ``<out>XXXXXX.dat``-style names and ignores
    the path we asked for. Passing an explicit, equal-length output list via
    ``--out_file_list`` makes it write exactly the files we expect.
    """
    nodes_dir = out_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    in_paths = [Path(p) for p in in_list_path.read_text().splitlines() if p.strip()]
    node_paths = [nodes_dir / f"{p.stem}.dat" for p in in_paths]
    node_list_path = out_dir / "candidate_nodes_files.txt"
    node_list_path.write_text("\n".join(str(p) for p in node_paths) + "\n")
    return node_list_path


def cleanup_netcdf_chunks(out_dir: Path) -> None:
    """Remove the per-chunk NetCDF intermediates written by write_netcdf_chunks."""
    nc_dir = out_dir / "netcdf"
    if nc_dir.is_dir():
        logger.info("Removing intermediate NetCDF directory %s", nc_dir)
        shutil.rmtree(nc_dir)


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


def run_detect_nodes(
    in_list: Path,
    out_file_list: Path,
    exe: str,
    lat_name: str,
    lon_name: str,
    log_dir: Path,
) -> None:
    """Run DetectNodes to find candidate TC centers at each timestep."""
    _check_exe(exe)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "--in_data_list",
        str(in_list),
        "--out_file_list",
        str(out_file_list),
        "--logdir",
        str(log_dir),
        "--timefilter",
        "6hr",
        "--searchbymin",
        "PSL",
        "--closedcontourcmd",
        f"{PSL_CONTOUR};{T3_CONTOUR}",
        "--mergedist",
        str(MERGE_DIST),
        "--outputcmd",
        OUTPUT_CMD,
        "--latname",
        lat_name,
        "--lonname",
        lon_name,
    ]
    logger.info("Running DetectNodes: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # DetectNodes catches its own internal exceptions and always exits 0, so a
    # nonzero return code never signals failure here; the only reliable check
    # is whether it actually wrote every file we told it to.
    expected = [Path(p) for p in out_file_list.read_text().splitlines() if p.strip()]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(
            f"DetectNodes did not produce {len(missing)}/{len(expected)} expected "
            f"output file(s); first missing: {missing[0]}. DetectNodes exits 0 even "
            f"when it hits an internal error, so check its per-file logs under "
            f"{log_dir} (e.g. {log_dir}/log000000.txt) for the actual cause."
        )


def run_stitch_nodes(in_list: Path, out_path: Path, exe: str) -> None:
    """Run StitchNodes to link candidate centers into tracks."""
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

    # As with DetectNodes, StitchNodes catches its own exceptions and always
    # exits 0, so check the output actually exists rather than trusting the
    # return code.
    if not out_path.exists():
        raise RuntimeError(
            f"StitchNodes did not produce {out_path}. StitchNodes exits 0 even "
            "when it hits an internal error, so check its stdout above for an "
            "EXCEPTION line."
        )


def parse_stitch_output(path: Path) -> pd.DataFrame:
    """Parse a StitchNodes ASCII track file into a tidy DataFrame.

    The format is a sequence of tracks, each introduced by a ``start`` header
    line, followed by one line per track point:

        start  <npts>  <year> <month> <day> <hour>
            <i> <j> <lon> <lat> <slp> <wind> <zs> <year> <month> <day> <hour>
            ...

    Returns columns: track_id, time, lon, lat, slp (Pa), wind (m/s), zs (m).
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
        # Point line: i, j, lon, lat, slp, wind, zs, year, month, day, hour
        _i, _j, lon, lat, slp, wind, zs, year, month, day, hour = tokens[:11]
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
                "zs": float(zs),
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
        "--zs-var",
        default="HGTsfc",
        help="Surface height (orography) var for the zs track filter.",
    )
    parser.add_argument(
        "--zs-zarr",
        default=None,
        help="Path/URI of a fallback zarr containing --zs-var, consulted only "
        "if that var is absent from the main dataset. Must be on the same "
        "lat/lon grid.",
    )
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
        "--skip-convert",
        action="store_true",
        help="Reuse existing NetCDF inputs in out_dir (skip zarr conversion).",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only build the NetCDF inputs; do not run TempestExtremes.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove the intermediate NetCDF files after DetectNodes finishes.",
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
    list_path = out_dir / "input_files.txt"

    lat_name, lon_name = args.lat_name, args.lon_name

    if not args.skip_convert:
        logger.info("Opening zarr %s", args.zarr)
        ds = xr.open_zarr(args.zarr)
        if args.time_start is not None or args.time_end is not None:
            ds = ds.sel(time=slice(args.time_start, args.time_end))
        if lat_name is None or lon_name is None:
            lat_name, lon_name = detect_lat_lon_names(ds)
            logger.info("Auto-detected lat/lon coord names: %s, %s", lat_name, lon_name)
        zs_ds = None
        if args.zs_zarr is not None:
            logger.info("Opening fallback zs dataset %s", args.zs_zarr)
            zs_ds = xr.open_zarr(args.zs_zarr)
        te_ds = build_te_dataset(
            ds,
            psl_var=args.psl_var,
            psfc_var=args.psfc_var,
            t3_var=args.t3_var,
            u_var=args.u_var,
            v_var=args.v_var,
            zs_var=args.zs_var,
            sample=args.sample,
            zs_ds=zs_ds,
        )
        list_path = write_netcdf_chunks(te_ds, out_dir, args.chunk_size)
    else:
        if not list_path.exists():
            raise FileNotFoundError(
                f"--skip-convert set but {list_path} not found; run conversion first."
            )
        if lat_name is None or lon_name is None:
            first_file = list_path.read_text().splitlines()[0]
            with xr.open_dataset(first_file) as probe_ds:
                lat_name, lon_name = detect_lat_lon_names(probe_ds)
            logger.info("Auto-detected lat/lon coord names: %s, %s", lat_name, lon_name)

    if args.convert_only:
        logger.info("--convert-only set; stopping after NetCDF conversion.")
        return

    tracks_path = out_dir / "tracks.dat"
    csv_path = out_dir / "tracks.csv"

    node_list_path = build_node_file_list(list_path, out_dir)
    run_detect_nodes(
        list_path,
        node_list_path,
        args.detect_exe,
        lat_name,
        lon_name,
        out_dir / "logs",
    )
    if args.cleanup:
        cleanup_netcdf_chunks(out_dir)
    run_stitch_nodes(node_list_path, tracks_path, args.stitch_exe)

    df = parse_stitch_output(tracks_path)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote tracks to %s", csv_path)

    if args.write_sel_args and not df.empty:
        write_track_sel_args(df, out_dir, lat_name, lon_name)


if __name__ == "__main__":
    main()
