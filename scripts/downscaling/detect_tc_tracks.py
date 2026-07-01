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
  1. Open the ACE zarr and assemble the four fields TempestExtremes needs
     (sea-level pressure, T3, 10m u/v wind) into per-chunk NetCDF files.
  2. Run ``DetectNodes`` to find candidate TC centers at each timestep.
  3. Run ``StitchNodes`` to link candidates into tracks.
  4. Parse the StitchNodes output into a tidy CSV (one row per track point).

TempestExtremes must be installed and on ``PATH`` (e.g.
``conda install -c conda-forge tempest-extremes``).

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
"""

import argparse
import logging
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
# Track must have wind >= 10 m/s for >= 10 steps and stay within +/- 50 lat.
STITCH_THRESHOLD = "wind,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10"
IN_FMT = "lon,lat,slp,wind"


def build_te_dataset(
    ds: xr.Dataset,
    psl_var: str,
    psfc_var: str,
    t3_var: str,
    u_var: str,
    v_var: str,
    sample: int,
) -> xr.Dataset:
    """Assemble the TempestExtremes input fields from an ACE dataset.

    Selects sea-level pressure (preferring ``psl_var`` if present, else falling
    back to surface pressure), the T3 temperature layer, and 10m winds, and
    renames them to the canonical names used in the DetectNodes command.
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

    for name in (t3_var, u_var, v_var):
        if name not in ds:
            raise KeyError(
                f"{name!r} not found in dataset; available: {list(ds.data_vars)}"
            )

    out = xr.Dataset(
        {
            "PSL": psl,
            "T3": ds[t3_var],
            "U10": ds[u_var],
            "V10": ds[v_var],
        }
    )
    out["PSL"].attrs["units"] = "Pa"
    out["T3"].attrs["units"] = "K"
    out["U10"].attrs["units"] = "m/s"
    out["V10"].attrs["units"] = "m/s"
    return out


def write_netcdf_chunks(ds: xr.Dataset, out_dir: Path, chunk_size: int) -> Path:
    """Write the dataset to per-chunk NetCDF files and a DetectNodes file list.

    TempestExtremes reads NetCDF (not zarr), and splitting the record dimension
    into multiple files keeps memory bounded and lets DetectNodes stream via
    ``--in_data_list``. Returns the path to the written file list.
    """
    n_time = ds.sizes["time"]
    nc_dir = out_dir / "netcdf"
    nc_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for start in range(0, n_time, chunk_size):
        sub = ds.isel(time=slice(start, start + chunk_size))
        path = nc_dir / f"te_input_{start:06d}.nc"
        logger.info("Writing %s (%d timesteps)", path.name, sub.sizes["time"])
        sub.to_netcdf(path)
        paths.append(path)

    list_path = out_dir / "input_files.txt"
    list_path.write_text("\n".join(str(p) for p in paths) + "\n")
    logger.info("Wrote file list %s (%d files)", list_path, len(paths))
    return list_path


def _check_exe(exe: str) -> None:
    """Raise a helpful error if a TempestExtremes binary is not on PATH.

    TempestExtremes is conda-only (not pip-installable); point the user at the
    make target that installs it.
    """
    if shutil.which(exe) is None and not Path(exe).is_file():
        raise FileNotFoundError(
            f"TempestExtremes executable {exe!r} not found on PATH. "
            "TempestExtremes is only distributed on conda-forge (not pip); "
            "install it with:\n"
            "    make -C scripts/downscaling tc_deps\n"
            "or, directly:\n"
            "    conda install -c conda-forge tempest-extremes\n"
            "Alternatively pass an explicit path via --detect-exe/--stitch-exe."
        )


def run_detect_nodes(
    in_list: Path, out_path: Path, exe: str, lat_name: str, lon_name: str
) -> None:
    """Run DetectNodes to find candidate TC centers at each timestep."""
    _check_exe(exe)
    cmd = [
        exe,
        "--in_data_list",
        str(in_list),
        "--out",
        str(out_path),
        "--timefilter",
        "6hr",
        "--searchbymin",
        "PSL",
        "--closedcontourcmd",
        f"{PSL_CONTOUR};{T3_CONTOUR}",
        "--mergedist",
        str(MERGE_DIST),
        "--outputcmd",
        "PSL,min,0;_VECMAG(U10,V10),max,2.0",
        "--latname",
        lat_name,
        "--lonname",
        lon_name,
    ]
    logger.info("Running DetectNodes: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_stitch_nodes(in_path: Path, out_path: Path, exe: str) -> None:
    """Run StitchNodes to link candidate centers into tracks."""
    _check_exe(exe)
    cmd = [
        exe,
        "--in",
        str(in_path),
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
        default=1460,
        help="Timesteps per NetCDF file (1460 = one year at 6-hourly).",
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
    parser.add_argument("--lat-name", default="lat", help="Latitude coord name.")
    parser.add_argument("--lon-name", default="lon", help="Longitude coord name.")
    parser.add_argument("--detect-exe", default="DetectNodes")
    parser.add_argument("--stitch-exe", default="StitchNodes")
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "input_files.txt"

    if not args.skip_convert:
        logger.info("Opening zarr %s", args.zarr)
        ds = xr.open_zarr(args.zarr)
        if args.time_start is not None or args.time_end is not None:
            ds = ds.sel(time=slice(args.time_start, args.time_end))
        te_ds = build_te_dataset(
            ds,
            psl_var=args.psl_var,
            psfc_var=args.psfc_var,
            t3_var=args.t3_var,
            u_var=args.u_var,
            v_var=args.v_var,
            sample=args.sample,
        )
        list_path = write_netcdf_chunks(te_ds, out_dir, args.chunk_size)
    elif not list_path.exists():
        raise FileNotFoundError(
            f"--skip-convert set but {list_path} not found; run conversion first."
        )

    if args.convert_only:
        logger.info("--convert-only set; stopping after NetCDF conversion.")
        return

    nodes_path = out_dir / "candidate_nodes.dat"
    tracks_path = out_dir / "tracks.dat"
    csv_path = out_dir / "tracks.csv"

    run_detect_nodes(
        list_path, nodes_path, args.detect_exe, args.lat_name, args.lon_name
    )
    run_stitch_nodes(nodes_path, tracks_path, args.stitch_exe)

    df = parse_stitch_output(tracks_path)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote tracks to %s", csv_path)


if __name__ == "__main__":
    main()
