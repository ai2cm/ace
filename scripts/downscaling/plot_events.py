#!/usr/bin/env python
"""
Fetch netCDF event files from a beaker dataset and generate map and histogram
plots for each variable (coarse, target, and predicted ensemble samples).

This works with saved event outputs from `fme.downscaling.evaluator` from a
beaker experiment. It downloads the dataset to a temporary directory, finds
``*.nc`` files recursively (including plain ``{event_name}.nc`` from
``EventEvaluator``, not only ``*_YYYYMMDD*.nc``), and optionally merges in
coarse data for map comparison.

Usage:
    python plot_events.py <beaker_dataset_id> [--output-dir <path>]
        [--coarse-data <path>] [--variables VAR1 VAR2 ...]

Requires:
    beaker CLI to be installed and authenticated (https://github.com/allenai/beaker).
"""

import argparse
import math
import re
import tempfile
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.feature import ShapelyFeature
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from utils import fetch_beaker_dataset, find_event_files

warnings.filterwarnings("ignore")

from plot_beaker_histograms import plot_histogram_lines

# default time subselection for coarse data
TIME_SEL = slice(cftime.DatetimeJulian(2023, 1, 1), None)
UNITS = {
    "PRMSL": "hPa",
    "PRATEsfc": "kg/m^2/s",
    "eastward_wind_at_ten_meters": "m/s",
    "northward_wind_at_ten_meters": "m/s",
    "wind_speed": "m/s",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate map plots from beaker dataset event files"
    )
    parser.add_argument(
        "beaker_dataset_id",
        help="The beaker dataset ID to fetch",
    )
    parser.add_argument(
        "--output-dir",
        default="./event_maps",
        help="Output directory for figures (default: ./event_maps)",
    )
    parser.add_argument(
        "--coarse-data",
        default=None,
        help="Path to coarse data (default: None)",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Filter to only these variables (default: all eligible variables)",
    )
    return parser.parse_args()


# Create a STATES feature with no fill
# Read state geometries from shapefile
shpfilename = shapereader.natural_earth(
    resolution="50m", category="cultural", name="admin_1_states_provinces"
)
reader = shapereader.Reader(shpfilename)
states_feature = ShapelyFeature(
    reader.geometries(),
    ccrs.PlateCarree(),
    facecolor="none",
    edgecolor="lightgrey",
    linewidth=0.35,
    linestyle="--",
)


def add_outer_latlon_grid(ax, *, show_left, show_bottom):
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    # Explicitly disable all labels first
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False

    # Enable only outer ones
    if show_left:
        gl.left_labels = True
    if show_bottom:
        gl.bottom_labels = True

    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # square grid cells
    ax.set_aspect("equal", adjustable="box")


def get_coarse_data(path: str | None, time_sel: slice | None = TIME_SEL) -> xr.Dataset:
    if path is not None:
        return xr.open_zarr(path)
    else:
        gcs_root = "gs://vcm-ml-raw-flexible-retention/2025-07-25-X-SHiELD-AMIP-FME/regridded-zarrs/gaussian_grid_180_by_360/control"
        winds = xr.open_zarr(f"{gcs_root}/instantaneous_physics_fields.zarr").sel(
            time=time_sel
        )[["eastward_wind_at_ten_meters", "northward_wind_at_ten_meters"]]
        prate = xr.open_zarr(f"{gcs_root}/fluxes_2d.zarr").sel(time=time_sel)[
            "PRATEsfc"
        ]
        pres = xr.open_zarr(f"{gcs_root}/column_integrated_dynamical_fields.zarr").sel(
            time=time_sel
        )["PRESsfc"]
        # in training, PRESsfc is used as input for outputting PRMSL
        prmsl = pres.rename("PRMSL")
        return xr.merge([winds, prate, pres, prmsl])


def bbox(lat, lon, width=2.0):
    return {
        "lat": slice(lat - width / 2.0, lat + width / 2.0),
        "lon": slice(lon - width / 2.0, lon + width / 2.0),
    }


def upsample_array(x: np.ndarray, upsample_factor: int = 32) -> np.ndarray:
    # upsample coarse data for plotting with fine res
    x = np.repeat(x, upsample_factor, axis=0)  # repeat rows
    x = np.repeat(x, upsample_factor, axis=1)  # repeat columns
    return x


def plot_event(ds, var_name, samples=None, sel=None, n_cols=5, **plot_kwargs):
    if samples is None:
        samples = list(range(ds.sample.size))
    N = 2 + len(samples)
    n_rows = math.ceil(N / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    axes = axes.ravel()  # 1D array, easy to index
    # Use only the first N axes
    for ax in axes[N:]:
        ax.set_visible(False)

    suffixes = ["coarse", "target", "predicted"]
    var_names_with_suffixes = [f"{var_name}_{suffix}" for suffix in suffixes]
    ds_ = ds[var_names_with_suffixes]

    if sel:
        ds_ = ds_.sel(sel)

    if len(samples) == 0:
        samples = [0]

    if var_name == "PRMSL":
        # fill PRMSL_coarse with nans
        ds_["PRMSL_coarse"].values[:] = np.nan
        # For colorbar range, use only target and predicted (coarse is hidden)
        arr = ds_[["PRMSL_target", "PRMSL_predicted"]].to_array()
    else:
        arr = ds_.to_array()

    vals = arr.values
    if hasattr(vals, "compute"):
        vals = vals.compute()
    data_min = np.nanmin(vals)
    data_max = np.nanmax(vals)
    if data_min < -0.2:
        plot_kwargs["cmap"] = "RdBu_r"
    else:
        plot_kwargs["cmap"] = "turbo"
        plot_kwargs["vmin"] = max(0, data_min)
    vmax = data_max
    if "vmax" not in plot_kwargs:
        plot_kwargs["vmax"] = vmax
    # coarse and target
    for i, var in enumerate(var_names_with_suffixes[:2]):
        ax = axes[i]

        da = ds_[var]
        img = da.plot(ax=ax, add_colorbar=False, **plot_kwargs)

        ax.set_title(suffixes[i], fontsize=10)
        ax.add_feature(states_feature)
        ax.add_feature(cfeature.BORDERS, color="lightgrey")
        ax.coastlines(color="lightgrey")

        row = i // n_cols
        col = i % n_cols

        add_outer_latlon_grid(
            ax,
            show_left=(col == 0),
            show_bottom=(row == n_rows - 1),
        )

    for i, s in enumerate(samples):
        ax = axes[2 + i]
        da = ds_[var_names_with_suffixes[-1]].isel(sample=s)

        img = da.plot(ax=ax, add_colorbar=False, **plot_kwargs)
        ax.set_title(f"predicted {s}", fontsize=10)
        ax.add_feature(states_feature)
        ax.coastlines(color="lightgrey")
        ax.add_feature(cfeature.BORDERS, linestyle="-", color="lightgrey")
        row = (i + 2) // n_cols
        col = (i + 2) % n_cols
        add_outer_latlon_grid(
            ax,
            show_left=(col == 0),
            show_bottom=(row == n_rows - 1),
        )
    cbar_ax = fig.add_axes([0.99, 0.25, 0.01, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label(f"{var_name} [{UNITS.get(var_name, '')}]")

    return fig, axes


def detect_variable_pairs(ds: xr.Dataset) -> list[str]:
    """Detect variables that have both _predicted and _target versions."""
    predicted = {
        v[: -len("_predicted")] for v in ds.data_vars if v.endswith("_predicted")
    }
    target = {v[: -len("_target")] for v in ds.data_vars if v.endswith("_target")}
    return sorted(predicted & target)


def infer_event_datetime_for_coarse(path: str | Path) -> cftime.DatetimeJulian:
    """Parse an event date from *path* for coarse-field time selection.

    Tries ``YYYYMMDD`` (and optional ``T{hour}``) anywhere in the string, then
    ISO ``YYYY-MM-DD``. EventEvaluator saves ``{event_name}.nc`` with no date;
    in that case we fall back to ``TIME_SEL.start`` and warn so maps still run.
    """
    s = str(path)
    match = re.search(r"(\d{4})(\d{2})(\d{2})(?:T(\d{2}))?", s)
    if match is not None:
        return cftime.DatetimeJulian(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4) or 12),
        )
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if iso is not None:
        return cftime.DatetimeJulian(
            int(iso.group(1)),
            int(iso.group(2)),
            int(iso.group(3)),
            12,
        )
    warnings.warn(
        f"No date found in path {s!r}; using coarse time {TIME_SEL.start!r} for "
        "merge_coarse. Include YYYYMMDD in the file or parent path, or verify "
        "coarse data covers this time.",
        UserWarning,
        stacklevel=2,
    )
    return TIME_SEL.start


def add_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    variables = detect_variable_pairs(ds)
    if (
        "eastward_wind_at_ten_meters" in variables
        and "northward_wind_at_ten_meters" in variables
    ):
        ds["wind_speed_target"] = np.sqrt(
            ds.eastward_wind_at_ten_meters_target**2
            + ds.northward_wind_at_ten_meters_target**2
        )
        ds["wind_speed_predicted"] = np.sqrt(
            ds.eastward_wind_at_ten_meters_predicted**2
            + ds.northward_wind_at_ten_meters_predicted**2
        )
        ds["wind_speed_coarse"] = np.sqrt(
            ds.eastward_wind_at_ten_meters_coarse**2
            + ds.northward_wind_at_ten_meters_coarse**2
        )
    return ds


def merge_coarse(
    event: xr.Dataset, coarse: xr.Dataset, datetime: cftime.DatetimeJulian
) -> xr.Dataset:
    _coarse = coarse.sel(
        time=datetime,
        grid_yt=slice(event.lat.min(), event.lat.max()),
        grid_xt=slice(event.lon.min(), event.lon.max()),
    )
    for var in detect_variable_pairs(event):
        event[f"{var}_coarse"] = xr.DataArray(
            upsample_array(_coarse[var].values, 32), dims=["lat", "lon"]
        )
    return event


def main():
    args = parse_args()
    beaker_id = args.beaker_dataset_id
    output_dir = Path(args.output_dir)
    coarse = get_coarse_data(args.coarse_data, time_sel=TIME_SEL)

    print(f"Fetching beaker dataset: {beaker_id}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # fetch_beaker_dataset may return a cache dir; temp_dir is empty on cache hit.
        data_dir = fetch_beaker_dataset(beaker_id, temp_dir)
        event_files = find_event_files(data_dir)
        if not event_files:
            print(f"No event files found in dataset {beaker_id} (searched {data_dir})")
            return

        print(f"Found {len(event_files)} event file(s)")

        for event_name, nc_file in event_files.items():
            output_event_dir = output_dir / beaker_id / event_name
            output_event_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing: {nc_file.name} -> {output_event_dir}")

            event = xr.open_dataset(nc_file)
            event = merge_coarse(
                event, coarse, datetime=infer_event_datetime_for_coarse(nc_file)
            )
            event = add_wind_speed(event)
            variables = detect_variable_pairs(event)
            if args.variables is not None:
                variables = [v for v in variables if v in args.variables]

            if not variables:
                print(f"  No variable pairs found in {nc_file.name}")
                continue
            for var in variables:
                fig, axes = plot_event(event, var)
                fig.savefig(
                    output_event_dir / f"{var}_generated_maps.png",
                    transparent=True,
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)
                plot_histogram_lines(
                    event,
                    var,
                    event_name,
                    save_path=output_event_dir / f"{var}_histogram.png",
                )
            event.close()

    print("Done!")


if __name__ == "__main__":
    main()
