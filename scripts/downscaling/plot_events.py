#!/usr/bin/env python
"""
Fetch netCDF event files from a beaker dataset and generate map and histogram
plots for each variable (coarse, target, and predicted ensemble samples).

This works with saved event outputs from `fme.downscaling.evaluator` from a
beaker experiment. It downloads the experiment files to a temporary directory,
parses filenames for *YYYYMMDD*.nc event outputs, and optionally merges in
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
import matplotlib.colors as colors
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
    "PRATEsfc": "mm/day",
    "eastward_wind_at_ten_meters": "m/s",
    "northward_wind_at_ten_meters": "m/s",
    "wind_speed": "m/s",
}
PRECIP_VARS = {"PRATEsfc"}
_KG_TO_MM_PER_DAY = 86400  # (kg/m^2/s) -> (mm/day)

_PRECIP_CLEVS = np.array(
    [
        0,
        0.5,
        1,
        2.5,
        5,
        7.5,
        10,
        15,
        20,
        30,
        40,
        50,
        70,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        700,
        1000,
        1500,
    ]
)


def make_precip_cmap(max_value=None):
    clevs = _PRECIP_CLEVS.copy()
    if max_value is not None:
        idx = np.argmax(clevs > max_value)
        if idx > 0:
            clevs = clevs[:idx]
        else:
            raise ValueError("max_value is less than the minimum clev value")
    clevs = clevs[1:]  # drop the 0 level (used only for bounds)
    base_cmap = plt.get_cmap("turbo")
    step = max(1, base_cmap.N // len(clevs))
    cmaplist = [base_cmap(i) for i in range(0, base_cmap.N, step)][: len(clevs)]
    cmap = colors.LinearSegmentedColormap.from_list(
        "precip_cmap", cmaplist, base_cmap.N
    )
    norm = colors.BoundaryNorm(clevs, cmap.N)
    return cmap, norm


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
    parser.add_argument(
        "--generation-only",
        action="store_true",
        default=False,
        help=(
            "Plot only generated (predicted) samples; "
            "do not require paired target data or coarse data merging."
        ),
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
    if len(samples) == 0:
        samples = [0]

    # Show whichever reference panels are present in the dataset
    reference_panels = [s for s in ("coarse", "target") if f"{var_name}_{s}" in ds]
    n_reference = len(reference_panels)
    N = n_reference + len(samples)
    n_rows = math.ceil(N / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    axes = axes.ravel()  # 1D array, easy to index
    for ax in axes[N:]:
        ax.set_visible(False)

    predicted_name = f"{var_name}_predicted"
    reference_names = [f"{var_name}_{s}" for s in reference_panels]
    vars_to_load = reference_names + [predicted_name]
    ds_ = ds[vars_to_load].copy(deep=True)

    if sel:
        ds_ = ds_.sel(sel)

    if var_name in PRECIP_VARS:
        for v in vars_to_load:
            ds_[v] = ds_[v] * _KG_TO_MM_PER_DAY
            ds_[v].attrs["units"] = "mm/day"
        cmap, norm = make_precip_cmap()
        plot_kwargs["cmap"] = cmap
        plot_kwargs["norm"] = norm
    else:
        if var_name == "PRMSL" and "PRMSL_coarse" in ds_:
            # fill PRMSL_coarse with nans; exclude it from color range
            ds_["PRMSL_coarse"].values[:] = np.nan
            range_vars = [v for v in vars_to_load if v != "PRMSL_coarse"]
            arr = ds_[range_vars].to_array()
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
        if "vmax" not in plot_kwargs:
            plot_kwargs["vmax"] = data_max

    for i, (var, label) in enumerate(zip(reference_names, reference_panels)):
        ax = axes[i]
        da = ds_[var]
        img = da.plot(ax=ax, add_colorbar=False, **plot_kwargs)
        ax.set_title(label, fontsize=10)
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
        ax = axes[n_reference + i]
        da = ds_[predicted_name].isel(sample=s)
        img = da.plot(ax=ax, add_colorbar=False, **plot_kwargs)
        ax.set_title(f"predicted {s}", fontsize=10)
        ax.add_feature(states_feature)
        ax.coastlines(color="lightgrey")
        ax.add_feature(cfeature.BORDERS, linestyle="-", color="lightgrey")
        row = (n_reference + i) // n_cols
        col = (n_reference + i) % n_cols
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


def detect_predicted_variables(ds: xr.Dataset) -> list[str]:
    """Detect variables that have a _predicted version (no target required)."""
    return sorted(
        v[: -len("_predicted")] for v in ds.data_vars if v.endswith("_predicted")
    )


def filename_to_datetime(filename: str) -> cftime.DatetimeJulian:
    match = re.search(r"(\d{4})(\d{2})(\d{2})(?:T(\d{2}))?", filename)
    if match is None:
        raise ValueError(f"Could not parse date from filename: {filename}")
    return cftime.DatetimeJulian(
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4) or 00),
    )


def add_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    u_pred = "eastward_wind_at_ten_meters_predicted"
    v_pred = "northward_wind_at_ten_meters_predicted"
    if u_pred not in ds or v_pred not in ds:
        return ds
    ds["wind_speed_predicted"] = np.sqrt(ds[u_pred] ** 2 + ds[v_pred] ** 2)
    for suffix in ("target", "coarse"):
        u = f"eastward_wind_at_ten_meters_{suffix}"
        v = f"northward_wind_at_ten_meters_{suffix}"
        if u in ds and v in ds:
            ds[f"wind_speed_{suffix}"] = np.sqrt(ds[u] ** 2 + ds[v] ** 2)
    return ds


_LAT_NAMES = ("lat", "latitude", "grid_yt", "y")
_LON_NAMES = ("lon", "longitude", "grid_xt", "x")


def _coerce_datetime(dt: cftime.DatetimeJulian, time_coord: xr.DataArray):
    """Convert dt to match the dtype of time_coord (cftime or numpy datetime64)."""
    import pandas as pd

    if np.issubdtype(time_coord.dtype, np.datetime64):
        return pd.Timestamp(dt.year, dt.month, dt.day, dt.hour)
    return dt


def _detect_coord(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(
        f"Could not find a coordinate matching any of {candidates} in dataset. "
        f"Available coords: {list(ds.coords)}"
    )


def merge_coarse(
    event: xr.Dataset, coarse: xr.Dataset, datetime: cftime.DatetimeJulian
) -> xr.Dataset:
    lat_coord = _detect_coord(coarse, _LAT_NAMES)
    lon_coord = _detect_coord(coarse, _LON_NAMES)
    t = _coerce_datetime(datetime, coarse.time)
    _coarse = coarse.sel(time=t, method="nearest").sel(
        **{
            lat_coord: slice(event.lat.min(), event.lat.max()),
            lon_coord: slice(event.lon.min(), event.lon.max()),
        }
    )
    candidates = detect_variable_pairs(event) or detect_predicted_variables(event)
    for var in candidates:
        if var not in _coarse:
            continue
        event[f"{var}_coarse"] = xr.DataArray(
            upsample_array(_coarse[var].values, 32), dims=["lat", "lon"]
        )
    return event


def main():
    args = parse_args()
    beaker_id = args.beaker_dataset_id
    output_dir = Path(args.output_dir)
    generation_only = args.generation_only

    load_coarse = not generation_only or args.coarse_data is not None
    if load_coarse:
        coarse = get_coarse_data(args.coarse_data, time_sel=TIME_SEL)

    print(f"Fetching beaker dataset: {beaker_id}")

    with tempfile.TemporaryDirectory() as temp_dir:
        fetch_beaker_dataset(beaker_id, temp_dir)

        event_files = find_event_files(temp_dir)
        if not event_files:
            print(f"No event files found in dataset {beaker_id}")
            return

        print(f"Found {len(event_files)} event file(s)")

        for event_name, nc_file in event_files.items():
            output_event_dir = output_dir / beaker_id / event_name
            output_event_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing: {nc_file.name} -> {output_event_dir}")

            event = xr.open_dataset(nc_file)
            if load_coarse:
                event = merge_coarse(
                    event, coarse, datetime=filename_to_datetime(nc_file.name)
                )
            event = add_wind_speed(event)

            if generation_only:
                variables = detect_predicted_variables(event)
                no_vars_msg = f"  No predicted variables found in {nc_file.name}"
            else:
                variables = detect_variable_pairs(event)
                no_vars_msg = f"  No variable pairs found in {nc_file.name}"

            if args.variables is not None:
                variables = [v for v in variables if v in args.variables]

            if not variables:
                print(no_vars_msg)
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
                if not generation_only:
                    plot_histogram_lines(
                        event,
                        var,
                        event_name,
                        save_path=output_event_dir / f"{var}_histogram.png",
                    )
                print(f"  Saved: {output_event_dir / f'{var}_generated_maps.png'}")
            event.close()

    print("Done!")


if __name__ == "__main__":
    main()
