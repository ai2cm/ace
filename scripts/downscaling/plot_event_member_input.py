#!/usr/bin/env python
"""
One-off: plot a single ensemble member from a downscaling event (see
scripts/downscaling/plot_events.py) as a clean "input example" map.

Differences from plot_events.py:
  - closer crop (center 4x4 deg by default, vs the full ~16x16 deg event)
  - land/ocean patches + coastlines underneath the precip field
  - values converted to mm/day (x86400) and masked below 0.5 mm/day
  - discrete, logarithmic turbo colorscale
  - 6x4 in @ 120 dpi, transparent figure background

Usage:
    python plot_event_member_input.py <beaker_dataset_id> \
        [--members 0 3 7] [--variable PRATEsfc] [--source predicted] \
        [--crop-degrees 4.0] [--output-dir ./member_inputs]

Requires the `beaker` CLI to be installed and authenticated.
"""

import argparse
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# kg/m^2/s -> mm/day
SECONDS_PER_DAY = 86400.0
PRECIP_FLOOR_MM_DAY = 0.5

CLEVS = np.array(
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

# Matching for <event_name>_YYYYMMDD*.nc
_EVENT_FILE_RE = re.compile(r"(.+)_(\d{8}).*\.nc$")


def fetch_beaker_dataset(dataset_id: str, target_dir: str) -> None:
    """Fetch a beaker dataset to the specified directory."""
    subprocess.run(
        ["beaker", "dataset", "fetch", dataset_id, "--output", target_dir],
        check=True,
    )


def find_event_files(directory: str) -> dict[str, Path]:
    """Find netCDF files matching the event naming pattern, keyed by event name."""
    event_files = {}
    for p in sorted(Path(directory).glob("*.nc")):
        matched = _EVENT_FILE_RE.match(p.name)
        if matched:
            event_files[matched.group(1)] = p
    return event_files


def get_precip_cmap_norm(clevs, max_value=None, cmap_factor=1.0):
    """Build a discrete, logarithmic turbo colormap + norm from precip levels."""
    clevs = np.asarray(clevs, dtype=float)

    if max_value is not None:
        # find index of first value in clevs greater than max_value
        idx = np.argmax(clevs > max_value)
        if idx > 0:
            clevs = clevs[:idx]
        else:
            raise ValueError("max_value is less than the minimum clev value")

    # Divide levels by factor to adjust for different timescales
    clevs = clevs / cmap_factor

    clevs = clevs[1:]
    cmap = plt.get_cmap("turbo")

    # extract evenly spaced colors from the turbo map
    step = max(1, cmap.N // len(clevs))
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist = cmaplist[0::step][: len(clevs)]

    # create the new map
    cmap = colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

    # define the bins and normalize
    bounds = clevs
    norm = colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def crop_center_degrees(da: xr.DataArray, size_deg: float) -> xr.DataArray:
    """Crop a (lat, lon) DataArray to a centered size_deg x size_deg window.

    Works regardless of whether lat/lon are ascending or descending.
    """
    lat_c = float(da.lat.mean())
    lon_c = float(da.lon.mean())
    half = size_deg / 2.0
    lat_idx = np.where(np.abs(da.lat.values - lat_c) <= half)[0]
    lon_idx = np.where(np.abs(da.lon.values - lon_c) <= half)[0]
    return da.isel(lat=lat_idx, lon=lon_idx)


def plot_member(
    da: xr.DataArray,
    target: xr.DataArray,
    output_path: Path,
    *,
    crop_degrees: float,
    dpi: int,
    lower_threshold: float = PRECIP_FLOOR_MM_DAY,
) -> None:
    # convert to mm/day and crop to the centered window
    da = crop_center_degrees(da, crop_degrees) * SECONDS_PER_DAY
    target = crop_center_degrees(target, crop_degrees) * SECONDS_PER_DAY

    # color range set from the target field, as in the reference plot
    cmap, norm = get_precip_cmap_norm(CLEVS, max_value=float(target.max().values))

    fig = plt.figure(figsize=(6, 4), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())

    plotme = da.where(da > lower_threshold)
    mesh = plotme.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        zorder=1,
        add_colorbar=False,
    )

    # land/ocean patches sit underneath; masked (sub-floor) precip reveals them
    ax.coastlines(lw=0.5, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    gl = ax.gridlines(
        draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_extent(
        [
            float(da.lon.min()),
            float(da.lon.max()),
            float(da.lat.min()),
            float(da.lat.max()),
        ],
        crs=ccrs.PlateCarree(),
    )
    ax.set_aspect("equal", adjustable="box")

    # ~7 tick labels, always including the 0.5 lower bound
    bounds = norm.boundaries
    tick_idx = np.unique(np.linspace(0, len(bounds) - 1, 7).round().astype(int))
    ticks = bounds[tick_idx]

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        boundaries=bounds,
        ticks=ticks,
        spacing="uniform",
        shrink=0.85,
        pad=0.03,
    )
    cbar.set_label("precip [mm/day]")
    cbar.ax.tick_params(labelsize=7)

    # transparent figure patch, but keep the axes background opaque
    fig.patch.set_alpha(0.0)
    ax.patch.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot single ensemble member(s) from beaker event files."
    )
    parser.add_argument("beaker_dataset_id", help="The beaker dataset ID to fetch")
    parser.add_argument(
        "--members",
        nargs="+",
        type=int,
        default=[0],
        help="Sample indices to plot (default: 0)",
    )
    parser.add_argument(
        "--event",
        nargs="*",
        default=None,
        help="Only plot these event name(s) (default: all events in the dataset)",
    )
    parser.add_argument(
        "--variable",
        default="PRATEsfc",
        help="Variable base name to plot (default: PRATEsfc)",
    )
    parser.add_argument(
        "--source",
        choices=["predicted", "target", "coarse"],
        default="predicted",
        help="Which field suffix to plot (default: predicted)",
    )
    parser.add_argument(
        "--crop-degrees",
        type=float,
        default=4.0,
        help="Size of centered crop window in degrees (default: 4.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="./member_inputs",
        help="Output directory for figures (default: ./member_inputs)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure dpi (default: 120)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    var = f"{args.variable}_{args.source}"

    print(f"Fetching beaker dataset: {args.beaker_dataset_id}")
    with tempfile.TemporaryDirectory() as temp_dir:
        fetch_beaker_dataset(args.beaker_dataset_id, temp_dir)
        event_files = find_event_files(temp_dir)
        if not event_files:
            print(f"No event files found in dataset {args.beaker_dataset_id}")
            return
        print(f"Found {len(event_files)} event file(s)")

        if args.event is not None:
            event_files = {
                name: path for name, path in event_files.items() if name in args.event
            }
            missing = set(args.event) - set(event_files)
            if missing:
                print(f"  requested event(s) not found: {sorted(missing)}")
            if not event_files:
                print("No matching events; nothing to plot")
                return

        for event_name, nc_file in event_files.items():
            out_dir = output_dir / args.beaker_dataset_id / event_name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processing: {nc_file.name} -> {out_dir}")

            event = xr.open_dataset(nc_file)
            if var not in event:
                print(f"  {var} not in {nc_file.name}; skipping")
                event.close()
                continue

            da = event[var]
            target_var = f"{args.variable}_target"
            target = event[target_var] if target_var in event else da
            for member in args.members:
                field = da.isel(sample=member) if "sample" in da.dims else da
                out_path = out_dir / (
                    f"{args.variable}_{args.source}_member{member}.png"
                )
                plot_member(
                    field,
                    target,
                    out_path,
                    crop_degrees=args.crop_degrees,
                    dpi=args.dpi,
                )
            event.close()

    print("Done!")


if __name__ == "__main__":
    main()
