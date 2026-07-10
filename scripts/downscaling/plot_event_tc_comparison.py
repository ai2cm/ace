#!/usr/bin/env python
"""
One-off: plot a single ensemble member vs. the target for one downscaling
event, as a 3-row (PRMSL, 10-m wind speed, precip) comparison cropped to the
storm region. By default two columns (predicted, target); pass a second
"teacher" beaker dataset to get a three-column student/teacher/target layout.

Companion to plot_event_member_input.py (single-variable member maps); this
one lays out the three headline fields side-by-side against the target and
derives wind speed as sqrt(u^2 + v^2) from the two wind components.

  - crop is centered on the target PRMSL minimum (the TC center)
  - each row shares one color scale across ALL panels so they are directly
    comparable
  - precip uses the same discrete logarithmic turbo scale (mm/day) as
    plot_event_member_input.py

Usage:
    # student vs target
    python plot_event_tc_comparison.py <student_dataset_id> \
        --event WPacific_hurricane_landfall_china_20230510 [--member 0]

    # student vs teacher vs target
    python plot_event_tc_comparison.py <student_dataset_id> \
        --teacher-dataset-id <teacher_dataset_id> \
        --event WPacific_hurricane_landfall_china_20230510 [--member 0]

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

_EVENT_FILE_RE = re.compile(r"(.+)_(\d{8}).*\.nc$")

WIND_U = "eastward_wind_at_ten_meters"
WIND_V = "northward_wind_at_ten_meters"


def fetch_beaker_dataset(dataset_id: str, target_dir: str, prefix: str | None) -> None:
    """Fetch a beaker dataset (optionally a single file) to target_dir."""
    cmd = ["beaker", "dataset", "fetch", dataset_id, "--output", target_dir]
    if prefix:
        cmd += ["--prefix", prefix]
    subprocess.run(cmd, check=True)


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
        idx = np.argmax(clevs > max_value)
        if idx > 0:
            clevs = clevs[:idx]
        else:
            raise ValueError("max_value is less than the minimum clev value")
    clevs = clevs / cmap_factor
    clevs = clevs[1:]
    cmap = plt.get_cmap("turbo")
    step = max(1, cmap.N // len(clevs))
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist = cmaplist[0::step][: len(clevs)]
    cmap = colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    norm = colors.BoundaryNorm(clevs, cmap.N)
    return cmap, norm


def wind_speed(ds: xr.Dataset, source: str) -> xr.DataArray:
    """sqrt(u^2 + v^2) 10-m wind speed for a given field suffix."""
    u = ds[f"{WIND_U}_{source}"]
    v = ds[f"{WIND_V}_{source}"]
    return np.sqrt(u**2 + v**2)


def crop_around(da: xr.DataArray, lat_c: float, lon_c: float, size_deg: float):
    """Crop a (lat, lon) DataArray to a size_deg x size_deg window around a center."""
    half = size_deg / 2.0
    lat_idx = np.where(np.abs(da.lat.values - lat_c) <= half)[0]
    lon_idx = np.where(np.abs(da.lon.values - lon_c) <= half)[0]
    return da.isel(lat=lat_idx, lon=lon_idx)


def _draw_geo(ax):
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.coastlines(lw=0.6, zorder=5)
    ax.add_feature(cfeature.BORDERS, lw=0.4, zorder=5, edgecolor="0.3")
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, alpha=0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6}


def plot_comparison(
    columns: list[tuple[str, xr.Dataset]],
    member: int,
    crop_degrees: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot a 3-row x N-column comparison.

    Args:
        columns: ordered list of (column_title, dataset) for each prediction
            source. The target is appended automatically as the final column,
            taken from the first dataset (all share the same ground truth).
        member: ensemble sample index used for every prediction column.
    """
    ref = columns[0][1]
    # TC center = minimum of the (uncropped) target PRMSL field.
    prmsl_t_full = ref["PRMSL_target"]
    j, i = np.unravel_index(int(prmsl_t_full.argmin()), prmsl_t_full.shape)
    lat_c = float(prmsl_t_full.lat[j])
    lon_c = float(prmsl_t_full.lon[i])

    def sel(da):
        da = da.isel(sample=member) if "sample" in da.dims else da
        return crop_around(da, lat_c, lon_c, crop_degrees)

    # Row spec: (label, kind, predicted-extractor, target-extractor).
    def _prmsl_p(ds):
        return sel(ds["PRMSL_predicted"])

    def _wind_p(ds):
        return sel(wind_speed(ds, "predicted"))

    def _precip_p(ds):
        return sel(ds["PRATEsfc_predicted"]) * SECONDS_PER_DAY

    row_specs = [
        ("PRMSL [hPa]", "prmsl", _prmsl_p, lambda ds: sel(ds["PRMSL_target"])),
        (
            "10-m wind speed [m s$^{-1}$]",
            "wind",
            _wind_p,
            lambda ds: sel(wind_speed(ds, "target")),
        ),
        (
            "Precip [mm day$^{-1}$]",
            "precip",
            _precip_p,
            lambda ds: sel(ds["PRATEsfc_target"]) * SECONDS_PER_DAY,
        ),
    ]

    # Column fields: one per prediction source, then the shared target last.
    col_titles = [title for title, _ in columns] + ["Target (X-SHiELD 3 km)"]
    ncols = len(col_titles)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(3.6 * ncols + 0.8, 10.5),
        dpi=dpi,
        subplot_kw={"projection": proj},
        constrained_layout=True,
        squeeze=False,
    )

    for r, (label, kind, pred_of, targ_of) in enumerate(row_specs):
        # Fields for this row: each prediction source + the target.
        fields = [pred_of(ds) for _, ds in columns] + [targ_of(ref)]

        vmax = float(max(f.max() for f in fields))
        if kind == "prmsl":
            vmin = float(min(f.min() for f in fields))
            cmap, norm, kw = plt.get_cmap("viridis"), None, dict(vmin=vmin, vmax=vmax)
        elif kind == "wind":
            cmap, norm, kw = plt.get_cmap("turbo"), None, dict(vmin=0.0, vmax=vmax)
        else:  # precip
            cmap, norm = get_precip_cmap_norm(CLEVS, max_value=vmax)
            kw = dict(norm=norm)

        mesh = None
        for c, field in enumerate(fields):
            ax = axes[r, c]
            plotme = (
                field.where(field > PRECIP_FLOOR_MM_DAY) if kind == "precip" else field
            )
            mesh = plotme.plot(
                ax=ax,
                transform=proj,
                add_colorbar=False,
                cmap=cmap,
                alpha=0.9 if kind == "precip" else 1.0,
                zorder=1,
                **kw,
            )
            _draw_geo(ax)
            ax.set_extent(
                [
                    float(field.lon.min()),
                    float(field.lon.max()),
                    float(field.lat.min()),
                    float(field.lat.max()),
                ],
                crs=proj,
            )
            ax.set_title(col_titles[c] if r == 0 else "", fontsize=9)

        # one colorbar per row, spanning all columns
        if kind == "precip":
            assert norm is not None  # set by get_precip_cmap_norm above
            bounds = norm.boundaries
            tick_idx = np.unique(np.linspace(0, len(bounds) - 1, 7).round().astype(int))
            cbar = fig.colorbar(
                mesh,
                ax=axes[r, :].tolist(),
                location="right",
                boundaries=bounds,
                ticks=bounds[tick_idx],
                spacing="uniform",
                shrink=0.9,
                pad=0.02,
            )
        else:
            cbar = fig.colorbar(
                mesh,
                ax=axes[r, :].tolist(),
                location="right",
                shrink=0.9,
                pad=0.02,
            )
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"TC region  (center {lat_c:.1f}°N, {lon_c:.1f}°E, "
        f"{crop_degrees:g}° window, sample {member})",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("beaker_dataset_id", help="Student/first-model beaker dataset ID")
    p.add_argument(
        "--teacher-dataset-id",
        default=None,
        help="Optional second (teacher) beaker dataset ID; adds a middle column",
    )
    p.add_argument(
        "--student-label",
        default="Distilled student",
        help="Column title for the primary dataset (default: 'Distilled student')",
    )
    p.add_argument(
        "--teacher-label",
        default="MoE teacher",
        help="Column title for the teacher dataset (default: 'MoE teacher')",
    )
    p.add_argument(
        "--event", required=True, help="Event name (file <event>_YYYYMMDD*.nc)"
    )
    p.add_argument("--member", type=int, default=0, help="Sample index (default: 0)")
    p.add_argument(
        "--crop-degrees",
        type=float,
        default=6.0,
        help="Size of the centered crop window in degrees (default: 6.0)",
    )
    p.add_argument("--output-dir", default="./tc_comparison", help="Output directory")
    p.add_argument("--dpi", type=int, default=150, help="Figure dpi (default: 150)")
    return p.parse_args()


def _fetch_event(dataset_id: str, event: str, temp_dir: str) -> Path:
    print(f"Fetching beaker dataset: {dataset_id}")
    fetch_beaker_dataset(dataset_id, temp_dir, prefix=event)
    event_files = {n: p for n, p in find_event_files(temp_dir).items() if n == event}
    if not event_files:
        raise FileNotFoundError(f"Event {event!r} not found in dataset {dataset_id}")
    return event_files[event]


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Primary (student / first model).
        student_nc = _fetch_event(
            args.beaker_dataset_id, args.event, str(Path(temp_dir) / "student")
        )
        columns = [
            (
                f"{args.student_label} — sample {args.member}",
                xr.open_dataset(student_nc),
            )
        ]

        if args.teacher_dataset_id:
            teacher_nc = _fetch_event(
                args.teacher_dataset_id, args.event, str(Path(temp_dir) / "teacher")
            )
            columns.append(
                (
                    f"{args.teacher_label} — sample {args.member}",
                    xr.open_dataset(teacher_nc),
                )
            )

        out_path = (
            Path(args.output_dir)
            / args.beaker_dataset_id
            / f"{args.event}_comparison_member{args.member}.png"
        )
        print(f"Plotting {len(columns) + 1}-column comparison -> {out_path}")
        plot_comparison(columns, args.member, args.crop_degrees, out_path, args.dpi)
        for _, ds in columns:
            ds.close()
    print("Done!")


if __name__ == "__main__":
    main()
