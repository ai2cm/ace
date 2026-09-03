"""
Plot spatial fields on timesteps immediately before and after suspected
ERA5 analysis artifact discontinuities, to visualize the jump.

Saves plots to scripts/era5_investigation/plots/
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ZARR_PATH = "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Discontinuity cases to investigate
# Each: (description, variable, date_before, date_after)
CASES = [
    # 1986 specific_total_water_0 jump (NOT a known stream boundary)
    (
        "1986 STW0 jump (not a stream boundary)",
        "specific_total_water_0",
        "1985-12-31T18:00",
        "1986-01-01T00:00",
    ),
    # 1986 April 1 stream boundary (where the actual STW0 drop occurs)
    (
        "1986 Apr stream boundary - specific_total_water_0",
        "specific_total_water_0",
        "1986-03-31T18:00",
        "1986-04-01T00:00",
    ),
    # 1993 air_temperature_0 drop (Stream 2→3 boundary is Sep 1992)
    # Check both the stream boundary and the volcanic timing
    (
        "1992 Sep stream boundary - air_temperature_0",
        "air_temperature_0",
        "1992-08-31T18:00",
        "1992-09-01T00:00",
    ),
    (
        "1993 Jan - air_temperature_0 (post-Pinatubo cooling)",
        "air_temperature_0",
        "1992-12-31T18:00",
        "1993-01-01T00:00",
    ),
    # 2000 specific_total_water_0 jump (IS a known stream boundary)
    (
        "2000 STW0 jump (known stream boundary)",
        "specific_total_water_0",
        "1999-12-31T18:00",
        "2000-01-01T00:00",
    ),
    # For comparison: a non-discontinuity year transition
    (
        "2005 STW0 (control - no known discontinuity)",
        "specific_total_water_0",
        "2004-12-31T18:00",
        "2005-01-01T00:00",
    ),
    # Also look at Sep 1992 for STW0 (stream boundary)
    (
        "1992 Sep stream boundary - specific_total_water_0",
        "specific_total_water_0",
        "1992-08-31T18:00",
        "1992-09-01T00:00",
    ),
]

# Also: plot broader time context (monthly means) around each discontinuity
MONTHLY_CASES = [
    (
        "specific_total_water_0",
        "1985-06",
        "1986-06",
        "1986 STW0 jump",
    ),
    (
        "air_temperature_0",
        "1992-03",
        "1993-03",
        "1992-93 Pinatubo + stream boundary",
    ),
    (
        "specific_total_water_0",
        "1999-06",
        "2000-06",
        "2000 STW0 jump (stream boundary)",
    ),
]


def plot_before_after(ds, desc, var, t_before, t_after):
    """Plot the field at two consecutive timesteps and their difference."""
    print(f"  Loading {var} at {t_before} and {t_after}...")
    t0 = time.time()

    field_before = ds[var].sel(time=t_before, method="nearest").load()
    field_after = ds[var].sel(time=t_after, method="nearest").load()
    actual_t_before = str(field_before.time.values)[:19]
    actual_t_after = str(field_after.time.values)[:19]

    diff = field_after - field_before
    elapsed = time.time() - t0
    print(f"    Loaded in {elapsed:.1f}s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Common colorbar range for before/after
    vmin = min(float(field_before.min()), float(field_after.min()))
    vmax = max(float(field_before.max()), float(field_after.max()))

    im0 = axes[0].pcolormesh(
        field_before.longitude,
        field_before.latitude,
        field_before.values,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    axes[0].set_title(f"Before: {actual_t_before}")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].pcolormesh(
        field_after.longitude,
        field_after.latitude,
        field_after.values,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    axes[1].set_title(f"After: {actual_t_after}")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    # Difference with symmetric colorbar
    diff_abs_max = float(np.abs(diff).max())
    if diff_abs_max == 0:
        diff_abs_max = 1
    im2 = axes[2].pcolormesh(
        diff.longitude,
        diff.latitude,
        diff.values,
        vmin=-diff_abs_max,
        vmax=diff_abs_max,
        cmap="RdBu_r",
    )
    axes[2].set_title(f"Difference (after - before)")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(f"{desc}\n{var}", fontsize=13)
    plt.tight_layout()

    safe_desc = (
        desc.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    )
    fname = os.path.join(PLOT_DIR, f"discontinuity_{safe_desc}.png")
    fig.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fname}")


def plot_monthly_context(ds, var, start_month, end_month, desc):
    """Plot monthly global-mean timeseries around a discontinuity."""
    print(f"  Loading monthly context for {var} from {start_month} to {end_month}...")
    t0 = time.time()

    chunk = ds[var].sel(time=slice(start_month, end_month))
    # Compute global mean timeseries
    gm = chunk.mean(dim=["latitude", "longitude"]).load()
    elapsed = time.time() - t0
    print(f"    Loaded in {elapsed:.1f}s")

    fig, ax = plt.subplots(figsize=(12, 4))
    times = gm.time.values
    values = gm.values

    ax.plot(times, values, "b-", linewidth=0.5, alpha=0.5, label="6-hourly")

    # Compute and overlay monthly means
    monthly = gm.resample(time="MS").mean()
    ax.plot(
        monthly.time.values,
        monthly.values,
        "r-o",
        linewidth=2,
        markersize=4,
        label="Monthly mean",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(var)
    ax.set_title(f"{desc}: {var} global mean")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_desc = (
        desc.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    )
    fname = os.path.join(PLOT_DIR, f"monthly_{safe_desc}_{var}.png")
    fig.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fname}")


def main():
    print("Opening zarr dataset...")
    ds = xr.open_zarr(ZARR_PATH)

    print("\n=== Before/After Spatial Plots ===")
    for desc, var, t_before, t_after in CASES:
        print(f"\nCase: {desc}")
        plot_before_after(ds, desc, var, t_before, t_after)

    print("\n=== Monthly Context Timeseries ===")
    for var, start, end, desc in MONTHLY_CASES:
        print(f"\nCase: {desc}")
        plot_monthly_context(ds, var, start, end, desc)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
