"""Geostrophic adjustment demonstration for PrimitiveEquationsStepper.

This script runs the classical geostrophic-adjustment test on a sphere:
a warm temperature dome is placed at 40°N with no initial wind, then
the model is integrated for 8 days.  Expected behaviour:

1. The warm dome raises the 500 hPa geopotential locally.
2. The resulting horizontal pressure gradient accelerates outflow (visible
   at day 1).
3. The Coriolis force deflects the outflow into an anticyclone (clockwise
   in the Northern Hemisphere), which develops over the inertial timescale
   f⁻¹ ≈ 17 h at 40°N.
4. The anticyclone amplitude oscillates (inertial oscillations) around the
   geostrophically balanced state.
5. The upper level (250 hPa) shows a stronger anticyclone than 500 hPa
   because its geopotential (φ₂) depends on ΔT at *both* lower levels
   through the hydrostatic relation.

Model notes
-----------
* ``phi_surface = 0``, so the lowest isobaric level (1000 hPa) has zero
  pressure-gradient force and its wind stays identically zero.  All
  interesting dynamics occur at 500 hPa and above.
* The temperature equation is ``dT/dt = −V·∇T`` (no adiabatic vertical-
  motion term), so each level's temperature changes only through horizontal
  advection by the wind *at that level*.

Usage
-----
Run from the repository root::

    uv run --with torch --with torch-harmonics --with numpy --with matplotlib \\
        python scripts/shallow_water/geostrophic_adjustment.py

Output files (PNG) are written to the current directory.
"""

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from the repository root or from this file's directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fme.core.disco._quadrature import precompute_latitudes, precompute_longitudes
from fme.core.shallow_water import PrimitiveEquationsStepper

# ── Physical constants ────────────────────────────────────────────────────────
A_EARTH = 6.371e6   # m, Earth radius
OMEGA   = 7.292e-5  # rad s⁻¹, Earth rotation rate
R_DRY   = 287.0     # J kg⁻¹ K⁻¹, dry-air gas constant
G       = 9.81      # m s⁻², gravitational acceleration

# ── Experiment parameters ─────────────────────────────────────────────────────
NLAT     = 32
NLON     = 64
PRESSURE = [100_000.0, 50_000.0, 25_000.0]   # Pa — surface, mid, upper trop.
DT       = 900.0                               # s (15 min)
NDAYS    = 8
SAVE_DAYS = [0, 1, 2, 4, 6, 8]

# Warm dome: three levels, decreasing amplitude with height, centred at 40°N 20°E
T_BACKGROUND = [280.0, 255.0, 230.0]   # K — background at each level
DOME_AMPLITUDE = [30.0, 20.0, 10.0]    # K — peak warming at each level
DOME_LAT   = math.radians(40.0)        # rad
DOME_LON   = math.radians(20.0)        # rad
DOME_SIGMA = math.radians(22.0)        # rad — Gaussian half-width (~2450 km)


def build_initial_conditions(lat, lon_n):
    """Return (T_init, uv_init, q_init) tensors on the model grid."""
    LAT, LON = np.meshgrid(lat, lon_n, indexing="ij")
    r2 = (LAT - DOME_LAT) ** 2 + (LON - DOME_LON) ** 2
    dome = np.exp(-r2 / DOME_SIGMA**2)

    T_init = np.zeros((1, 3, NLAT, NLON))
    for k in range(3):
        T_init[0, k] = T_BACKGROUND[k] + DOME_AMPLITUDE[k] * dome

    uv_init = np.zeros((1, 3, NLAT, NLON, 2))
    q_init  = np.zeros_like(T_init)
    return (
        torch.tensor(T_init,  dtype=torch.float32),
        torch.tensor(uv_init, dtype=torch.float32),
        torch.tensor(q_init,  dtype=torch.float32),
    )


def run_simulation(T_t, uv_t, q_t, stepper, dt, save_days):
    """Integrate the stepper and return {day: (uv, T)} snapshots."""
    nsteps      = int(NDAYS * 86400 / dt)
    save_steps  = {int(d * 86400 / dt): d for d in save_days}
    snapshots   = {}

    with torch.no_grad():
        for step in range(nsteps + 1):
            if step in save_steps:
                d = save_steps[step]
                snapshots[d] = (uv_t.clone(), T_t.clone())
                v1_max = uv_t[0, 1].norm(dim=-1).max().item()
                print(f"  day {d:2d}  |V_500hPa|_max = {v1_max:.2f} m/s")
            if step == nsteps:
                break
            uv_t, T_t, q_t = stepper.step(uv_t, T_t, q_t, dt)
            if not torch.isfinite(T_t).all():
                print(f"  NaN/Inf at step {step} (day {step*dt/86400:.1f}); stopping.")
                break

    return snapshots


def plot_geopotential_and_wind(snapshots, lat, lon_n, save_days, outfile):
    """Plot 500 hPa geopotential anomaly (colours) + wind (arrows) at each saved day."""
    lat_d = np.degrees(lat)
    lon_d = np.degrees(lon_n)
    days  = sorted(snapshots.keys())
    ncols = 3
    nrows = math.ceil(len(days) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).ravel()

    for i, d in enumerate(days):
        ax = axes[i]
        # 500 hPa geopotential anomaly = R * ΔT₀ * ln(p₀/p₁)
        T0 = snapshots[d][1][0, 0].numpy()
        dphi1 = R_DRY * (T0 - T_BACKGROUND[0]) * math.log(PRESSURE[0] / PRESSURE[1])

        uv1 = snapshots[d][0][0, 1].numpy()   # (nlat, nlon, 2)
        spd = np.hypot(uv1[..., 0], uv1[..., 1])

        vmax = abs(dphi1).max()
        cf = ax.contourf(lon_d, lat_d, dphi1,
                         levels=np.linspace(-vmax, vmax, 21),
                         cmap="RdBu_r", extend="both")
        ax.contour(lon_d, lat_d, spd,
                   levels=5, colors="k", linewidths=0.7, alpha=0.7)
        sk = 4
        ax.quiver(lon_d[::sk], lat_d[::sk],
                  uv1[::sk, ::sk, 0], uv1[::sk, ::sk, 1],
                  scale=80, width=0.005, color="k", alpha=0.8)
        ax.set_title(f"Day {d}", fontsize=11)
        ax.set_xlim([-60, 120])
        ax.set_ylim([0, 80])
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.colorbar(cf, ax=ax, label="Δφ₁ (m² s⁻²)", shrink=0.9)

    for ax in axes[len(days):]:
        ax.set_visible(False)

    fig.suptitle(
        "500 hPa geopotential anomaly Δφ₁ (colours) and wind V₁ (arrows + speed contours)\n"
        "Geostrophic adjustment: warm dome at 40°N 20°E, f⁻¹ ≈ 17 h at 40°N",
        fontsize=12,
    )
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_vertical_structure(snapshots, lat, lon_n, day, outfile):
    """Plot wind speed and vectors at all three pressure levels for one day."""
    lat_d = np.degrees(lat)
    lon_d = np.degrees(lon_n)
    level_names = ["1000 hPa (surface)", "500 hPa (mid-trop.)", "250 hPa (upper trop.)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for k, ax in enumerate(axes):
        uv_k = snapshots[day][0][0, k].numpy()
        spd  = np.hypot(uv_k[..., 0], uv_k[..., 1])
        cf   = ax.contourf(lon_d, lat_d, spd, levels=15, cmap="YlOrRd")
        sk = 4
        ax.quiver(lon_d[::sk], lat_d[::sk],
                  uv_k[::sk, ::sk, 0], uv_k[::sk, ::sk, 1],
                  scale=60, width=0.005, color="0.3", alpha=0.7)
        ax.set_xlim([-60, 120])
        ax.set_ylim([0, 80])
        ax.set_title(level_names[k], fontsize=11)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.colorbar(cf, ax=ax, label="|V| (m s⁻¹)")

    fig.suptitle(
        f"Day {day}: wind speed & vectors at three pressure levels\n"
        "Anticyclone intensifies with height (hydrostatic coupling: φ_k depends on T at all lower levels)",
        fontsize=12,
    )
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f"Saved {outfile}")


def main():
    # ── Grid ─────────────────────────────────────────────────────────────────
    colat_t, _ = precompute_latitudes(NLAT, grid="equiangular")
    lon_t       = precompute_longitudes(NLON)
    lat  = 0.5 * math.pi - colat_t.numpy()   # geographic lat, north → south
    lon_n = lon_t.numpy()                     # [0, 2π)

    # ── Initial conditions ────────────────────────────────────────────────────
    T_t, uv_t, q_t = build_initial_conditions(lat, lon_n)

    # Theoretical geostrophic wind for reference
    sigma_m = DOME_SIGMA * A_EARTH          # m
    ln_ratio = math.log(PRESSURE[0] / PRESSURE[1])
    dphi_max = R_DRY * DOME_AMPLITUDE[0] * ln_ratio * math.sqrt(2) / DOME_SIGMA * math.exp(-0.5)
    f_40 = 2 * OMEGA * math.sin(DOME_LAT)
    u_geo_expected = dphi_max / (f_40 * A_EARTH)
    print(f"\nExpected geostrophic wind at dome edge: {u_geo_expected:.1f} m/s")
    print(f"(may be underestimated at 32×64 resolution with ~{sigma_m/1e3:.0f} km dome)\n")

    # ── Stepper ───────────────────────────────────────────────────────────────
    stepper = PrimitiveEquationsStepper(
        shape=(NLAT, NLON),
        n_levels=3,
        pressure_levels=PRESSURE,
        g=G,
        R=R_DRY,
        omega=OMEGA,
        phi_surface=0.0,
        kernel_shape=5,
    )
    stepper.eval()

    # ── Run ───────────────────────────────────────────────────────────────────
    print("Integrating …")
    snapshots = run_simulation(T_t, uv_t, q_t, stepper, DT, SAVE_DAYS)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_geopotential_and_wind(
        snapshots, lat, lon_n, SAVE_DAYS,
        "geostrophic_adjustment_phi_wind.png",
    )
    plot_vertical_structure(
        snapshots, lat, lon_n, day=4,
        outfile="geostrophic_adjustment_vertical_day4.png",
    )


if __name__ == "__main__":
    main()
