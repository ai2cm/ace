"""Jablonowski & Williamson (2006) baroclinic-wave test for SigmaCoordinateStepper.

Reference
---------
Jablonowski, C. & Williamson, D. L. (2006). A baroclinic instability test case
for atmospheric model dynamical cores. Quarterly Journal of the Royal
Meteorological Society, 132(621), 2943–2975.

Initialises a balanced mid-latitude jet in hydrostatic and geostrophic equilibrium,
then superimposes a small temperature perturbation to seed baroclinic instability.

Key differences from the earlier isobaric version
--------------------------------------------------
* Uses ``SigmaCoordinateStepper`` so p_s(λ,φ,t) is prognostic.  The surface
  level (σ=1) now has a non-zero pressure-gradient force, allowing the full
  column to participate in the baroclinic instability.
* Temperature is initialised directly from JW eq. 6 (not from hydrostatic
  inversion of the geopotential), giving a statically stable profile.
* p_s is initialised to a uniform P₀ = 10⁵ Pa everywhere.

Usage
-----
Run from the repository root::

    uv run --with torch --with torch-harmonics --with numpy --with matplotlib \\
        python scripts/shallow_water/jw_baroclinic_wave.py

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fme.core.disco._quadrature import precompute_latitudes, precompute_longitudes
from fme.core.shallow_water import SigmaCoordinateStepper

# ── Physical constants ────────────────────────────────────────────────────────
A_EARTH = 6.371e6    # m
OMEGA   = 7.292e-5   # rad s⁻¹
R_DRY   = 287.0      # J kg⁻¹ K⁻¹
C_P     = 1004.0     # J kg⁻¹ K⁻¹
KAPPA   = R_DRY / C_P

# ── JW2006 parameters ─────────────────────────────────────────────────────────
U0    = 35.0          # m/s, jet maximum speed
ETA_0 = 0.252         # σ at the jet core
T0    = 288.0         # K, surface reference temperature
P0    = 1.0e5         # Pa, reference surface pressure

# ── Perturbation (temperature, surface level) ─────────────────────────────────
UP   = 1.0                     # K amplitude
PHIP = math.radians(40.0)      # lat of perturbation centre
LAMP = math.radians(20.0)      # lon of perturbation centre
XP   = 0.1                     # angular half-width (rad)

# ── Experiment parameters ─────────────────────────────────────────────────────
NLAT      = 64
NLON      = 128
# Five sigma levels: surface (σ=1) to tropopause (σ=0.2)
SIGMA_LEVELS = [1.0, 0.75, 0.50, 0.35, 0.20]
DT        = 1800.0    # s (30 min)
NDAYS     = 12
SAVE_DAYS = [0, 4, 6, 8, 10, 12]
# Light ∇² diffusion: keeps simulation stable over 30 days
DIFFUSION_COEFF = 5e5   # m²/s


# ── JW analytical profiles ────────────────────────────────────────────────────

def _eta_v(eta: float) -> float:
    """Vertical coordinate of the jet core: η_v = (η − η₀)·π/2."""
    return (eta - ETA_0) * math.pi / 2.0


def jw_u(phi: np.ndarray, eta: float) -> np.ndarray:
    """Zonal wind u = U₀·cos^(3/2)(η_v)·sin²(2φ)  (JW eq. 2)."""
    ev = _eta_v(eta)
    return U0 * math.cos(ev) ** 1.5 * np.sin(2.0 * phi) ** 2


def jw_T(phi: np.ndarray, eta: float) -> np.ndarray:
    """Temperature from JW eq. 6 (direct formula, statically stable).

    T(φ, η) = T₀·η^κ
              + (3π/4)·(η·U₀/R)·sin(η_v)·cos^(1/2)(η_v) × bracket(φ, η_v)

    bracket = (-2sin^6φ·(cos²φ+1/3) + 10/63)·2U₀·cos^(3/2)(η_v)
              + (8/5·cos³φ·(sin²φ+2/3) − π/4)·a·2Ω
    """
    ev = _eta_v(eta)
    prefactor = (3.0 * math.pi / 4.0) * (eta * U0 / R_DRY) * (
        math.sin(ev) * math.cos(ev) ** 0.5
    )
    sinp = np.sin(phi)
    cosp = np.cos(phi)
    bracket = (
        (-2.0 * sinp**6 * (cosp**2 + 1.0 / 3.0) + 10.0 / 63.0)
        * 2.0 * U0 * math.cos(ev) ** 1.5
        + (8.0 / 5.0 * cosp**3 * (sinp**2 + 2.0 / 3.0) - math.pi / 4.0)
        * A_EARTH * 2.0 * OMEGA
    )
    return T0 * eta**KAPPA + prefactor * bracket


def build_initial_conditions(
    lat: np.ndarray, lon_n: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (T, uv, q, p_s) tensors for the JW balanced initial state.

    Temperature is set from JW eq. 6 directly (statically stable everywhere).
    A 1 K Gaussian temperature perturbation at the surface level seeds the
    baroclinic instability.
    Surface pressure is uniform: p_s = P₀ = 10⁵ Pa.
    """
    K  = len(SIGMA_LEVELS)
    H  = len(lat)
    W  = len(lon_n)
    LAT, LON = np.meshgrid(lat, lon_n, indexing="ij")   # (H, W)

    T  = np.zeros((K, H, W), dtype=np.float32)
    uv = np.zeros((K, H, W, 2), dtype=np.float32)
    for k, eta in enumerate(SIGMA_LEVELS):
        T[k]       = jw_T(lat[:, None], eta)
        uv[k, :, :, 0] = jw_u(lat[:, None], eta)

    # Temperature perturbation at surface level (σ=1) to seed instability
    r2 = (LAT - PHIP) ** 2 + (LON - LAMP) ** 2
    T[0] += UP * np.exp(-r2 / XP**2)

    T_t  = torch.tensor(T[None])                            # (1, K, H, W)
    uv_t = torch.tensor(uv[None])                           # (1, K, H, W, 2)
    q_t  = torch.zeros(1, K, H, W)
    ps_t = torch.full((1, H, W), P0)                        # (1, H, W)
    return T_t, uv_t, q_t, ps_t


def run_simulation(
    T_t: torch.Tensor,
    uv_t: torch.Tensor,
    q_t: torch.Tensor,
    ps_t: torch.Tensor,
    stepper: SigmaCoordinateStepper,
    save_days: list[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Integrate and return {day: (uv, T, p_s)} snapshots."""
    nsteps     = int(NDAYS * 86400 / DT)
    save_steps = {int(d * 86400 / DT): d for d in save_days}
    snapshots: dict = {}

    with torch.no_grad():
        for step in range(nsteps + 1):
            if step in save_steps:
                d = save_steps[step]
                snapshots[d] = (uv_t.clone(), T_t.clone(), ps_t.clone())
                v_max = uv_t[0].norm(dim=-1).max().item()
                ps_range = (ps_t.min().item(), ps_t.max().item())
                print(f"  day {d:2d}  |V|_max={v_max:.1f} m/s  "
                      f"p_s=[{ps_range[0]:.0f},{ps_range[1]:.0f}] Pa")
            if step == nsteps:
                break
            uv_t, T_t, q_t, ps_t = stepper.step(uv_t, T_t, q_t, ps_t, DT)
            if not torch.isfinite(T_t).all():
                print(f"  NaN/Inf at step {step} (day {step*DT/86400:.1f})")
                break

    return snapshots


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_surface_T_anomaly(
    snapshots: dict, lat: np.ndarray, lon_n: np.ndarray, outfile: str
) -> None:
    """Surface (σ=1) temperature departure from its zonal mean."""
    lat_d = np.degrees(lat)
    lon_d = np.degrees(lon_n)
    days  = sorted(snapshots.keys())
    ncols = 3
    nrows = math.ceil(len(days) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                             constrained_layout=True)
    axes = np.array(axes).ravel()

    for i, d in enumerate(days):
        ax = axes[i]
        T_k  = snapshots[d][1][0, 0].numpy()
        anom = T_k - T_k.mean(axis=1, keepdims=True)
        vmax = max(abs(anom).max(), 0.5)
        cf = ax.contourf(lon_d, lat_d, anom,
                         levels=np.linspace(-vmax, vmax, 21),
                         cmap="RdBu_r", extend="both")
        ax.set_title(f"Day {d}", fontsize=11)
        ax.set_xlim([-180, 180]); ax.set_ylim([-90, 90])
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        plt.colorbar(cf, ax=ax, label="T′ (K)", shrink=0.9)

    for ax in axes[len(days):]:
        ax.set_visible(False)
    fig.suptitle(
        "JW baroclinic wave (σ coordinates) — surface T departure from zonal mean (K)",
        fontsize=12,
    )
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_surface_pressure(
    snapshots: dict, lat: np.ndarray, lon_n: np.ndarray, outfile: str
) -> None:
    """Surface pressure anomaly p_s - P₀ (Pa)."""
    lat_d = np.degrees(lat)
    lon_d = np.degrees(lon_n)
    days  = sorted(snapshots.keys())
    ncols = 3
    nrows = math.ceil(len(days) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                             constrained_layout=True)
    axes = np.array(axes).ravel()

    for i, d in enumerate(days):
        ax = axes[i]
        ps  = snapshots[d][2][0].numpy()     # (H, W)
        anom = ps - P0
        vmax = max(abs(anom).max(), 10.0)
        cf = ax.contourf(lon_d, lat_d, anom,
                         levels=np.linspace(-vmax, vmax, 21),
                         cmap="RdBu_r", extend="both")
        ax.set_title(f"Day {d}", fontsize=11)
        ax.set_xlim([-180, 180]); ax.set_ylim([-90, 90])
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        plt.colorbar(cf, ax=ax, label="p_s − P₀ (Pa)", shrink=0.9)

    for ax in axes[len(days):]:
        ax.set_visible(False)
    fig.suptitle(
        "JW baroclinic wave (σ coordinates) — surface pressure anomaly p_s − P₀ (Pa)\n"
        "Baroclinic wave creates high/low pressure couplets",
        fontsize=12,
    )
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_vorticity_mid(
    snapshots: dict, lat: np.ndarray, lon_n: np.ndarray, outfile: str
) -> None:
    """Relative vorticity at the mid-troposphere level (σ≈0.5)."""
    lat_d = np.degrees(lat)
    lon_d = np.degrees(lon_n)
    days  = sorted(snapshots.keys())
    mid_k = SIGMA_LEVELS.index(0.50)
    ncols = 3
    nrows = math.ceil(len(days) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                             constrained_layout=True)
    axes = np.array(axes).ravel()

    for i, d in enumerate(days):
        ax = axes[i]
        uv_k = snapshots[d][0][0, mid_k].numpy()   # (H, W, 2)
        u, v = uv_k[..., 0], uv_k[..., 1]
        dlat = lat[1] - lat[0] if len(lat) > 1 else 1.0
        dlon = lon_n[1] - lon_n[0] if len(lon_n) > 1 else 1.0
        dvdx = np.gradient(v, dlon, axis=1) / (A_EARTH * np.cos(lat[:, None]))
        dudy = np.gradient(u, dlat, axis=0) / A_EARTH
        vort = dvdx - dudy
        vmax = max(abs(vort).max(), 1e-5)
        cf = ax.contourf(lon_d, lat_d, vort * 1e4,
                         levels=np.linspace(-vmax * 1e4, vmax * 1e4, 21),
                         cmap="RdBu_r", extend="both")
        ax.set_title(f"Day {d}", fontsize=11)
        ax.set_xlim([-180, 180]); ax.set_ylim([-90, 90])
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        plt.colorbar(cf, ax=ax, label="ζ (×10⁻⁴ s⁻¹)", shrink=0.9)

    for ax in axes[len(days):]:
        ax.set_visible(False)
    fig.suptitle(
        f"JW baroclinic wave (σ coordinates) — vorticity at σ=0.5 (×10⁻⁴ s⁻¹)\n"
        "Zonal bands break into wave structure as instability grows",
        fontsize=12,
    )
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f"Saved {outfile}")


def main() -> None:
    # ── Grid ──────────────────────────────────────────────────────────────────
    colat_t, _ = precompute_latitudes(NLAT, grid="equiangular")
    lon_t       = precompute_longitudes(NLON)
    lat   = (0.5 * math.pi - colat_t).numpy()
    lon_n = lon_t.numpy()

    # ── Initial conditions ─────────────────────────────────────────────────────
    print("Building initial conditions …")
    T_t, uv_t, q_t, ps_t = build_initial_conditions(lat, lon_n)
    print(f"  T  range: {T_t.min():.1f} – {T_t.max():.1f} K")
    print(f"  |V| max: {uv_t.norm(dim=-1).max():.1f} m/s")
    print(f"  p_s:     {ps_t.min():.0f} – {ps_t.max():.0f} Pa")

    # ── Stepper ────────────────────────────────────────────────────────────────
    stepper = SigmaCoordinateStepper(
        shape=(NLAT, NLON),
        n_levels=len(SIGMA_LEVELS),
        sigma_levels=SIGMA_LEVELS,
        R=R_DRY,
        omega=OMEGA,
        phi_surface=0.0,
        kernel_shape=5,
        diffusion_coeff=DIFFUSION_COEFF,
        diffusion_order=1,
    )
    stepper.eval()

    # ── Run ────────────────────────────────────────────────────────────────────
    print("Integrating …")
    snapshots = run_simulation(T_t, uv_t, q_t, ps_t, stepper, SAVE_DAYS)

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_surface_T_anomaly(snapshots, lat, lon_n, "jw_surface_T_anomaly.png")
    plot_surface_pressure(snapshots, lat, lon_n, "jw_surface_pressure.png")
    plot_vorticity_mid(snapshots, lat, lon_n, "jw_vorticity_sigma05.png")


if __name__ == "__main__":
    main()
