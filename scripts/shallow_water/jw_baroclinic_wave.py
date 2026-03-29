"""Jablonowski & Williamson (2006) baroclinic-wave test for PrimitiveEquationsStepper.

Reference
---------
Jablonowski, C. & Williamson, D. L. (2006). A baroclinic instability test case
for atmospheric model dynamical cores. Quarterly Journal of the Royal
Meteorological Society, 132(621), 2943–2975.

The test initialises a balanced mid-latitude jet in hydrostatic and geostrophic
equilibrium, then superimposes a small temperature perturbation to seed
baroclinic instability.  The expected behaviour is:

1. Days 0–6: nearly steady state (the perturbation is small).
2. Days 7–12: exponential growth of a travelling baroclinic wave with
   zonal wave-number 6.
3. The wave is visible in surface temperature, 500 hPa vorticity, and the
   zonal-mean zonal-wind cross-section.

Model notes
-----------
* The current isobaric model uses a fixed pressure grid.  The JW test is
  formulated in hybrid-σ coordinates; we map it to pure pressure levels by
  taking η_k = p_k / p_0 (η_s ≡ 1 at the surface, η_t ≡ 0.2 at the top).
* The balanced geopotential at each level is constructed analytically from
  the JW wind profile and the geostrophic-balance equation, then the
  temperature is recovered hydrostatically.
* Level 0 has zero pressure-gradient force (φ_surface = 0), so its wind
  stays zero.  The baroclinic wave develops at the upper levels.

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
from fme.core.shallow_water import PrimitiveEquationsStepper

# ── Physical constants ────────────────────────────────────────────────────────
A_EARTH = 6.371e6    # m, Earth radius
OMEGA   = 7.292e-5   # rad s⁻¹, Earth rotation rate
R_DRY   = 287.0      # J kg⁻¹ K⁻¹, dry-air gas constant
C_P     = 1004.0     # J kg⁻¹ K⁻¹, dry-air specific heat
G       = 9.81       # m s⁻², gravitational acceleration
KAPPA   = R_DRY / C_P  # ≈ 0.2857

# ── JW2006 parameters ─────────────────────────────────────────────────────────
U0    = 35.0    # m/s, jet maximum wind speed (JW eq. 2)
ETA_0 = 0.252   # η at the jet core (controls jet height; JW eq. 2)
T0    = 288.0   # K, reference temperature (JW eq. 4)
DELTA_T = 4.8e5 # K, temperature lapse parameter (JW eq. 4)
ETA_T = 0.2     # η of the tropopause (JW eq. 4)

# ── Perturbation parameters (JW eq. 10) ──────────────────────────────────────
UP    = 1.0             # K temperature perturbation amplitude
PHIP  = math.radians(40.0)  # latitude of perturbation centre
LAMP  = math.radians(20.0)  # longitude of perturbation centre
XP    = 0.1             # angular half-width (rad)

# ── Experiment parameters ─────────────────────────────────────────────────────
NLAT     = 64
NLON     = 128
P0       = 1.0e5        # Pa, reference surface pressure
# Five levels spanning the troposphere (η = 1.0, 0.75, 0.5, 0.35, 0.2)
ETA_LEVELS = [1.0, 0.75, 0.50, 0.35, 0.20]
PRESSURE   = [eta * P0 for eta in ETA_LEVELS]   # Pa
DT         = 1800.0     # s (30 min)
NDAYS      = 12
SAVE_DAYS  = [0, 4, 6, 8, 10, 12]


# ── JW analytical profiles ─────────────────────────────────────────────────────

def _eta_v(eta):
    """Vertical coordinate of the jet core: η_v = (η − η₀)·π/2 (JW eq. 2)."""
    return (eta - ETA_0) * math.pi / 2.0


def jw_u(phi, eta):
    """Zonal wind: u = u₀·cos^(3/2)(η_v)·sin²(2φ)  (JW eq. 2)."""
    ev = _eta_v(eta)
    return U0 * math.cos(ev) ** 1.5 * np.sin(2.0 * phi) ** 2


def jw_T_ref(eta):
    """Horizontally-uniform reference temperature (JW eq. 4).

    T_ref(η) = T₀·η^(R/c_p)  +  ΔT·(η_T/η − 1)^5    for η ≤ η_T
             = T₀·η^(R/c_p)                             for η > η_T
    """
    t = T0 * eta**KAPPA
    if eta <= ETA_T:
        t += DELTA_T * (ETA_T / eta - 1.0) ** 5
    return t


def jw_phi_jet(phi, eta):
    """Geopotential anomaly from the balanced jet (JW eq. 5 integrated).

    φ_jet(φ, η) = −a·∫₀^φ (u·(2Ω·sin + tan/a·u)) dφ'
                           ← geostrophic/gradient-wind balance

    JW give the antiderivative analytically (their eq. 5):
      −a·u₀·cos^(3/2)(η_v)·[(−2·sin^6(φ)·(cos^2(φ) + 1/3) + 10/63)
                              ·2·Ω·u₀·cos^(3/2)(η_v)
                           + (1.6·cos^3(φ)·(sin^2(φ) + 2/3) − π/4)·a·f₀·u₀^... ]

    We implement the JW closed form directly (their equations 3–5):
      Φ_jet = u₀·cos^(3/2)(η_v)·{
                (−2·sin^6φ·(cos^2φ + 1/3) + 10/63)·u₀·cos^(3/2)(η_v)
                + (8/5·cos^3φ·(sin^2φ + 2/3) − π/4)·a·2·Ω }
    """
    ev = _eta_v(eta)
    fc = math.cos(ev) ** 1.5
    sinp = np.sin(phi)
    cosp = np.cos(phi)
    term1 = (-2.0 * sinp**6 * (cosp**2 + 1.0 / 3.0) + 10.0 / 63.0) * U0 * fc
    term2 = (8.0 / 5.0 * cosp**3 * (sinp**2 + 2.0 / 3.0) - math.pi / 4.0) * A_EARTH * 2.0 * OMEGA
    return U0 * fc * (term1 + term2)


def build_initial_conditions(lat, lon_n):
    """Build balanced (T, uv, q) initial conditions for the JW test.

    Temperature at each level is recovered from the geopotential via the
    hydrostatic relation:  T_k = (φ_{k+1} − φ_k) / (R·ln(p_k / p_{k+1}))

    A small surface-level temperature perturbation seeds the instability.
    """
    K  = len(ETA_LEVELS)
    H  = len(lat)
    W  = len(lon_n)
    LAT, LON = np.meshgrid(lat, lon_n, indexing="ij")  # (H, W)

    # ── Geopotential at each pressure level ──────────────────────────────────
    phi_ref = np.array([jw_T_ref(eta) * R_DRY / KAPPA for eta in ETA_LEVELS])  # just T×R/κ
    phi = np.zeros((K, H, W))
    for k, eta in enumerate(ETA_LEVELS):
        phi[k] = jw_phi_jet(lat[:, None], eta)  # (H, W) via broadcasting

    # ── Temperature from hydrostatic inversion ────────────────────────────────
    # φ_k = φ_{k-1} + R·T_{k-1}·ln(p_{k-1}/p_k)  →  T_{k-1} = (φ_k − φ_{k-1}) / (R·ln(p_{k-1}/p_k))
    # Level 0 (surface): use T_ref(η_0) + φ_jet correction referenced to the
    # upper layer.  We derive T at each layer from the difference of φ values:
    p = np.array(PRESSURE)
    T = np.zeros((K, H, W))
    # Derive T_k from φ_{k+1} − φ_k = R·T_k·ln(p_k/p_{k+1})
    for k in range(K - 1):
        ln_ratio = math.log(p[k] / p[k + 1])
        T[k] = (phi[k + 1] - phi[k]) / (R_DRY * ln_ratio)
    # Top level: use reference T_ref; phi[K-1] is already consistent
    T[K - 1] = jw_T_ref(ETA_LEVELS[K - 1]) * np.ones((H, W))

    # Add horizontal reference temperature (uniform) to each level
    for k, eta in enumerate(ETA_LEVELS):
        T[k] += jw_T_ref(eta)

    # ── Zonal wind ────────────────────────────────────────────────────────────
    uv = np.zeros((K, H, W, 2))
    for k, eta in enumerate(ETA_LEVELS):
        uv[k, :, :, 0] = jw_u(lat[:, None], eta)  # zonal (u) component

    # ── Temperature perturbation (surface level only) ─────────────────────────
    # JW eq. 10: δT = U_p · exp(−((λ − λ_p)² + (φ − φ_p)²) / X_p²)
    r2 = (LAT - PHIP) ** 2 + (LON - LAMP) ** 2
    T[0] += UP * np.exp(-r2 / XP**2)

    # ── Pack into tensors ─────────────────────────────────────────────────────
    T_t  = torch.tensor(T[None].astype(np.float32))    # (1, K, H, W)
    uv_t = torch.tensor(uv[None].astype(np.float32))   # (1, K, H, W, 2)
    q_t  = torch.zeros(1, K, H, W)
    return T_t, uv_t, q_t


def run_simulation(T_t, uv_t, q_t, stepper, save_days):
    nsteps     = int(NDAYS * 86400 / DT)
    save_steps = {int(d * 86400 / DT): d for d in save_days}
    snapshots  = {}

    with torch.no_grad():
        for step in range(nsteps + 1):
            if step in save_steps:
                d = save_steps[step]
                snapshots[d] = (uv_t.clone(), T_t.clone())
                v_max = uv_t[0].norm(dim=-1).max().item()
                print(f"  day {d:2d}  |V|_max = {v_max:.1f} m/s")
            if step == nsteps:
                break
            uv_t, T_t, q_t = stepper.step(uv_t, T_t, q_t, DT)
            if not torch.isfinite(T_t).all():
                print(f"  NaN/Inf at step {step} (day {step*DT/86400:.1f}); stopping.")
                break

    return snapshots


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_surface_T_anomaly(snapshots, lat, lon_n, save_days, outfile):
    """Surface (level-0) temperature anomaly relative to zonal mean.

    Subtracting the zonal mean at each latitude removes the background
    meridional temperature gradient and isolates the wave signal.
    """
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
        T_k  = snapshots[d][1][0, 0].numpy()       # (H, W)
        anom = T_k - T_k.mean(axis=1, keepdims=True)  # remove zonal mean
        vmax = max(abs(anom).max(), 0.5)
        cf = ax.contourf(lon_d, lat_d, anom,
                         levels=np.linspace(-vmax, vmax, 21),
                         cmap="RdBu_r", extend="both")
        ax.set_title(f"Day {d}", fontsize=11)
        ax.set_xlim([-180, 180]); ax.set_ylim([0, 90])
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        plt.colorbar(cf, ax=ax, label="T′ (K)", shrink=0.9)

    for ax in axes[len(days):]:
        ax.set_visible(False)

    fig.suptitle(
        "JW baroclinic wave — surface temperature departure from zonal mean (K)\n"
        "(wave signal at level 0; perturbation at 40°N 20°E seeds instability)",
        fontsize=12,
    )
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_vorticity_500(snapshots, lat, lon_n, stepper, save_days, outfile):
    """500 hPa (mid-level) relative vorticity."""
    lat_d = np.degrees(lat)
    lon_d = np.degrees(lon_n)
    days  = sorted(snapshots.keys())
    # Find level closest to 500 hPa
    p_arr = stepper.pressure_levels.numpy()
    mid_k = int(np.argmin(np.abs(p_arr - 5e4)))

    ncols = 3
    nrows = math.ceil(len(days) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).ravel()

    for i, d in enumerate(days):
        ax = axes[i]
        uv_k = snapshots[d][0][0, mid_k]  # (H, W, 2)
        # Finite-difference vorticity: ∂v/∂x − ∂u/∂y (crude; longitude/latitude)
        u = uv_k[..., 0].numpy()
        v = uv_k[..., 1].numpy()
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
        ax.set_xlim([-180, 180]); ax.set_ylim([0, 90])
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        plt.colorbar(cf, ax=ax, label="ζ (×10⁻⁴ s⁻¹)", shrink=0.9)

    for ax in axes[len(days):]:
        ax.set_visible(False)

    p_hpa = p_arr[mid_k] / 100.0
    fig.suptitle(
        f"JW baroclinic wave — relative vorticity at {p_hpa:.0f} hPa (×10⁻⁴ s⁻¹)\n"
        "Wave-number-6 baroclinic wave expected to emerge ~day 8",
        fontsize=12,
    )
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_zonal_mean_u(snapshots, lat, stepper, day, outfile):
    """Zonal-mean zonal wind as a latitude–pressure cross-section."""
    lat_d = np.degrees(lat)
    p_arr = stepper.pressure_levels.numpy() / 100.0  # hPa

    uv = snapshots[day][0][0].numpy()   # (K, H, W, 2)
    # u_zm: average u over longitudes → shape (K, H)
    u_zm = uv[..., 0].mean(axis=-1)     # (K, H)

    # contourf(x, y, z): x=lat (H,), y=pressure (K,), z must be (K, H)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    vmax = min(abs(u_zm).max(), 80.0)   # cap for readability
    cf = ax.contourf(lat_d, p_arr, u_zm,
                     levels=np.linspace(-vmax, vmax, 17),
                     cmap="RdBu_r", extend="both")
    ax.contour(lat_d, p_arr, u_zm, levels=[0], colors="k", linewidths=1.5)
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_ylim([p_arr.max(), p_arr.min()])   # surface at bottom
    ax.set_xlim([-90, 90])
    ax.set_title(f"Day {day}: zonal-mean zonal wind [u] (m s⁻¹)", fontsize=12)
    plt.colorbar(cf, ax=ax, label="[u] (m s⁻¹)")
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f"Saved {outfile}")


def main():
    # ── Grid ──────────────────────────────────────────────────────────────────
    colat_t, _ = precompute_latitudes(NLAT, grid="equiangular")
    lon_t       = precompute_longitudes(NLON)
    lat   = (0.5 * math.pi - colat_t).numpy()   # geographic lat, north → south
    lon_n = lon_t.numpy()                         # [0, 2π)

    # ── Initial conditions ─────────────────────────────────────────────────────
    print("Building initial conditions …")
    T_t, uv_t, q_t = build_initial_conditions(lat, lon_n)
    print(f"  T range: {T_t.min().item():.1f} – {T_t.max().item():.1f} K")
    print(f"  |V| max: {uv_t.norm(dim=-1).max().item():.1f} m/s")

    # ── Stepper ────────────────────────────────────────────────────────────────
    stepper = PrimitiveEquationsStepper(
        shape=(NLAT, NLON),
        n_levels=len(PRESSURE),
        pressure_levels=PRESSURE,
        g=G,
        R=R_DRY,
        omega=OMEGA,
        phi_surface=0.0,
        kernel_shape=5,
    )
    stepper.eval()

    # ── Run ────────────────────────────────────────────────────────────────────
    print("Integrating …")
    snapshots = run_simulation(T_t, uv_t, q_t, stepper, SAVE_DAYS)

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_surface_T_anomaly(
        snapshots, lat, lon_n, SAVE_DAYS,
        "jw_surface_T_anomaly.png",
    )
    plot_vorticity_500(
        snapshots, lat, lon_n, stepper, SAVE_DAYS,
        "jw_vorticity_500hPa.png",
    )
    plot_zonal_mean_u(
        snapshots, lat, stepper, day=8,
        outfile="jw_zonal_mean_u_day8.png",
    )


if __name__ == "__main__":
    main()
