"""Radial-based structural verification metrics for tropical cyclones.

This module implements the framework described in ``eval.md``: 2D storm-centered
fields (10 m wind speed, sea-level pressure, precipitation rate) are transformed
into 1D azimuthal-radial profiles, and structural / dynamical errors are scored
from those profiles. Working in radius space isolates structural errors (size,
symmetry, gradient-wind balance) and sidesteps the double-penalty of grid-to-grid
RMSE for a slightly displaced storm.

Standardized inputs
-------------------
Two conventions are used throughout:

* ``gen``    -- generated / forecast field(s): ``[time,] ensemble, lat, lon``
* ``target`` -- verification field:            ``[time,] lat, lon``

The array functions here operate on the trailing ``(lat, lon)`` axes and
broadcast over any leading batch dims (e.g. ``ensemble``). **Time is looped
externally** because the storm center moves each step and every function takes a
single center. Layer-2 structural metrics take a single 1D azimuthal-mean profile
(one storm / one member) and return scalars; ensemble scoring (Layer 4) then runs
CRPS / RMSE over the per-member results or over the radial profiles directly.

Design layers
-------------
1. Coordinate transform & binning: great-circle distance, radial bins, azimuthal
   mean / variance / count.
2. Structural metrics on 1D radial profiles: radius of max wind, wind radii,
   pressure-gradient force, radial mismatch, wind-pressure imbalance, precip
   morphology, profile MAE / variance.
3. Scorecard: assemble the ``eval.md`` metric table for one storm (sim vs obs).
4. Ensemble scoring: CRPS (fair estimator, matching
   ``fme/downscaling/metrics_and_maths.py``) and ensemble RMSE.

Units: pass sea-level pressure in **Pa** (the source ``PRMSL`` is millibar --
multiply by 100). Wind in m/s, precip in whatever rate unit you like (metrics are
unit-agnostic except the 10 m/s valid-bin threshold and the kt wind radii).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_KM = 6371.0
KT_TO_MS = 0.514444
# 34 / 50 / 64 kt operational wind-radii thresholds, in m/s.
WIND_RADII_THRESHOLDS_MS = (34 * KT_TO_MS, 50 * KT_TO_MS, 64 * KT_TO_MS)
# Bins whose azimuthal-mean wind is below this are dropped (eval.md sec. 1.3).
MIN_WIND_MS = 10.0
# Reference near-surface air density for pressure-gradient-force diagnostics.
RHO_AIR = 1.15


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of ``y`` over ``x`` (np.trapz was removed in NumPy 2)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


# --------------------------------------------------------------------------- #
# Layer 1: coordinate transform & radial binning
# --------------------------------------------------------------------------- #
def _as_lat_lon_2d(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D (lat, lon) grids from either 1D axes or 2D grids."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if lats.ndim == 1 and lons.ndim == 1:
        return np.meshgrid(lats, lons, indexing="ij")
    return np.broadcast_arrays(lats, lons)


def great_circle_distance_km(
    center_lat: float,
    center_lon: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km from a center to every grid point.

    Uses the haversine formula so it is correct across the 0/360 longitude seam
    and at high latitude. ``lats``/``lons`` may be 1D axes (meshgridded here) or
    2D grids; the result has the 2D ``(nlat, nlon)`` shape.
    """
    lat2d, lon2d = _as_lat_lon_2d(lats, lons)
    lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
    lat2, lon2 = np.radians(lat2d), np.radians(lon2d)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def radial_bin_edges(dr_km: float = 25.0, r_max_km: float = 600.0) -> np.ndarray:
    """Uniform radial-ring edges from 0 to ``r_max_km`` in steps of ``dr_km``.

    Default ``r_max_km`` (500--600 km) keeps every ring well-sampled inside a 16
    degree (~1600 km) patch; at 25 km data the natural ``dr_km`` is 25 km.
    """
    return np.arange(0.0, r_max_km + dr_km, dr_km)


def bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """Midpoint radius of each ring."""
    bin_edges = np.asarray(bin_edges, dtype=float)
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


def azimuthal_stats(
    field: np.ndarray,
    distance_km: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthal mean, variance and sample count per radial ring.

    Args:
        field: values on the grid, shape ``(..., nlat, nlon)`` (leading dims are
            batched, e.g. an ensemble axis).
        distance_km: great-circle distance of each grid point from the center,
            shape ``(nlat, nlon)``.
        bin_edges: radial ring edges, shape ``(nbins + 1,)``.

    Returns:
        ``(mean, var, count)`` where ``mean`` and ``var`` have shape
        ``(..., nbins)`` and ``count`` has shape ``(nbins,)``. Empty rings are
        ``NaN`` in ``mean``/``var`` and 0 in ``count``. Variance is the
        population (ddof=0) azimuthal variance.
    """
    field = np.asarray(field, dtype=float)
    distance_km = np.asarray(distance_km, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)
    nbins = len(bin_edges) - 1
    lead_shape = field.shape[:-2]
    npix = distance_km.size

    field_flat = field.reshape(lead_shape + (npix,))
    dist_flat = distance_km.reshape(npix)
    # digitize returns i where edges[i-1] <= x < edges[i]; subtract 1 -> ring idx.
    ring = np.digitize(dist_flat, bin_edges) - 1

    mean = np.full(lead_shape + (nbins,), np.nan)
    var = np.full(lead_shape + (nbins,), np.nan)
    count = np.zeros(nbins, dtype=int)
    for b in range(nbins):
        sel = ring == b
        count[b] = int(sel.sum())
        if count[b] == 0:
            continue
        vals = field_flat[..., sel]
        mean[..., b] = vals.mean(axis=-1)
        var[..., b] = vals.var(axis=-1)
    return mean, var, count


def refine_center_min_slp(
    pressure: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    center: tuple[float, float],
    search_radius_km: float = 150.0,
) -> tuple[float, float]:
    """Refine a storm center to the minimum-pressure grid point nearby.

    ``eval.md`` allows identifying the eye via minimum pressure. Given an initial
    ``center = (lat, lon)`` (e.g. a track position), return the ``(lat, lon)`` of
    the lowest-pressure grid point within ``search_radius_km``.
    """
    lat2d, lon2d = _as_lat_lon_2d(lats, lons)
    dist = great_circle_distance_km(center[0], center[1], lats, lons)
    masked = np.where(dist <= search_radius_km, np.asarray(pressure, float), np.inf)
    j = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return float(lat2d[j]), float(lon2d[j])


@dataclass
class RadialProfile:
    """A single field's azimuthal profile: bin centers, mean, variance, count."""

    r_km: np.ndarray  # (nbins,)
    mean: np.ndarray  # (..., nbins)
    var: np.ndarray  # (..., nbins)
    count: np.ndarray  # (nbins,)


def compute_radial_profile(
    field: np.ndarray,
    center_lat: float,
    center_lon: float,
    lats: np.ndarray,
    lons: np.ndarray,
    bin_edges: np.ndarray,
) -> RadialProfile:
    """Convenience wrapper: distance + ``azimuthal_stats`` -> ``RadialProfile``.

    ``field`` may carry leading batch dims (e.g. ``ensemble, lat, lon``); the
    resulting ``mean``/``var`` keep those leading dims.
    """
    dist = great_circle_distance_km(center_lat, center_lon, lats, lons)
    mean, var, count = azimuthal_stats(field, dist, bin_edges)
    return RadialProfile(r_km=bin_centers(bin_edges), mean=mean, var=var, count=count)


# --------------------------------------------------------------------------- #
# Layer 2: structural metrics on 1D azimuthal-mean profiles
# --------------------------------------------------------------------------- #
def radius_of_extremum(
    profile: np.ndarray, r_km: np.ndarray, kind: str = "max"
) -> tuple[np.ndarray, np.ndarray]:
    """Radius and value of the profile extremum along the last axis.

    ``kind`` is ``"max"`` or ``"min"``. NaN rings are ignored. Works on a 1D
    profile (returns scalars) or batched profiles ``(..., nbins)``.
    """
    profile = np.asarray(profile, dtype=float)
    r_km = np.asarray(r_km, dtype=float)
    argfn = np.nanargmax if kind == "max" else np.nanargmin
    idx = argfn(profile, axis=-1)
    r = r_km[idx]
    val = np.take_along_axis(profile, np.expand_dims(idx, -1), axis=-1)
    return r, np.squeeze(val, axis=-1)


def radius_of_max_wind(
    wind_mean: np.ndarray, r_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Radius of maximum winds ``R_max`` and the peak wind ``V_max``."""
    return radius_of_extremum(wind_mean, r_km, kind="max")


def _outermost_threshold_radius(
    profile: np.ndarray, r_km: np.ndarray, threshold: float
) -> float:
    """Outermost radius where a 1D profile crosses down through ``threshold``.

    Returns the linearly-interpolated radius between the last ring at/above the
    threshold and the next ring below it. If the profile never reaches the
    threshold, returns NaN; if it is still at/above the threshold at the domain
    edge, returns the edge radius.
    """
    profile = np.asarray(profile, dtype=float)
    r_km = np.asarray(r_km, dtype=float)
    above = np.where(np.isfinite(profile) & (profile >= threshold))[0]
    if above.size == 0:
        return np.nan
    last = int(above.max())
    if last == len(profile) - 1:
        return float(r_km[last])
    r0, r1 = r_km[last], r_km[last + 1]
    v0, v1 = profile[last], profile[last + 1]
    if v0 == v1:
        return float(r0)
    frac = (v0 - threshold) / (v0 - v1)
    return float(r0 + frac * (r1 - r0))


def wind_radii(
    wind_mean: np.ndarray,
    r_km: np.ndarray,
    thresholds_ms: tuple[float, ...] = WIND_RADII_THRESHOLDS_MS,
) -> dict[float, float]:
    """Critical wind radii: outermost radius where azimuthal wind drops below
    each threshold (default 34/50/64 kt in m/s). Returns ``{threshold: radius}``.
    """
    return {
        thr: _outermost_threshold_radius(wind_mean, r_km, thr) for thr in thresholds_ms
    }


def profile_mae(
    sim: np.ndarray, obs: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    """Mean absolute error between two radial profiles along the last axis.

    Generic building block for the integrated wind-profile MAE and the azimuthal
    variance-profile MAE. ``valid_mask`` (over rings) and NaN rings are excluded.
    """
    diff = np.abs(np.asarray(sim, float) - np.asarray(obs, float))
    if valid_mask is not None:
        diff = np.where(np.asarray(valid_mask, bool), diff, np.nan)
    return np.nanmean(diff, axis=-1)


def profile_total_variance(
    mean_profile: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    """Variance across the 1D mean radial profile itself (gradient sharpness)."""
    x = np.asarray(mean_profile, float)
    if valid_mask is not None:
        x = np.where(np.asarray(valid_mask, bool), x, np.nan)
    return np.nanvar(x, axis=-1)


def pressure_gradient_force(
    pressure_pa: np.ndarray, r_km: np.ndarray, rho: float = RHO_AIR
) -> np.ndarray:
    """Pressure-gradient force profile ``(1/rho) dP/dr`` (r in metres).

    ``pressure_pa`` must be in Pa. Uses centered finite differences.
    """
    p = np.asarray(pressure_pa, float)
    r_m = np.asarray(r_km, float) * 1000.0
    return np.gradient(p, r_m, axis=-1) / rho


def radial_mismatch(
    pgf_profile: np.ndarray, wind_mean: np.ndarray, r_km: np.ndarray
) -> float:
    """Per-storm ``R_PGFmax - R_Vmax`` (radius of max PGF minus radius of max wind).

    The scorecard's ``dR_mismatch`` is ``|sim - obs|`` of this quantity.
    """
    r_pgf, _ = radius_of_extremum(pgf_profile, r_km, kind="max")
    r_v, _ = radius_of_max_wind(wind_mean, r_km)
    return float(r_pgf - r_v)


def wind_pressure_imbalance(
    wind_mean: np.ndarray,
    pressure_pa: np.ndarray,
    r_km: np.ndarray,
    rho: float = RHO_AIR,
    r_outer_km: float | None = None,
) -> float:
    """Cyclostrophic residual ``dP/rho - integral_0^R V^2/r dr`` for one storm.

    ``eval.md`` writes ``dP - integral V^2/r dr``; we divide ``dP`` by ``rho`` so
    both terms are specific energy (J/kg) and the residual is physically the
    cyclostrophic imbalance. ``dP = P(R_outer) - P(center)``. The integral is a
    trapezoid over the rings out to ``r_outer_km`` (default: last finite ring).
    """
    v = np.asarray(wind_mean, float)
    p = np.asarray(pressure_pa, float)
    r_m = np.asarray(r_km, float) * 1000.0
    finite = np.isfinite(v) & np.isfinite(p)
    if r_outer_km is not None:
        finite &= np.asarray(r_km, float) <= r_outer_km
    vs, ps, rs = v[finite], p[finite], r_m[finite]
    delta_p = ps[-1] - ps[0]
    integral = _trapezoid(vs**2 / rs, rs)
    return float(delta_p / rho - integral)


def efolding_radius(
    precip_mean: np.ndarray,
    r_km: np.ndarray,
    r_m: float | None = None,
    t_m: float | None = None,
) -> float:
    """Moisture-envelope e-folding radius.

    Outward from the peak-rain radius ``R_m`` (with peak rate ``T_m``), the radius
    where the azimuthal-mean rain rate first drops to ``T_m / e`` (~37%). ``R_m``
    and ``T_m`` are found from ``precip_mean`` if not supplied.
    """
    p = np.asarray(precip_mean, float)
    r = np.asarray(r_km, float)
    if r_m is None or t_m is None:
        r_m, t_m = radius_of_extremum(p, r, kind="max")
    target = t_m / np.e
    i_m = int(np.argmin(np.abs(r - r_m)))
    for i in range(i_m, len(p) - 1):
        if p[i] >= target > p[i + 1]:
            r0, r1, v0, v1 = r[i], r[i + 1], p[i], p[i + 1]
            frac = (v0 - target) / (v0 - v1)
            return float(r0 + frac * (r1 - r0))
    return np.nan


# --------------------------------------------------------------------------- #
# Layer 3: scorecard (eval.md metric table for one storm, sim vs obs)
# --------------------------------------------------------------------------- #
@dataclass
class StormProfiles:
    """Azimuthal-mean (and variance) radial profiles for one storm.

    ``pressure_mean`` must be in Pa. ``wind_var`` (azimuthal variance of the wind
    field per ring) is used for the symmetry metrics; ``precip_mean`` is optional.
    """

    r_km: np.ndarray
    wind_mean: np.ndarray
    pressure_mean: np.ndarray
    wind_var: np.ndarray | None = None
    precip_mean: np.ndarray | None = None


def valid_wind_mask(
    obs_wind_mean: np.ndarray, min_wind_ms: float = MIN_WIND_MS
) -> np.ndarray:
    """Rings kept for profile comparisons: where the *target* azimuthal wind is at
    least ``min_wind_ms`` (eval.md sec. 1.3), defining the meaningful storm extent.
    """
    w = np.asarray(obs_wind_mean, float)
    return np.isfinite(w) & (w >= min_wind_ms)


def scorecard(
    sim: StormProfiles, obs: StormProfiles, min_wind_ms: float = MIN_WIND_MS
) -> dict[str, float]:
    """Compute the ``eval.md`` scorecard metrics for one storm (sim vs obs).

    The valid-ring mask is derived from the *observed* wind (>= ``min_wind_ms``)
    and applied to both profiles so the comparison covers the same storm extent.
    """
    r = obs.r_km
    mask = valid_wind_mask(obs.wind_mean, min_wind_ms)

    r_max_sim, _ = radius_of_max_wind(sim.wind_mean, r)
    r_max_obs, _ = radius_of_max_wind(obs.wind_mean, r)

    sim_wr = wind_radii(sim.wind_mean, r)
    obs_wr = wind_radii(obs.wind_mean, r)

    out: dict[str, float] = {
        "dR_max_km": abs(float(r_max_sim) - float(r_max_obs)),
        "MAE_v_radial": float(profile_mae(sim.wind_mean, obs.wind_mean, mask)),
    }
    for thr in WIND_RADII_THRESHOLDS_MS:
        out[f"dR_{round(thr / KT_TO_MS)}kt_km"] = float(sim_wr[thr] - obs_wr[thr])

    # Mass-momentum / dynamic consistency.
    pgf_sim = pressure_gradient_force(sim.pressure_mean, r)
    pgf_obs = pressure_gradient_force(obs.pressure_mean, r)
    out["dR_mismatch_km"] = abs(
        radial_mismatch(pgf_sim, sim.wind_mean, r)
        - radial_mismatch(pgf_obs, obs.wind_mean, r)
    )
    out["dImbalance"] = abs(
        wind_pressure_imbalance(sim.wind_mean, sim.pressure_mean, r)
        - wind_pressure_imbalance(obs.wind_mean, obs.pressure_mean, r)
    )

    # Symmetry / variance metrics.
    if sim.wind_var is not None and obs.wind_var is not None:
        out["MAE_sigma_radial"] = float(profile_mae(sim.wind_var, obs.wind_var, mask))
    out["dVar_profile"] = float(
        profile_total_variance(sim.wind_mean, mask)
        - profile_total_variance(obs.wind_mean, mask)
    )

    # Precipitation morphology.
    if sim.precip_mean is not None and obs.precip_mean is not None:
        rm_sim, tm_sim = radius_of_extremum(sim.precip_mean, r, kind="max")
        rm_obs, tm_obs = radius_of_extremum(obs.precip_mean, r, kind="max")
        out["dR_m_km"] = abs(float(rm_sim) - float(rm_obs))
        out["dT_m"] = float(tm_sim) - float(tm_obs)
        out["dR_e_km"] = abs(
            efolding_radius(sim.precip_mean, r, rm_sim, tm_sim)
            - efolding_radius(obs.precip_mean, r, rm_obs, tm_obs)
        )
    return out


# --------------------------------------------------------------------------- #
# Layer 4: ensemble scoring (CRPS / RMSE)
# --------------------------------------------------------------------------- #
def crps_ensemble(
    target: np.ndarray, ensemble: np.ndarray, member_axis: int = 0
) -> np.ndarray:
    """Continuous Ranked Probability Score of an ensemble forecast.

    Uses the fair (unbiased) estimator of the identity
    ``CRPS = E|X - x| - 1/2 E|X - X'|`` with
    ``E|X - X'| = 1/(M(M-1)) sum_{i,j} |X_i - X_j|`` -- matching
    ``fme/downscaling/metrics_and_maths.py``. Returns NaN spread for a single
    member. ``target`` broadcasts against ``ensemble`` with ``member_axis`` removed.
    """
    ens = np.moveaxis(np.asarray(ensemble, float), member_axis, 0)
    target = np.asarray(target, float)
    n_members = ens.shape[0]
    mae = np.mean(np.abs(ens - target[None, ...]), axis=0)
    if n_members < 2:
        return mae - 0.5 * np.full(mae.shape, np.nan)
    pairwise = np.abs(ens[:, None, ...] - ens[None, :, ...]).sum(axis=(0, 1))
    spread = pairwise / (n_members * (n_members - 1))
    return mae - 0.5 * spread


def ensemble_rmse(
    target: np.ndarray, ensemble: np.ndarray, member_axis: int = 0
) -> np.ndarray:
    """Error of the ensemble *mean* about the target: ``|mean_m(X_m) - x|``.

    This is the RMSE of the ensemble mean evaluated per element -- each element is
    a single prediction, so RMSE reduces to the absolute error; aggregate over
    radius / cases outside to form a domain RMSE. It is the conventional companion
    to CRPS: both equal ``|error|`` for a deterministic (zero-spread) forecast, and
    unlike the member-RMS there is no forced ordering between CRPS(r) and RMSE(r).
    """
    ens = np.moveaxis(np.asarray(ensemble, float), member_axis, 0)
    target = np.asarray(target, float)
    return np.abs(np.nanmean(ens, axis=0) - target)


def radial_scores(
    target_mean: np.ndarray, gen_mean: np.ndarray, member_axis: int = 0
) -> dict[str, np.ndarray]:
    """Per-ring CRPS(r) and RMSE(r) of an ensemble of azimuthal-mean profiles.

    ``target_mean`` has shape ``(nbins,)`` and ``gen_mean`` has an ensemble axis
    (default axis 0), e.g. ``(members, nbins)``. Returns ``{"crps", "rmse"}`` each
    shape ``(nbins,)`` -- the radial error curves.
    """
    return {
        "crps": crps_ensemble(target_mean, gen_mean, member_axis),
        "rmse": ensemble_rmse(target_mean, gen_mean, member_axis),
    }


def _crps_reference(target: float, members: np.ndarray) -> float:
    """Independent, un-vectorized fair-CRPS for a 1D ensemble (test reference)."""
    members = np.asarray(members, float)
    n = members.size
    mae = np.mean(np.abs(members - target))
    pairs = [abs(a - b) for a, b in itertools.permutations(members, 2)]
    spread = sum(pairs) / (n * (n - 1))
    return mae - 0.5 * spread
