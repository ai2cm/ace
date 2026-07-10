"""Verified unit tests for tc_radial_metrics.

Each test checks a calculation against an independent reference: a closed-form
value, a hand-computed small case, or a second (un-vectorized) implementation --
not merely that the function runs.

Run:
    ~/miniconda3/bin/conda run -n fme python -m pytest \
        scripts/tropical_cyclones/test_tc_radial_metrics.py -q
"""

import itertools

import numpy as np
import pytest
import tc_radial_metrics as trm

np.random.seed(0)


# --------------------------------------------------------------------------- #
# Layer 1
# --------------------------------------------------------------------------- #
def test_great_circle_one_degree_at_equator():
    # 1 degree of arc = R * (pi/180); identical point = 0.
    dist = trm.great_circle_distance_km(0.0, 0.0, np.array([0.0]), np.array([0.0, 1.0]))
    assert dist.shape == (1, 2)
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-9)
    expected = trm.EARTH_RADIUS_KM * np.radians(1.0)  # ~111.19 km
    assert dist[0, 1] == pytest.approx(expected, rel=1e-6)


def test_great_circle_symmetric_across_dateline():
    # 179E to 179W (i.e. 181) is 2 degrees apart, not 358.
    d = trm.great_circle_distance_km(0.0, 179.0, np.array([0.0]), np.array([-179.0]))
    assert d[0, 0] == pytest.approx(trm.EARTH_RADIUS_KM * np.radians(2.0), rel=1e-6)


def test_azimuthal_stats_binning_mean_var_count():
    # Direct synthetic distance field so binning is unambiguous.
    distance = np.array([[5.0, 5.0], [30.0, 30.0]])
    edges = np.array([0.0, 10.0, 20.0, 40.0])
    field = np.array([[1.0, 3.0], [10.0, 20.0]])
    mean, var, count = trm.azimuthal_stats(field, distance, edges)
    np.testing.assert_array_equal(count, [2, 0, 2])
    np.testing.assert_allclose(mean[0], 2.0)  # (1+3)/2
    assert np.isnan(mean[1])  # empty ring
    np.testing.assert_allclose(mean[2], 15.0)  # (10+20)/2
    np.testing.assert_allclose(var[0], 1.0)  # var([1,3]) = 1
    np.testing.assert_allclose(var[2], 25.0)  # var([10,20]) = 25


def test_azimuthal_stats_batched_leading_dim():
    distance = np.array([[5.0, 5.0], [30.0, 30.0]])
    edges = np.array([0.0, 10.0, 20.0, 40.0])
    field = np.stack([np.array([[1.0, 3.0], [10.0, 20.0]]), np.ones((2, 2))])
    mean, var, count = trm.azimuthal_stats(field, distance, edges)
    assert mean.shape == (2, 3)
    np.testing.assert_allclose(mean[1, 0], 1.0)  # second member all ones
    np.testing.assert_allclose(var[1, 0], 0.0)


def test_refine_center_min_slp():
    lats = np.array([0.0, 5.0, 10.0])
    lons = np.array([0.0, 5.0, 10.0])
    pressure = np.full((3, 3), 101000.0)
    pressure[1, 1] = 98000.0  # min at (5, 5)
    got = trm.refine_center_min_slp(
        pressure, lats, lons, center=(0.0, 0.0), search_radius_km=1500.0
    )
    assert got == (5.0, 5.0)
    # With a tight radius the far min is excluded; nearest point (0,0) wins.
    got2 = trm.refine_center_min_slp(
        pressure, lats, lons, center=(0.0, 0.0), search_radius_km=100.0
    )
    assert got2 == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# Layer 2
# --------------------------------------------------------------------------- #
def test_radius_of_max_wind():
    r = np.array([10.0, 20.0, 30.0, 40.0])
    wind = np.array([5.0, 20.0, 15.0, 8.0])
    r_max, v_max = trm.radius_of_max_wind(wind, r)
    assert float(r_max) == 20.0
    assert float(v_max) == 20.0


def test_radius_of_extremum_min():
    r = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([100.0, 90.0, 95.0, 99.0])
    r_min, v_min = trm.radius_of_extremum(p, r, kind="min")
    assert float(r_min) == 20.0
    assert float(v_min) == 90.0


def test_wind_radii_interpolated_crossings():
    r = np.array([10.0, 20.0, 30.0, 40.0])
    wind = np.array([40.0, 30.0, 20.0, 10.0])
    # threshold 25: between r=20 (30) and r=30 (20) -> 25 km
    assert trm._outermost_threshold_radius(wind, r, 25.0) == pytest.approx(25.0)
    # threshold 15: between r=30 (20) and r=40 (10) -> 35 km
    assert trm._outermost_threshold_radius(wind, r, 15.0) == pytest.approx(35.0)
    # never reached
    assert np.isnan(trm._outermost_threshold_radius(wind, r, 45.0))
    # still above at domain edge -> edge radius
    assert trm._outermost_threshold_radius(wind, r, 5.0) == pytest.approx(40.0)


def test_profile_mae_with_mask():
    sim = np.array([1.0, 2.0, 3.0])
    obs = np.array([1.0, 1.0, 1.0])
    assert trm.profile_mae(sim, obs) == pytest.approx(1.0)  # mean(|0,1,2|)
    mask = np.array([True, True, False])
    assert trm.profile_mae(sim, obs, mask) == pytest.approx(0.5)  # mean(|0,1|)


def test_profile_total_variance():
    assert trm.profile_total_variance(np.array([1.0, 2.0, 3.0])) == pytest.approx(
        2.0 / 3.0
    )


def test_pressure_gradient_force_linear():
    r = np.array([10.0, 20.0, 30.0, 40.0])
    slope_pa_per_km = 5.0
    p = 100000.0 + slope_pa_per_km * r
    pgf = trm.pressure_gradient_force(p, r, rho=1.15)
    expected = (slope_pa_per_km / 1000.0) / 1.15  # dP/dr[m] / rho, constant
    np.testing.assert_allclose(pgf, expected, rtol=1e-10)


def test_radial_mismatch():
    r = np.array([10.0, 20.0, 30.0, 40.0])
    pgf = np.array([1.0, 2.0, 5.0, 3.0])  # max at r=30
    wind = np.array([5.0, 20.0, 10.0, 8.0])  # max at r=20
    assert trm.radial_mismatch(pgf, wind, r) == pytest.approx(10.0)


def test_wind_pressure_imbalance_zero_wind_is_pressure_term():
    r = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([98000.0, 99000.0, 100000.0, 101000.0])
    wind = np.zeros_like(r)
    got = trm.wind_pressure_imbalance(wind, p, r, rho=1.15)
    assert got == pytest.approx((p[-1] - p[0]) / 1.15)


def test_wind_pressure_imbalance_matches_explicit_trapezoid():
    r = np.array([10.0, 20.0, 30.0, 40.0])
    r_m = r * 1000.0
    p = np.array([98000.0, 99000.0, 100000.0, 101000.0])
    wind = np.array([30.0, 25.0, 18.0, 12.0])
    integrand = wind**2 / r_m
    # Independent trapezoid (not np.trapz).
    integral = sum(
        0.5 * (integrand[i] + integrand[i + 1]) * (r_m[i + 1] - r_m[i])
        for i in range(len(r_m) - 1)
    )
    expected = (p[-1] - p[0]) / 1.15 - integral
    assert trm.wind_pressure_imbalance(wind, p, r, rho=1.15) == pytest.approx(expected)


def test_efolding_radius_exponential_decay():
    r = np.arange(2.5, 400.0, 2.5)
    r_m, t_m, length = 50.0, 10.0, 60.0
    precip = np.where(
        r < r_m,
        t_m * (r / r_m),  # ramp up to the peak
        t_m * np.exp(-(r - r_m) / length),  # exponential decay outward
    )
    # value hits T_m/e at r = r_m + length.
    got = trm.efolding_radius(precip, r)
    assert got == pytest.approx(r_m + length, abs=2.5)


# --------------------------------------------------------------------------- #
# Layer 4: ensemble scoring
# --------------------------------------------------------------------------- #
def test_crps_deterministic_is_absolute_error():
    members = np.full(4, 3.0)
    assert trm.crps_ensemble(2.0, members) == pytest.approx(1.0)  # spread 0 -> |3-2|


def test_crps_two_member_analytic():
    # members {0, 4}, x = 5: mae = (5+1)/2 = 3, fair spread = |0-4| = 4.
    assert trm.crps_ensemble(5.0, np.array([0.0, 4.0])) == pytest.approx(
        3.0 - 0.5 * 4.0
    )
    # x = 2 (between members): crps = 1 - 0.5*|1-3| = 0.
    assert trm.crps_ensemble(2.0, np.array([1.0, 3.0])) == pytest.approx(0.0)


def test_crps_matches_independent_reference():
    members = np.random.rand(6) * 10.0
    target = 4.2
    ref = trm._crps_reference(target, members)
    assert trm.crps_ensemble(target, members) == pytest.approx(ref)


def test_crps_single_member_is_nan():
    assert np.isnan(trm.crps_ensemble(1.0, np.array([2.0])))


def test_crps_and_rmse_batched_match_per_bin_reference():
    gen = np.random.rand(8, 5)  # (members, nbins)
    target = np.random.rand(5)
    crps = trm.crps_ensemble(target, gen, member_axis=0)
    rmse = trm.ensemble_rmse(target, gen, member_axis=0)
    assert crps.shape == (5,) and rmse.shape == (5,)
    # Each bin must equal the 1D per-bin computation (verifies the vectorization).
    for b in range(5):
        assert crps[b] == pytest.approx(trm._crps_reference(target[b], gen[:, b]))
        assert rmse[b] == pytest.approx(abs(np.mean(gen[:, b]) - target[b]))


def test_ensemble_rmse_is_ensemble_mean_error():
    members = np.array([1.0, 2.0, 3.0])  # mean 2.0
    assert trm.ensemble_rmse(0.5, members) == pytest.approx(1.5)  # |2.0 - 0.5|
    # deterministic (zero-spread) forecast -> equals |error|, like CRPS.
    assert trm.ensemble_rmse(0.5, np.full(3, 2.0)) == pytest.approx(1.5)


def test_crps_reference_equals_permutation_definition():
    # Guard the test's own reference against the plain permutation formula.
    members = np.array([1.0, 5.0, 9.0])
    target = 3.0
    mae = np.mean(np.abs(members - target))
    pairs = [abs(a - b) for a, b in itertools.permutations(members, 2)]
    spread = sum(pairs) / (3 * 2)
    assert trm._crps_reference(target, members) == pytest.approx(mae - 0.5 * spread)


# --------------------------------------------------------------------------- #
# Layer 3: scorecard assembly (signs, sim/obs ordering, valid-bin mask)
# --------------------------------------------------------------------------- #
def test_scorecard_assembly_signs_and_wiring():
    r = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # obs wind peaks at r=20; sim wind peaks at r=30. Last bin below 10 m/s (masked).
    obs = trm.StormProfiles(
        r_km=r,
        wind_mean=np.array([12.0, 30.0, 20.0, 15.0, 8.0]),
        pressure_mean=np.array([98000.0, 98500.0, 99500.0, 100200.0, 100600.0]),
        wind_var=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        precip_mean=np.array([20.0, 15.0, 8.0, 3.0, 1.0]),
    )
    sim = trm.StormProfiles(
        r_km=r,
        wind_mean=np.array([12.0, 25.0, 28.0, 15.0, 8.0]),
        pressure_mean=np.array([98200.0, 98900.0, 99700.0, 100100.0, 100500.0]),
        wind_var=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        precip_mean=np.array([15.0, 12.0, 7.0, 3.0, 1.0]),
    )
    card = trm.scorecard(sim, obs)

    # dR_max: |30 - 20| = 10 (absolute, exact).
    assert card["dR_max_km"] == pytest.approx(10.0)
    # dT_m: signed (sim peak - obs peak) = 15 - 20 = -5.
    assert card["dT_m"] == pytest.approx(-5.0)
    # dR_m: both precip peaks at r=10 -> 0.
    assert card["dR_m_km"] == pytest.approx(0.0)

    # Masking: only bins where obs wind >= 10 (first four) enter the wind MAE.
    mask = trm.valid_wind_mask(obs.wind_mean)
    np.testing.assert_array_equal(mask, [True, True, True, True, False])
    assert card["MAE_v_radial"] == pytest.approx(
        trm.profile_mae(sim.wind_mean, obs.wind_mean, mask)
    )

    # Dynamic-consistency metrics wired to PGF-of-pressure with correct sim/obs order.
    pgf_sim = trm.pressure_gradient_force(sim.pressure_mean, r)
    pgf_obs = trm.pressure_gradient_force(obs.pressure_mean, r)
    assert card["dR_mismatch_km"] == pytest.approx(
        abs(
            trm.radial_mismatch(pgf_sim, sim.wind_mean, r)
            - trm.radial_mismatch(pgf_obs, obs.wind_mean, r)
        )
    )
    # dVar_profile: signed (sim - obs); sim wind is flatter so it should be negative.
    assert card["dVar_profile"] == pytest.approx(
        trm.profile_total_variance(sim.wind_mean, mask)
        - trm.profile_total_variance(obs.wind_mean, mask)
    )
    assert card["dVar_profile"] < 0.0


# --------------------------------------------------------------------------- #
# Integration: transform + structural metric on a synthetic vortex
# --------------------------------------------------------------------------- #
def test_radial_profile_recovers_rmw_of_synthetic_vortex():
    lats = np.linspace(10.0, 26.0, 65)  # 16 deg box centered ~18N
    lons = np.linspace(122.0, 138.0, 65)  # centered ~130E
    center_lat, center_lon = 18.0, 130.0
    dist = trm.great_circle_distance_km(center_lat, center_lon, lats, lons)
    # Modified-Rankine-like profile with RMW at ~100 km.
    rmw_km = 100.0
    safe_dist = np.maximum(dist, 1e-6)  # avoid 0/0 at the exact center grid point
    wind = np.where(
        dist <= rmw_km, 50.0 * dist / rmw_km, 50.0 * (rmw_km / safe_dist) ** 0.5
    )
    edges = trm.radial_bin_edges(dr_km=25.0, r_max_km=500.0)
    prof = trm.compute_radial_profile(wind, center_lat, center_lon, lats, lons, edges)
    r_max, v_max = trm.radius_of_max_wind(prof.mean, prof.r_km)
    assert float(r_max) == pytest.approx(rmw_km, abs=25.0)  # within one ring
    assert float(v_max) == pytest.approx(50.0, rel=0.1)
