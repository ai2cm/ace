"""Tests for the stats.nc-based normalization aggregator.

The two questions this file pins down:

1. **Per-source pool is exact**: aggregating per-experiment stats from
   stats.nc files reproduces the concatenate-then-compute baseline the
   old (zarr-rescanning) ``make_normalization`` produced — within
   float32 round-trip noise.
2. **Cross-source pool is equal-weight**: averaging per-model dicts
   collapses cleanly to ``mean(per_model_means)`` /
   ``mean(per_model_stds)`` regardless of how many experiments each
   model had.

Together these are the load-bearing math we lose if the stats-files
shape ever drifts away from what ``_aggregate_one_source`` /
``_pool_across_models`` expect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))


def _stats_for(rng: np.random.Generator, shape: tuple[int, int, int]) -> xr.Dataset:
    """Run ``compute_dataset_stats`` on a synthetic Gaussian field with
    the full-period stats slice that the aggregator expects."""
    from compute_stats import area_weights_2d, compute_dataset_stats
    from config import StatsPeriod

    nt, nlat, nlon = shape
    times = xr.date_range(
        "2010-01-01", periods=nt, freq="D", calendar="noleap"
    ).to_list()
    data = rng.normal(0, 1, size=shape).astype(np.float32)
    ds = xr.Dataset(
        {
            "V": xr.DataArray(
                data,
                dims=("time", "lat", "lon"),
                coords={"time": times},
            )
        }
    )
    w2d = area_weights_2d("F22.5", nlon)
    stats, _ = compute_dataset_stats(ds, w2d, periods=[StatsPeriod("full", None, None)])
    return stats, data, w2d


def test_aggregate_one_source_matches_concat():
    """Pooling two experiments via _aggregate_one_source matches
    concatenate-then-compute mean and std (and per-cell time mean)
    to float32 precision."""
    from make_normalization import _aggregate_one_source

    rng = np.random.default_rng(0)
    s1, a1, w2d = _stats_for(rng, (50, 45, 90))
    s2, a2, _ = _stats_for(rng, (30, 45, 90))

    agg = _aggregate_one_source([s1, s2], "full")["V"]

    # Ground truth: concatenate-then-compute, area-weighted.
    concat = np.concatenate([a1, a2], axis=0)
    w3 = np.broadcast_to(w2d, concat.shape)
    ws = w3 / w3.sum()
    true_mean = (concat * ws).sum()
    true_std = np.sqrt(((concat - true_mean) ** 2 * ws).sum())
    true_tmap = concat.mean(axis=0)

    assert abs(agg["mean"] - true_mean) < 1e-5, (agg["mean"], true_mean)
    assert abs(agg["std"] - true_std) < 1e-5, (agg["std"], true_std)
    # Per-cell map: float32 round-trip noise dominates. Loose tolerance.
    np.testing.assert_allclose(
        agg["time_mean_map"], true_tmap, atol=2e-4, err_msg="time_mean_map mismatch"
    )


def test_pool_across_models_equal_weight():
    """Three models contribute three different per-source means; the
    cross-source pooled mean must be their unweighted average,
    regardless of how many experiments each model carried."""
    from make_normalization import _aggregate_one_source, _pool_across_models

    per_model: dict[str, dict] = {}
    expected_means = []
    expected_stds = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        # Different number of experiments per model — pooling must not
        # over-weight the model with more experiments.
        n_exps = 1 + seed
        files = [_stats_for(rng, (40, 45, 90))[0] for _ in range(n_exps)]
        per_model[f"M{seed}"] = _aggregate_one_source(files, "full")
        expected_means.append(per_model[f"M{seed}"]["V"]["mean"])
        expected_stds.append(per_model[f"M{seed}"]["V"]["std"])

    pooled, contributors = _pool_across_models(per_model)
    assert sorted(contributors["V"]) == ["M0", "M1", "M2"]
    assert abs(pooled["V"]["mean"] - float(np.mean(expected_means))) < 1e-12
    assert abs(pooled["V"]["std"] - float(np.mean(expected_stds))) < 1e-12


def test_aggregate_static_var_passes_through_static_map():
    """Static (no-time) vars carry a static_map in stats.nc; the
    aggregator must surface it as time_mean_map (which is what
    _write_map_nc consumes) and skip the d1 pooling."""
    from compute_stats import area_weights_2d, compute_dataset_stats
    from config import StatsPeriod
    from make_normalization import _aggregate_one_source

    nlat, nlon = 45, 90
    rng = np.random.default_rng(7)
    static = rng.uniform(0, 1, size=(nlat, nlon)).astype(np.float32)
    ds = xr.Dataset({"mask": xr.DataArray(static, dims=("lat", "lon"))})
    w2d = area_weights_2d("F22.5", nlon)
    stats, _ = compute_dataset_stats(ds, w2d, periods=[StatsPeriod("full", None, None)])

    agg = _aggregate_one_source([stats], "full")["mask"]
    np.testing.assert_allclose(agg["time_mean_map"], static, atol=0)
    assert agg["d1_mean"] == 0.0
    assert agg["d1_std"] == 0.0


def test_inject_trivial_norm_writes_mean0_std1_for_masks_and_land_fraction():
    """``_inject_trivial_norm`` populates the convention entries for
    every mask name + ``land_fraction``. Used by both cohort and
    per-source paths so that downstream lookups always succeed for
    the trivial-norm set regardless of upstream aggregation."""
    from make_normalization import _TRIVIAL_NORM_VARS, _inject_trivial_norm

    stats: dict[str, dict] = {}
    _inject_trivial_norm(stats)

    assert "land_fraction" in stats
    assert "below_surface_mask500" in stats
    assert "oday_tos_mask" in stats
    for name in _TRIVIAL_NORM_VARS:
        assert stats[name] == {
            "mean": 0.0,
            "std": 1.0,
            "d1_mean": 0.0,
            "d1_std": 1.0,
            "time_mean_map": None,
        }


def test_inject_trivial_norm_overrides_existing_land_fraction_entry():
    """``land_fraction`` has real per-dataset stats from the aggregator
    (~mean 0.3, std 0.45). The injection must *override* those with
    the 0/1 convention — preserving the natural [0, 1] domain so the
    network sees raw values, not a (val - 0.3) / 0.45 standardisation
    that scrambles the semantics."""
    from make_normalization import _inject_trivial_norm

    stats = {
        "land_fraction": {
            "mean": 0.3,
            "std": 0.45,
            "d1_mean": 0.0,
            "d1_std": 0.0,
            "time_mean_map": np.zeros((4, 8), dtype=np.float32),
        },
        "TMP2m": {
            "mean": 287.0,
            "std": 21.0,
            "d1_mean": 0.0,
            "d1_std": 0.4,
            "time_mean_map": None,
        },
    }
    _inject_trivial_norm(stats)

    assert stats["land_fraction"]["mean"] == 0.0
    assert stats["land_fraction"]["std"] == 1.0
    # Non-trivial-norm variable left untouched.
    assert stats["TMP2m"]["mean"] == 287.0
    assert stats["TMP2m"]["std"] == 21.0
