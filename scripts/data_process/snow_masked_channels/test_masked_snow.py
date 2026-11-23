import shutil

import numpy as np
import pytest
import xarray as xr
from fit_masked_snow_stats import Moments
from masked_snow import (
    MASKED,
    PARENTS,
    SHARD_STEPS,
    SOURCE_SETS,
    land_snow_fields,
    store_complete,
)


def _inputs():
    land_frac = np.array([[1.0, 0.5], [0.8, 0.2]])
    valid = np.array([[True, True], [True, False]])
    swe = np.array([[[100.0, 100.0], [100.0, 100.0]]])
    return land_frac, valid, swe


def test_era5_divides_by_land_fraction_and_clips_cover():
    land_frac, valid, swe = _inputs()
    scf = np.array([[[0.5, 0.5], [0.9, 1.0]]])
    out_swe, out_scf = land_snow_fields(swe, scf, PARENTS["era5"], land_frac, valid)
    np.testing.assert_allclose(out_swe[0, 0], [100.0, 200.0])
    np.testing.assert_allclose(out_swe[0, 1, 0], 125.0)
    np.testing.assert_allclose(out_scf[0, 0], [0.5, 1.0])
    np.testing.assert_allclose(out_scf[0, 1, 0], 1.0)


def test_cm4_rescales_cover_only():
    land_frac, valid, swe = _inputs()
    scf = np.array([[[50.0, 100.0], [25.0, 0.0]]])
    out_swe, out_scf = land_snow_fields(swe, scf, PARENTS["cm4"], land_frac, valid)
    np.testing.assert_allclose(out_swe[0, 0], [100.0, 100.0])
    np.testing.assert_allclose(out_scf[0, 0], [0.5, 1.0])
    np.testing.assert_allclose(out_scf[0, 1, 0], 0.25)


@pytest.mark.parametrize("dataset", ["era5", "cm4"])
def test_nan_outside_mask_and_float32(dataset):
    land_frac, valid, swe = _inputs()
    scf = np.zeros_like(swe)
    out_swe, out_scf = land_snow_fields(swe, scf, PARENTS[dataset], land_frac, valid)
    assert np.isnan(out_swe[0, 1, 1]) and np.isnan(out_scf[0, 1, 1])
    assert np.isfinite(out_swe[0][valid]).all()
    assert out_swe.dtype == np.float32 and out_scf.dtype == np.float32


def test_era5_rejects_mask_cells_with_low_land_fraction():
    land_frac, _, swe = _inputs()
    valid = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        land_snow_fields(swe, np.zeros_like(swe), PARENTS["era5"], land_frac, valid)


def test_moments_match_direct_statistics_and_pool_across_blocks():
    rng = np.random.default_rng(0)
    values = rng.normal(3.0, 2.0, size=(40, 4, 5))
    values[:, 0, 0] = np.nan
    diffs = rng.normal(0.0, 0.5, size=values.shape)
    diffs[:, 0, 0] = np.nan
    pooled = Moments()
    for block in (slice(0, 15), slice(15, 40)):
        pooled.add(values[block], diffs[block])
    entry = pooled.entry()
    assert entry["mean"] == pytest.approx(np.nanmean(values))
    assert entry["std"] == pytest.approx(np.nanstd(values))
    assert entry["residual_std"] == pytest.approx(np.nanstd(diffs))
    valid = np.ones((4, 5), bool)
    valid[0, 0] = False
    time_mean = pooled.time_mean(valid)
    assert np.isnan(time_mean[0, 0])
    np.testing.assert_allclose(time_mean[valid], np.nanmean(values, axis=0)[valid])


def test_source_sets_name_registered_parents():
    for keys in SOURCE_SETS.values():
        assert all(key in PARENTS for key in keys)
    assert SOURCE_SETS["pic-1pct"] == SOURCE_SETS["pic-1pct-randco2"][:2]
    assert not any("ic3" in key for key in SOURCE_SETS["pic-1pct-randco2"])


def _write_masked_store(path, n_steps):
    shape = (n_steps, 2, 3)
    ds = xr.Dataset(
        {
            name: (("time", "lat", "lon"), np.zeros(shape, np.float32))
            for name in MASKED.values()
        },
        coords={"time": np.arange(n_steps)},
    )
    encoding = {
        name: {"chunks": (1, 2, 3), "shards": (SHARD_STEPS, 2, 3)}
        for name in MASKED.values()
    }
    ds.to_zarr(path, mode="w", encoding=encoding)


def test_store_complete_requires_every_step_and_shard(tmp_path):
    n_steps = SHARD_STEPS + 40
    store = tmp_path / "store.zarr"
    assert not store_complete(str(store), n_steps)
    _write_masked_store(store, n_steps)
    assert store_complete(str(store), n_steps)
    assert not store_complete(str(store), n_steps + SHARD_STEPS)
    shutil.rmtree(store / MASKED["surface_snow_amount"] / "c" / "1")
    assert not store_complete(str(store), n_steps)
