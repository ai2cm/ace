import dacite
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from get_pooled_stats import (
    Config,
    DatasetPair,
    _resolve_zarr_stores,
    compute_store_stats,
    pool_stats,
)


def _store_stats(values: np.ndarray) -> dict:
    """Per-store stats for a 1-D sample array, laid out as compute_store_stats
    returns them."""
    da = xr.DataArray(values, dims=["time"])
    return {
        "centering": xr.Dataset({"a": da.mean()}),
        "scaling_full_field": xr.Dataset({"a": da.std()}),
        "scaling_residual": xr.Dataset({"a": da.diff("time").std()}),
        "n_samples": len(values),
    }


def test_pool_stats_matches_stats_of_pooled_samples():
    rng = np.random.default_rng(0)
    # Distinct means and sizes: the pooled full-field std must include the
    # between-dataset spread of means, not just the weighted within-store
    # variances, to equal the std of the concatenated samples.
    samples = [rng.normal(0.0, 1.0, 100), rng.normal(5.0, 2.0, 50)]
    pooled = pool_stats([_store_stats(s) for s in samples])
    all_samples = np.concatenate(samples)
    np.testing.assert_allclose(pooled["centering.nc"]["a"], all_samples.mean())
    np.testing.assert_allclose(pooled["scaling-full-field.nc"]["a"], all_samples.std())


def test_pool_stats_residual_ignores_offset_between_datasets():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 100)
    # A constant offset changes a store's mean but not its time-difference std,
    # so pooling the residual std must not pick up a between-dataset term.
    pooled = pool_stats([_store_stats(base), _store_stats(base + 100.0)])
    expected = np.diff(base).std()
    np.testing.assert_allclose(pooled["scaling-residual.nc"]["a"], expected)


def test_dataset_pair_rejects_group_and_groups():
    with pytest.raises(ValueError, match="group/groups"):
        DatasetPair(dataset="a", group="x", groups=["y"])


def test_dataset_pair_group_names():
    assert DatasetPair(dataset="a").group_names == []
    assert DatasetPair(dataset="a", group="x").group_names == ["x"]
    assert DatasetPair(dataset="a", groups=["x", "y"]).group_names == ["x", "y"]


def test_config_rejects_group_mismatch():
    with pytest.raises(ValueError, match="not claimed"):
        Config(dataset_pairs=[DatasetPair(dataset="a")], groups=["x"])
    with pytest.raises(ValueError, match="not listed"):
        Config(dataset_pairs=[DatasetPair(dataset="a", group="x")], groups=[])
    Config(
        dataset_pairs=[DatasetPair(dataset="a", groups=["x", "y"])], groups=["x", "y"]
    )


def test_config_rejects_empty_dataset_pairs():
    with pytest.raises(ValueError, match="at least one"):
        Config(dataset_pairs=[])


def test_config_from_file_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset_pairs:\n"
        "  - dataset: gs://bucket/store.zarr\n"
        "    stop_time: '2000-01-01'\n"
    )
    with pytest.raises(dacite.UnexpectedDataError):
        Config.from_file(str(path))


def test_output_subdir():
    pair = DatasetPair(dataset="d", start_time="2000-01-01")
    assert pair.output_subdir(3, "gs://b/ic_0001.zarr/") == "03_ic_0001_2000-01-01_max"


def _write_store(path, n_time: int = 10) -> xr.Dataset:
    time = pd.date_range("2000-01-01", periods=n_time, freq="D")
    values = np.arange(n_time, dtype=float)[:, None, None] * np.ones((1, 2, 3))
    ds = xr.Dataset(
        {
            "a": (("time", "lat", "lon"), values),
            "land_sea_mask": (("lat", "lon"), np.ones((2, 3))),
        },
        coords={"time": time},
    )
    ds.to_zarr(path, mode="w")
    return ds


def test_resolve_zarr_stores_finds_nested_stores(tmp_path):
    for name in ["ic_0001.zarr", "ic_0002.zarr", "sub/ic_0003.zarr"]:
        _write_store(tmp_path / name)
    stores = _resolve_zarr_stores(str(tmp_path))
    assert [s.rsplit("/", 1)[-1] for s in stores] == [
        "ic_0001.zarr",
        "ic_0002.zarr",
        "ic_0003.zarr",
    ]
    # A path pointing directly at a store resolves to just that store.
    direct = str(tmp_path / "ic_0001.zarr")
    assert _resolve_zarr_stores(direct) == [direct]


def test_resolve_zarr_stores_raises_when_none_found(tmp_path):
    with pytest.raises(ValueError, match="No zarr store"):
        _resolve_zarr_stores(str(tmp_path))


def test_compute_store_stats_slices_time(tmp_path):
    pytest.importorskip("dask")
    store = str(tmp_path / "store.zarr")
    _write_store(store)
    stats = compute_store_stats(store, slice("2000-01-03", "2000-01-06"))
    # Days 3..6 are values 2..5; the unsliced store (0..9) would give mean 4.5.
    assert stats["n_samples"] == 4
    np.testing.assert_allclose(stats["centering"]["a"], 3.5)
    np.testing.assert_allclose(stats["scaling_full_field"]["a"], np.std([2, 3, 4, 5]))
    np.testing.assert_allclose(stats["scaling_residual"]["a"], 0.0)
    assert "land_sea_mask" not in stats["centering"]


def test_compute_store_stats_rejects_single_timestep(tmp_path):
    pytest.importorskip("dask")
    store = str(tmp_path / "store.zarr")
    _write_store(store)
    with pytest.raises(ValueError, match="1 timesteps"):
        compute_store_stats(store, slice("2000-01-03", "2000-01-03"))
