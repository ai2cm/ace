import numpy as np
import pytest
import xarray as xr
from get_stats import compute_variogram_scaling, get_variogram_edge_offsets
from get_variogram_scaling import get_merged_variogram_scaling

from fme.core.ensemble import (
    get_variogram_edge_offsets as get_loss_variogram_edge_offsets,
)


def _dataset(seed: int = 0, n_time: int = 3, n_lat: int = 4, n_lon: int = 6):
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "a": (("time", "lat", "lon"), rng.normal(size=(n_time, n_lat, n_lon))),
            "static": (("lat", "lon"), rng.normal(size=(n_lat, n_lon))),
            "series": (("time",), rng.normal(size=n_time)),
        },
        coords={
            "time": np.arange(n_time),
            "lat": np.linspace(-60, 60, n_lat),
            "lon": np.arange(n_lon) * 360.0 / n_lon,
        },
    )


def _oracle(x: np.ndarray, di: int, dj: int) -> float:
    """RMS increment by explicit loop over point pairs, NaNs excluded."""
    n_lat, n_lon = x.shape[-2:]
    squares = []
    for i in range(n_lat - di):
        for j in range(n_lon):
            inc = x[..., i + di, (j + dj) % n_lon] - x[..., i, j]
            squares.extend(np.atleast_1d(inc**2).tolist())
    squares_arr = np.array(squares)
    return float(np.sqrt(np.nanmean(squares_arr)))


def test_edge_offsets_match_loss():
    for window_size in [3, 5]:
        assert get_variogram_edge_offsets(window_size) == (
            get_loss_variogram_edge_offsets(window_size)
        )


@pytest.mark.parametrize("window_size", [3, 5])
def test_compute_variogram_scaling_matches_oracle(window_size: int):
    ds = _dataset()
    ds["a"][0, 1, 2] = np.nan  # a masked point drops out of its pairs
    result = compute_variogram_scaling(
        ds, lat_dim="lat", lon_dim="lon", window_size=window_size
    )
    offsets = get_variogram_edge_offsets(window_size)
    assert set(result.data_vars) == {"a", "static"}
    assert result["a"].dims == ("edge",)
    assert list(zip(result["di"].values, result["dj"].values)) == offsets
    assert np.issubdtype(result["di"].dtype, np.integer)
    for name in ["a", "static"]:
        expected = [_oracle(ds[name].values, di, dj) for di, dj in offsets]
        np.testing.assert_allclose(result[name].values, expected)


def test_compute_variogram_scaling_no_pole_wrap():
    """Rows 0, 0, 1: the north-south pairs are row 0 to 1 (increment 0) and
    row 1 to 2 (increment 1), so the RMS is sqrt(1/2). A wrap from row 2 back
    to row 0 would add another unit increment (sqrt(2/3))."""
    x = np.zeros((1, 3, 5))
    x[:, 2, :] = 1.0
    ds = xr.Dataset({"a": (("time", "lat", "lon"), x)})
    result = compute_variogram_scaling(ds, lat_dim="lat", lon_dim="lon", window_size=3)
    by_offset = dict(
        zip(zip(result["di"].values, result["dj"].values), result["a"].values)
    )
    assert by_offset[(0, 1)] == 0.0
    for dj in [-1, 0, 1]:
        np.testing.assert_allclose(by_offset[(1, dj)], np.sqrt(0.5))


def test_get_merged_variogram_scaling_first_store_wins(tmp_path):
    first = _dataset(seed=0).drop_vars("static")
    second = _dataset(seed=1)
    second["b"] = second["a"] * 2.0
    first_path = str(tmp_path / "first.zarr")
    second_path = str(tmp_path / "second.zarr")
    first.to_zarr(first_path)
    second.to_zarr(second_path)
    result = get_merged_variogram_scaling(
        [first_path, second_path],
        lat_dim="lat",
        lon_dim="lon",
        start_date=None,
        end_date=None,
        time_chunk=2,
    )
    assert set(result.data_vars) == {"a", "b", "static"}
    assert list(zip(result["di"].values, result["dj"].values)) == (
        get_variogram_edge_offsets(5)
    )
    expected_a = compute_variogram_scaling(first, lat_dim="lat", lon_dim="lon")
    expected_b = compute_variogram_scaling(second, lat_dim="lat", lon_dim="lon")
    np.testing.assert_allclose(result["a"].values, expected_a["a"].values)
    np.testing.assert_allclose(result["b"].values, expected_b["b"].values)
    np.testing.assert_allclose(result["static"].values, expected_b["static"].values)
