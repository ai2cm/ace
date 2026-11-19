import numpy as np
import pytest
import xarray as xr
import yaml
from compute_sea_ice_mask_limits import compute_limits, config_block

LAT = np.round(0.8 * np.arange(-9.5, 10.0, 1.0), 1)  # not 1 deg apart: D16 (b)


def _write_store(path, ice_cells: list[tuple[int, float, int, float]]) -> str:
    """ice_cells: (time index, lat, lon index, sea-ice fraction)."""
    sif = np.zeros((3, LAT.size, 4), dtype=np.float32)
    sif[:, :, 3] = np.nan  # a land column
    for t, lat, i_lon, value in ice_cells:
        sif[t, np.flatnonzero(LAT == lat)[0], i_lon] = value
    ds = xr.Dataset(
        {"ice": (("time", "lat", "lon"), sif)},
        coords={"lat": LAT, "lon": np.arange(4.0)},
    )
    ds.to_zarr(path)
    return str(path)


def test_compute_limits_per_hemisphere_first_ice_free_row(tmp_path):
    store = _write_store(
        tmp_path / "a.zarr",
        [(0, -6.8, 0, 0.5), (2, -5.2, 1, 1e-20), (1, 6.0, 2, 0.5), (0, 7.6, 0, 0.9)],
    )
    result = compute_limits(store, "ice", time_chunk=2, num_workers=1)
    assert result["config"] == {"latitude_limits": {"north": 5.2, "south": 4.4}}
    assert yaml.safe_load(config_block(result)) == result["config"]
    assert result["hemispheres"]["north"]["edge_lat"] == 6.0
    assert result["hemispheres"]["south"]["edge_lat"] == -5.2
    assert result["hemispheres"]["north"]["masked_rows"] == 6
    assert result["hemispheres"]["south"]["masked_rows"] == 5


def test_compute_limits_raises_without_ice_in_a_hemisphere(tmp_path):
    store = _write_store(tmp_path / "a.zarr", [(0, 5.2, 0, 0.5)])
    with pytest.raises(ValueError, match="south"):
        compute_limits(store, "ice", 2, 1)


def test_compute_limits_raises_when_ice_reaches_equator_row(tmp_path):
    store = _write_store(tmp_path / "a.zarr", [(0, -5.2, 0, 0.5), (0, 0.4, 0, 0.5)])
    with pytest.raises(ValueError, match="nearest the equator"):
        compute_limits(store, "ice", 2, 1)


def test_config_block_keeps_unrounded_limits():
    config = {"latitude_limits": {"north": 27.42370865495453, "south": 1 / 3}}
    assert yaml.safe_load(config_block({"config": config})) == config
