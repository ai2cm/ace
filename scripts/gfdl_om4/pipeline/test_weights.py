"""Native-area artifact build and load, on small ESMF grids.

Source: a regular regional grid (60°S–60°N). Target: a global grid, so its
rows past the source are unmapped (NaN markers in ``w``, ``A_J = 0``).
"""

import os

import numpy as np
import pytest
import xarray as xr
import xesmf as xe

from . import weights
from .grids import MEAN_EARTH_RADIUS_M
from .native_area import (
    ARTIFACT_ATTRS,
    ESMPY_AREA_RTOL,
    F64_RTOL,
    assert_close,
    esmf_cell_areas,
    native_area_weights,
)
from .native_area_weights import build_native_area_artifact
from .weights import (
    AREACELLO_FILENAME,
    REGRIDDER_KWARGS,
    SOURCE_GRID_FILENAME,
    WEIGHTS_FILENAME,
)


def _source_artifact(tmp_path):
    """Write an ESMF source artifact and a native ``areacello`` store.

    Returns (source_weights_url, a_source_url, target grid)."""
    source = xe.util.grid_2d(0, 360, 20, -60, 60, 15)
    target = xe.util.grid_global(30, 20)
    url = tmp_path / "source-artifact"
    url.mkdir()
    source.reset_coords().to_netcdf(url / SOURCE_GRID_FILENAME)
    xe.Regridder(source, target, **REGRIDDER_KWARGS).to_netcdf(
        str(url / WEIGHTS_FILENAME)
    )
    a_e = esmf_cell_areas(source).reshape(source.lat.shape)
    rng = np.random.default_rng(0)
    a = MEAN_EARTH_RADIUS_M**2 * a_e * (1 + 0.01 * rng.standard_normal(a_e.shape))
    a_url = str(tmp_path / "ocean_static.zarr")
    xr.Dataset({"areacello": (source.lat.dims, a)}).to_zarr(a_url)
    return str(url), a_url, target


def _built_artifact(tmp_path):
    """Returns (output_url, source_weights_url, a_source_url, target grid)."""
    source_url, a_url, target = _source_artifact(tmp_path)
    output_url = str(tmp_path / "native-area-v1" / "artifact")
    build_native_area_artifact(source_url, target, a_url, output_url)
    return output_url, source_url, a_url, target


def _fresh_caches(monkeypatch, target):
    monkeypatch.setattr(weights, "make_target_grid", lambda name: target)
    monkeypatch.setattr(weights, "_REGRIDDER_CACHE", {})
    monkeypatch.setattr(weights, "_AREACELLO_CACHE", {})


@pytest.mark.parametrize("attr", ARTIFACT_ATTRS)
def test_get_regridder_refuses_missing_attr(tmp_path, monkeypatch, attr):
    url, _, _, target = _built_artifact(tmp_path)
    _fresh_caches(monkeypatch, target)
    weights.get_regridder(url, "target")

    path = os.path.join(url, WEIGHTS_FILENAME)
    ds = xr.open_dataset(path).load()
    ds.close()
    del ds.attrs[attr]
    ds.to_netcdf(path)
    _fresh_caches(monkeypatch, target)
    with pytest.raises(ValueError, match=attr):
        weights.get_regridder(url, "target")


def test_open_areacello_cached(tmp_path, monkeypatch):
    url, _, _, target = _built_artifact(tmp_path)
    _fresh_caches(monkeypatch, target)
    area = weights.open_areacello(url)
    assert area.dtype == np.float64
    assert area.dims == target.lat.dims
    source_lat = weights.open_source_grid(url).lat
    regridded = weights.get_regridder(url, "target")(xr.ones_like(source_lat))
    assert regridded.dims == area.dims

    os.remove(os.path.join(url, AREACELLO_FILENAME))
    assert weights.open_areacello(url) is area


def test_build_artifact_contents(tmp_path):
    url, source_url, a_url, target = _built_artifact(tmp_path)
    source_grid = weights.open_source_grid(source_url)
    a = xr.open_zarr(a_url)["areacello"].values.ravel()
    expected = native_area_weights(
        weights.read_weights(source_url),
        esmf_cell_areas(target),
        esmf_cell_areas(source_grid),
        a,
        radius=MEAN_EARTH_RADIUS_M,
    )

    S, row, col = weights.read_weights(url)
    zero_rows = np.flatnonzero(expected.area == 0)
    assert zero_rows.size > 0
    assert not np.isin(row, zero_rows).any(), "no ŵ entry on Z rows"
    assert np.isfinite(S).all()
    assert S.dtype == np.float64

    with xr.open_dataset(os.path.join(url, AREACELLO_FILENAME)) as ds:
        area = ds["areacello"].load()
    assert area.dtype == np.float64
    assert_close(
        area.values.ravel(),
        expected.area,
        rtol=F64_RTOL,
        scale=expected.area,
        context="areacello = A_J",
    )
    w_hat = expected.w_hat.tocoo()
    np.testing.assert_array_equal(row, w_hat.row)
    np.testing.assert_array_equal(col, w_hat.col)
    assert_close(
        S, w_hat.data, rtol=F64_RTOL, scale=w_hat.data, context="weights.nc S = ŵ"
    )

    xr.testing.assert_identical(weights.open_source_grid(url), source_grid)

    with xr.open_dataset(os.path.join(url, WEIGHTS_FILENAME)) as ds:
        attrs = ds.attrs
    assert set(ARTIFACT_ATTRS) <= set(attrs)
    assert attrs["source_weights_url"] == source_url
    assert attrs["a_source_url"] == a_url
    assert attrs["esmpy_area_rtol"] == ESMPY_AREA_RTOL
    assert attrs["esmpy_max_rel_residual"] == expected.max_rel_residual
    assert attrs["a_over_a_tilde_spread"] == expected.a_over_a_tilde_spread


def test_build_artifact_refuses_existing_prefix(tmp_path):
    url, source_url, a_url, target = _built_artifact(tmp_path)
    with pytest.raises(FileExistsError):
        build_native_area_artifact(source_url, target, a_url, url)
