"""Step-1 tests on the PRD toy example and on small ESMF grids.

Toy (PRD ``## Toy example``): native a = (1, 1, 2), ESMF sizes ã = (1, 1.1, 2),
ESMF overlaps o_J1 = (1, 0.55, 0), o_J2 = (0, 0.55, 2); table 3 adds row J3
with no overlap, stored as one NaN marker at source column 0. R = 1, and
w_Ji = o_Ji / A^esmf_J with A^esmf_J = Σ_i o_Ji (J3: any positive area).

Table values are read through the load path the pipeline ships: the toy is
written as an artifact and loaded with ``get_regridder``, ``open_areacello``
and ``read_weights``.
"""

import os
from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse
import xarray as xr
import xesmf as xe

from . import weights
from .grids import MEAN_EARTH_RADIUS_M
from .native_area import (
    ESMPY_AREA_RTOL,
    F64_RTOL,
    assert_artifact_invariants,
    assert_bound,
    assert_close,
    esmf_cell_areas,
    native_area_weights,
)
from .native_area_weights import write_native_area_artifact
from .weights import REGRIDDER_KWARGS, WEIGHTS_FILENAME

A_NATIVE = np.array([1.0, 1.0, 2.0])
A_TILDE_ESMF = np.array([1.0, 1.1, 2.0])
O_ESMF = np.array([[1.0, 0.55, 0.0], [0.0, 0.55, 2.0]])
A_ESMF_J3 = 1.0


def _toy_weights(z_row: bool):
    """Toy (S, row, col) and A^esmf_J; with ``z_row``, J3 holds only the NaN marker."""
    A_e = O_ESMF.sum(axis=1)
    w = scipy.sparse.coo_matrix(O_ESMF / A_e[:, None])
    S, row, col = w.data, w.row, w.col
    if z_row:
        S, row, col = np.append(S, np.nan), np.append(row, 2), np.append(col, 0)
        A_e = np.append(A_e, A_ESMF_J3)
    return (S, row, col), A_e


def _toy(z_row: bool = False):
    w, A_e = _toy_weights(z_row)
    return w, native_area_weights(w, A_e, A_TILDE_ESMF, A_NATIVE, radius=1.0)


def _toy_grid(lon_b) -> xr.Dataset:
    """A one-row lat/lon grid; geometry is unused, the toy weights are given."""
    lon_b = np.asarray(lon_b, dtype=np.float64)
    return xr.Dataset(
        coords={
            "lat": ("lat", [0.0]),
            "lon": ("lon", 0.5 * (lon_b[:-1] + lon_b[1:])),
            "lat_b": ("lat_b", [-0.5, 0.5]),
            "lon_b": ("lon_b", lon_b),
        }
    )


TOY_ATTRS = {
    "source_weights_url": "toy",
    "a_source_url": "toy",
    "esmpy_area_rtol": ESMPY_AREA_RTOL,
    "esmpy_max_rel_residual": 0.0,
}


def _toy_artifact(tmp_path, monkeypatch, z_row: bool = False):
    """Write the toy as a weight artifact and load it as the pipeline does.

    Returns the regridder from ``get_regridder`` and the artifact URL."""
    w, result = _toy(z_row)
    target = _toy_grid([0.0, 1.5, 3.0, 4.0] if z_row else [0.0, 1.5, 3.0])
    url = str(tmp_path / "toy-artifact")
    attrs = TOY_ATTRS | {"a_over_a_tilde_spread": result.a_over_a_tilde_spread}
    write_native_area_artifact(result, _toy_grid([0, 1, 2, 3]), target, attrs, url)
    monkeypatch.setattr(weights, "make_target_grid", lambda name: target)
    monkeypatch.setattr(weights, "_REGRIDDER_CACHE", {})
    monkeypatch.setattr(weights, "_AREACELLO_CACHE", {})
    return weights.get_regridder(url, "toy"), url


def _regrid(regridder, native) -> np.ndarray:
    native = np.asarray(native, dtype=np.float64)
    return regridder(xr.DataArray(native[None, :], dims=("lat", "lon"))).values.ravel()


def _overlap(regridder, url) -> np.ndarray:
    """O_Ji = A_J · regridder(e_i)_J, 0 on Z."""
    area = weights.open_areacello(url).values.ravel()
    columns = [_regrid(regridder, e) for e in np.eye(A_NATIVE.size)]
    return np.nan_to_num(area[:, None] * np.stack(columns, axis=1))


def _assert_tabled(lhs, tabled, context, scale=None):
    rhs = np.array([float(x) for x in np.ravel(tabled)]).reshape(np.shape(tabled))
    s = np.abs(rhs) if scale is None else scale
    assert_close(lhs, rhs, rtol=F64_RTOL, scale=s, context=context)


def test_toy_table1_as_built_and_rescaled(tmp_path, monkeypatch):
    regridder, url = _toy_artifact(tmp_path, monkeypatch)
    zeta = np.array([1.0, 1.0, -1.0])
    global_scale = np.sum(A_NATIVE * np.abs(zeta))

    built_weight = O_ESMF.sum(axis=1)
    built_value = O_ESMF @ zeta / built_weight
    _assert_tabled(built_value, [F(1), F(-145, 255)], "as built: value")
    _assert_tabled(built_weight, [F(155, 100), F(255, 100)], "as built: weight")
    _assert_tabled(built_weight @ built_value, F(1, 10), "as built: Σ weight·value")

    area = weights.open_areacello(url).values.ravel()
    _assert_tabled(
        _overlap(regridder, url), [[1, F(1, 2), 0], [0, F(1, 2), 2]], "O = o·a/ã", 1.0
    )
    value = _regrid(regridder, zeta)
    _assert_tabled(value, [F(1), F(-3, 5)], "rescaled: value")
    _assert_tabled(area, [F(3, 2), F(5, 2)], "rescaled: weight W' = Σ O")
    _assert_tabled(area @ value, 0, "rescaled: Σ weight·value", scale=global_scale)
    # ρ_J = A_J / Σ_i o_Ji, ρ̄ = Σ a / Σ ã = 4 / 4.1; max at J1: 1 − (1.5/1.55)(4.1/4)
    with xr.open_dataset(os.path.join(url, WEIGHTS_FILENAME)) as ds:
        spread = ds.attrs["a_over_a_tilde_spread"]
    _assert_tabled(spread, F(1, 124), "a/ã spread")


def test_toy_table2_land_cell(tmp_path, monkeypatch):
    regridder, url = _toy_artifact(tmp_path, monkeypatch)
    m = np.array([1.0, 1.0, 0.0])
    zeta = np.array([1.0, -1.0, 0.0])
    global_scale = np.sum(A_NATIVE * m * np.abs(zeta))
    area = weights.open_areacello(url).values.ravel()

    ocean_fraction = _regrid(regridder, m)
    value = _regrid(regridder, m * zeta) / ocean_fraction
    _assert_tabled(value, [F(1, 3), F(-1)], "rescaled value Σ O m ζ / Σ O m")

    mask_weight = area * (ocean_fraction > 0)
    _assert_tabled(mask_weight, [F(3, 2), F(5, 2)], "binary mask weight A·mask")
    _assert_tabled(mask_weight @ value, -2, "A·mask: Σ weight·value")

    W_prime = area * ocean_fraction
    _assert_tabled(W_prime, [F(3, 2), F(1, 2)], "W' = Σ O m")
    _assert_tabled(W_prime @ value, 0, "W': Σ weight·value", scale=global_scale)


def test_toy_table3_weights(tmp_path, monkeypatch):
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    _assert_tabled(
        _overlap(regridder, url),
        [[1, F(1, 2), 0], [0, F(1, 2), 2], [0, 0, 0]],
        "O",
        1.0,
    )
    area = weights.open_areacello(url).values.ravel()
    _assert_tabled(area, [F(3, 2), F(5, 2), 0], "A_J = Σ O", 1.0)
    assert_close(
        area.sum(),
        A_NATIVE.sum(),
        rtol=F64_RTOL,
        scale=A_NATIVE.sum(),
        context="Σ_J A_J = Σ a",
    )
    S, row, col = weights.read_weights(url)
    w_hat = scipy.sparse.coo_matrix((S, (row, col)), shape=(3, 3)).toarray()
    _assert_tabled(
        w_hat,
        [[F(2, 3), F(1, 3), 0], [0, F(1, 5), F(4, 5)], [0, 0, 0]],
        "ŵ = O / A (file rows)",
        1.0,
    )
    assert 2 not in row, "ŵ row J3 is empty in the file"
    assert_close(
        w_hat.T @ area,
        A_NATIVE,
        rtol=F64_RTOL,
        scale=A_NATIVE,
        context="Σ_J ŵ_Ji A_J = a",
    )
    assert np.isnan(_regrid(regridder, np.ones(3))[2]), "raw regrid is NaN on Z"
    assert S.dtype == np.float64
    assert weights.open_areacello(url).dtype == np.float64


def test_artifact_asserts_raise_nan_outside_col0():
    (S, row, col), result = _toy(z_row=True)
    marker = ~np.isfinite(S)
    assert_artifact_invariants((S, row, col), result.w_hat, result.area)
    col_moved = np.where(marker, 1, col)
    with pytest.raises(AssertionError, match="outside source column 0"):
        assert_artifact_invariants((S, row, col_moved), result.w_hat, result.area)


def test_artifact_asserts_raise_negative_w_hat():
    w, result = _toy(z_row=True)
    w_hat = result.w_hat.copy()
    w_hat.data[0] = -w_hat.data[0]
    with pytest.raises(AssertionError, match="ŵ are < 0"):
        assert_artifact_invariants(w, w_hat, result.area)


def test_artifact_asserts_raise_marker_row_mismatch():
    (S, row, col), result = _toy(z_row=True)
    S, row, col = np.append(S, np.nan), np.append(row, 0), np.append(col, 0)
    with pytest.raises(AssertionError, match="NaN-marker rows"):
        assert_artifact_invariants((S, row, col), result.w_hat, result.area)


def test_esmpy_acceptance_raises():
    w, A_e = _toy_weights(z_row=True)
    perturbed = A_TILDE_ESMF.copy()
    perturbed[1] *= 1.0 + 10 * ESMPY_AREA_RTOL
    with pytest.raises(AssertionError, match="ESMPy acceptance"):
        native_area_weights(w, A_e, perturbed, A_NATIVE, radius=1.0)
    within = A_TILDE_ESMF.copy()
    within[1] *= 1.0 + 0.1 * ESMPY_AREA_RTOL
    native_area_weights(w, A_e, within, A_NATIVE, radius=1.0)


def test_assert_close_rule():
    assert_close(3.0, 2.0, rtol=0.5, scale=2.0, context="edge")
    with pytest.raises(AssertionError):
        assert_close(np.nextafter(3.0, 4.0), 2.0, rtol=0.5, scale=2.0, context="past")
    with pytest.raises(AssertionError):
        assert_close(np.nan, 2.0, rtol=0.5, scale=2.0, context="NaN")
    assert_bound(3.0, upper=2.0, rtol=0.5, scale=2.0, context="upper edge")
    assert_bound(1.0, lower=2.0, rtol=0.5, scale=2.0, context="lower edge")
    with pytest.raises(AssertionError):
        assert_bound(
            np.nextafter(3.0, 4.0), upper=2.0, rtol=0.5, scale=2.0, context="upper"
        )
    with pytest.raises(AssertionError):
        assert_bound(
            np.nextafter(1.0, 0.0), lower=2.0, rtol=0.5, scale=2.0, context="lower"
        )


def test_esmf_cell_areas_cover_sphere():
    grid = xe.util.grid_global(30, 20)
    area = esmf_cell_areas(grid)
    assert_close(area.sum(), 4 * np.pi, rtol=F64_RTOL, scale=4 * np.pi, context="Σ A")
    by_lat = area.reshape(grid.lat.shape)
    assert_close(
        by_lat, by_lat[:, :1], rtol=F64_RTOL, scale=by_lat[:, :1], context="zonal"
    )
    assert by_lat[0, 0] < by_lat[by_lat.shape[0] // 2, 0], "row-major (lat, lon)"


def _xesmf_triplets(regridder: xe.Regridder, tmp_path) -> tuple:
    path = os.path.join(tmp_path, "weights.nc")
    regridder.to_netcdf(path)
    ds = xr.open_dataset(path).load()
    return ds.S.values, ds.row.values - 1, ds.col.values - 1


def test_native_area_weights_on_esmf_grids(tmp_path):
    source = xe.util.grid_2d(0, 360, 20, -60, 60, 15)
    target = xe.util.grid_global(30, 20)  # rows south of -60 and north of 60 unmapped
    w = _xesmf_triplets(xe.Regridder(source, target, **REGRIDDER_KWARGS), tmp_path)
    assert not np.isfinite(w[0]).all(), "target rows past the source carry markers"

    a_e = esmf_cell_areas(source)
    rng = np.random.default_rng(0)
    a = MEAN_EARTH_RADIUS_M**2 * a_e * (1 + 0.01 * rng.standard_normal(a_e.size))
    result = native_area_weights(
        w, esmf_cell_areas(target), a_e, a, radius=MEAN_EARTH_RADIUS_M
    )

    assert_close(
        result.w_hat.T @ result.area, a, rtol=F64_RTOL, scale=a, context="Σ_J O_Ji = a"
    )
    assert_close(
        result.area.sum(), a.sum(), rtol=F64_RTOL, scale=a.sum(), context="Σ A = Σ a"
    )
    zero_rows = np.flatnonzero(result.area == 0)
    assert zero_rows.size > 0
    assert all(result.w_hat.getrow(J).nnz == 0 for J in zero_rows)
    assert result.max_rel_residual <= ESMPY_AREA_RTOL
    assert result.a_over_a_tilde_spread > 0
