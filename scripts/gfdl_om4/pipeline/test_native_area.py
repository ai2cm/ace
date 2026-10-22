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
    LEVEL_DIM,
    assert_artifact_invariants,
    assert_bound,
    assert_close,
    assert_native_statics,
    check_extensive_identities,
    check_field_identities,
    check_static_identities,
    esmf_cell_areas,
    level_thickness,
    native_area_statics,
    native_area_weights,
    regrid_full_cell,
    regrid_thickness_weighted,
    z_star_residual,
    z_star_stretch,
    z_star_thickness,
)
from .native_area_weights import write_native_area_artifact
from .ocean_emulators_port import regrid_normalized
from .om4.postprocess import sea_ice
from .postprocess import ChunkContext
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
    area = weights.open_areacello(url, "toy").values.ravel()
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

    area = weights.open_areacello(url, "toy").values.ravel()
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
    area = weights.open_areacello(url, "toy").values.ravel()

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
    area = weights.open_areacello(url, "toy").values.ravel()
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
    assert weights.open_areacello(url, "toy").dtype == np.float64


# Table 3 statics: two levels, interfaces z = (0, 1, 3) m, H = (3, 2, 0) m
# (cell 3 land), so m_k = (1, 1, 0) at both levels and Δz_1 = (2, 1, 0) m.
TOY_INTERFACES = np.array([0.0, 1.0, 3.0])
TOY_DEPTHO = [3.0, 2.0, np.nan]
TOY_SFTOF = [100.0, 100.0, np.nan]


def _native(values, name) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=np.float64)[None, :], dims=("lat", "lon"), name=name
    )


def _toy_wetmask(deptho=TOY_DEPTHO) -> xr.DataArray:
    H = np.nan_to_num(np.asarray(deptho, dtype=np.float64))
    wet = H[None, :] > TOY_INTERFACES[:-1, None]
    centers = 0.5 * (TOY_INTERFACES[:-1] + TOY_INTERFACES[1:])
    return xr.DataArray(
        wet[:, None, :], dims=(LEVEL_DIM, "lat", "lon"), coords={LEVEL_DIM: centers}
    )


def _toy_statics(regridder, url):
    wetmask = _toy_wetmask()
    deptho = _native(TOY_DEPTHO, "deptho")
    dz = level_thickness(deptho, TOY_INTERFACES)
    statics = native_area_statics(
        regridder,
        weights.open_areacello(url, "toy"),
        wetmask,
        dz,
        _native(TOY_SFTOF, "sftof"),
    )
    regridded, _ = regrid_normalized(
        deptho.to_dataset(), regridder, wetmask.isel({LEVEL_DIM: 0}, drop=True)
    )
    return statics, wetmask, deptho, regridded["deptho"]


def _check_toy_statics(statics, regridder, wetmask, deptho, deptho_J):
    check_static_identities(
        statics, regridder, wetmask, _native(A_NATIVE, "areacello"), deptho, deptho_J
    )


def test_toy_table3_statics(tmp_path, monkeypatch):
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    statics, wetmask, deptho, deptho_J = _toy_statics(regridder, url)
    stored = statics.stored

    _assert_tabled(
        level_thickness(deptho, TOY_INTERFACES).values[:, 0, :],
        [[1, 1, 0], [2, 1, 0]],
        "Δz_ik",
        1.0,
    )
    m0 = wetmask.isel({LEVEL_DIM: 0}, drop=True)
    fraction = _regrid(regridder, m0.values[0])
    _assert_tabled(fraction[:2], [1, F(1, 5)], "regridder(m_0)")
    assert np.isnan(fraction[2]), "regridder(m_0) is NaN on Z"
    _assert_tabled(
        np.nan_to_num(fraction) * 100, [100, 20, 0], "regridder(m_0).fillna(0) × 100"
    )
    _assert_tabled(statics.ocean_area.values.ravel(), [F(3, 2), F(1, 2), 0], "W'_J0", 1)
    _assert_tabled(stored["sftof"].values.ravel(), [100, 20, 0], "sftof_J", 100)
    _assert_tabled(
        stored["sea_surface_fraction"].values.ravel(), [1, F(1, 5), 0], "ssf_J", 1
    )
    _assert_tabled(
        statics.ocean_volume.values[:, 0, :],
        [[F(3, 2), F(1, 2), 0], [F(5, 2), F(1, 2), 0]],
        "V'_J0, V'_J1",
        1.0,
    )
    for k, h_prime in enumerate([[1, 1, 0], [F(5, 3), 1, 0]]):
        _assert_tabled(stored[f"mask_{k}"].values.ravel(), [1, 1, 0], f"mask_J{k}", 1)
        _assert_tabled(stored[f"h_prime_{k}"].values.ravel(), h_prime, f"h'_J{k}", 1)
    _assert_tabled(stored["areacello"].values.ravel(), [F(3, 2), F(5, 2), 0], "A_J", 1)
    _assert_tabled(deptho_J.values.ravel()[:2], [F(8, 3), 2], "deptho_J", 1.0)
    assert np.isnan(deptho_J.values.ravel()[2]), "deptho_J is NaN on Z"
    for name in stored.data_vars:
        assert stored[name].dtype == np.float64, name
        assert stored[name].values.ravel()[2] == 0, f"{name} is 0 on Z"
    _check_toy_statics(statics, regridder, wetmask, deptho, deptho_J)


# Table 3 fields: x_k = (1, −1, ·) at both levels, ζ = (1, −1, ·), sea ice
# c = (1, 1/2, ·), HI = (2, 1, ·) m.
TOY_X = [1.0, -1.0, np.nan]
TOY_ZETA = [1.0, -1.0, np.nan]
TOY_C = [1.0, 0.5, np.nan]
TOY_HI = [2.0, 1.0, np.nan]


def _level(da: xr.DataArray, k: int) -> xr.DataArray:
    return da.isel({LEVEL_DIM: k}, drop=True)


def _toy_fields(regridder, url):
    """Every table-3 field through the shipped regrid functions, float64,
    with what the identities need. Native inputs are float32, as many
    native stream variables are."""
    wetmask = _toy_wetmask()
    dz = level_thickness(_native(TOY_DEPTHO, "deptho"), TOY_INTERFACES)
    m0 = _level(wetmask, 0)

    def native(**values):
        return xr.Dataset(
            {n: _native(v, n).astype(np.float32) for n, v in values.items()}
        )

    x = native(x=TOY_X)
    per_volume = [
        regrid_thickness_weighted(x, regridder, _level(wetmask, k), _level(dz, k))
        for k in range(2)
    ]
    surface = native(zeta=TOY_ZETA, sea_ice_fraction=TOY_C, HI=TOY_HI)
    per_ocean_area, ocean_fraction = regrid_normalized(surface, regridder, m0)
    full_cell = xr.Dataset(
        {
            n: regrid_full_cell(regridder, surface[n], m0, ocean_fraction)
            for n in ("zeta", "sea_ice_fraction")
        }
    )
    ice_product = (surface["HI"] * surface["sea_ice_fraction"]).astype(np.float64)
    return {
        "wetmask": wetmask,
        "dz": dz,
        "x": x.astype(np.float64),
        "surface": surface.astype(np.float64),
        "per_volume": per_volume,
        "per_ocean_area": per_ocean_area,
        "ocean_fraction": ocean_fraction,
        "full_cell": full_cell,
        "ice_product": ice_product,
        "ice_volume": regrid_full_cell(regridder, ice_product, m0, ocean_fraction),
        "area": weights.open_areacello(url, "toy"),
    }


def _assert_row(da, tabled, context):
    """Tabled values on J1, J2; NaN on J3 (Z)."""
    values = da.values.ravel()
    assert da.dtype == np.float64, context
    _assert_tabled(values[:2], tabled, context, 1.0)
    assert np.isnan(values[2]), f"{context}: NaN on Z"


def test_toy_table3_fields(tmp_path, monkeypatch):
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    f = _toy_fields(regridder, url)
    for k, tabled in enumerate([[F(1, 3), -1], [F(3, 5), -1]]):
        _assert_row(f["per_volume"][k][0]["x"], tabled, f"X^tw_J{k}")
    _assert_row(f["per_ocean_area"]["zeta"], [F(1, 3), -1], "X^a_J of ζ")
    _assert_row(f["full_cell"]["zeta"], [F(1, 3), F(-1, 5)], "X_J of ζ (full cell)")

    x_a, _ = regrid_normalized(f["x"], regridder, _level(f["wetmask"], 1))
    _assert_row(x_a["x"], [F(1, 3), -1], "X^a_J1 (not stored)")
    x_tw = f["per_volume"][1][0]["x"].values.ravel()
    assert abs(x_a["x"].values.ravel()[0] - x_tw[0]) > 0.1, "X^a_J1 ≠ X^tw_J1 on J1"


def _toy_sea_ice(f):
    output = xr.Dataset(
        {
            "sea_ice_fraction": f["full_cell"]["sea_ice_fraction"],
            "ocean_sea_ice_fraction": f["per_ocean_area"]["sea_ice_fraction"],
            "HI": f["per_ocean_area"]["HI"],
            "UI": 0.0 * f["per_ocean_area"]["HI"],
            "VI": 0.0 * f["per_ocean_area"]["HI"],
        }
    )
    context = ChunkContext(
        ocean_fraction=f["ocean_fraction"],
        store="toy",
        areacello=f["area"],
        ice_volume_per_cell_area=f["ice_volume"],
    )
    return sea_ice(
        output,
        context,
        sea_ice_fraction="sea_ice_fraction",
        ocean_sea_ice_fraction="ocean_sea_ice_fraction",
    )


def test_toy_table3_sea_ice(tmp_path, monkeypatch):
    f = _toy_fields(*_toy_artifact(tmp_path, monkeypatch, z_row=True))
    out = _toy_sea_ice(f)
    _assert_row(out["sea_ice_fraction"], [F(5, 6), F(1, 10)], "sea_ice_fraction_J")
    _assert_row(f["ice_volume"], [F(3, 2), F(1, 10)], "(HI c)_J")
    _assert_row(out["sea_ice_volume"], [F(9, 4), F(1, 4)], "sea_ice_volume_J")
    _assert_row(out["HI"], [F(5, 3), 1], "stored HI")
    ratio = out["sea_ice_volume"] / (f["area"] * out["sea_ice_fraction"])
    _assert_row(ratio, [F(9, 5), 1], "sea_ice_volume_J / (A_J sea_ice_fraction_J)")
    assert abs(ratio.values.ravel()[0] - out["HI"].values.ravel()[0]) > 0.1


def _check_toy_fields(regridder, f, per_volume_1=None, full_cell=None) -> None:
    a = _native(A_NATIVE, "areacello")
    for k in range(2):
        out, fraction = f["per_volume"][k]
        check_field_identities(
            f["x"],
            regridder,
            _level(f["wetmask"], k),
            f["area"],
            fraction,
            a,
            f"level {k}",
            per_volume=per_volume_1 if k == 1 and per_volume_1 is not None else out,
            dz=_level(f["dz"], k),
        )
    check_field_identities(
        f["surface"],
        regridder,
        _level(f["wetmask"], 0),
        f["area"],
        f["ocean_fraction"],
        a,
        "surface",
        per_ocean_area=f["per_ocean_area"],
        full_cell=f["full_cell"] if full_cell is None else full_cell,
    )


def _check_toy_extensive(f, output) -> None:
    check_extensive_identities(
        output,
        {
            "hfds_total_area": f["surface"]["zeta"],
            "sea_ice_volume": f["ice_product"],
        },
        _level(f["wetmask"], 0),
        f["area"],
        f["ocean_fraction"],
        _native(A_NATIVE, "areacello"),
        "toy",
    )


def _toy_extensive_output(f) -> xr.Dataset:
    out = _toy_sea_ice(f)
    out["hfds_total_area"] = f["per_ocean_area"]["zeta"] * f["ocean_fraction"]
    return out


def test_toy_table3_identities(tmp_path, monkeypatch):
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    statics, wetmask, deptho, deptho_J = _toy_statics(regridder, url)
    _check_toy_statics(statics, regridder, wetmask, deptho, deptho_J)
    f = _toy_fields(regridder, url)
    _check_toy_fields(regridder, f)
    _check_toy_extensive(f, _toy_extensive_output(f))


def test_toy_identity_catches_mask_only_3d(tmp_path, monkeypatch):
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    f = _toy_fields(regridder, url)
    mask_only, _ = regrid_normalized(f["x"], regridder, _level(f["wetmask"], 1))
    with pytest.raises(AssertionError, match="level 1: Σ_J V'_Jk X\\^tw_Jk"):
        _check_toy_fields(regridder, f, per_volume_1=mask_only)


@pytest.mark.parametrize(
    "name, values, match",
    [
        ("zeta", [F(1, 3), F(-3, 10)], "Σ_J A_J X_J = Σ_i a_i m_i0 X_i, zeta"),
        # Σ_J A_J X_J kept at 7/5: only the bound can catch it.
        ("sea_ice_fraction", [F(3, 2), F(-3, 10)], "sea_ice_fraction_J ≤ 1"),
    ],
)
def test_field_identities_raise_on_full_cell(
    tmp_path, monkeypatch, name, values, match
):
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    f = _toy_fields(regridder, url)
    full_cell = f["full_cell"].copy(deep=True)
    for j, value in enumerate(values):
        _set(full_cell[name], j, float(value))
    with pytest.raises(AssertionError, match=match):
        _check_toy_fields(regridder, f, full_cell=full_cell)


def test_full_cell_nan_off_mask(tmp_path, monkeypatch):
    """m = (1, 0, 0): J2 overlaps only land, so mask_J2 = 0 and X_J2 is NaN."""
    regridder, _ = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    m = _native([1.0, 0.0, 0.0], "m") > 0
    fraction = regridder(m.astype(np.float64)).fillna(0.0)
    full = regrid_full_cell(regridder, _native(TOY_ZETA, "zeta"), m, fraction)
    _assert_tabled(full.values.ravel()[:1], [F(2, 3)], "X_J1", 1.0)
    assert np.isnan(full.values.ravel()[1:]).all(), "NaN where mask_J0 = 0 and on Z"


def test_field_identities_raise_on_native_coords_off_wetmask(tmp_path, monkeypatch):
    """Native a_i on other lon labels than the wetmask raise, not inner-join."""
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    f = _toy_fields(regridder, url)
    lon = {"lon": [0.0, 1.0, 2.0]}
    with pytest.raises(ValueError, match="exact"):
        check_field_identities(
            f["surface"].assign_coords(lon),
            regridder,
            _level(f["wetmask"], 0).assign_coords(lon),
            f["area"],
            f["ocean_fraction"],
            _native(A_NATIVE, "areacello").assign_coords(lon=[0.0, 1.0, 2.5]),
            "surface",
            per_ocean_area=f["per_ocean_area"],
        )


def test_extensive_identity_catches_hi_area_fraction(tmp_path, monkeypatch):
    """Today's sea_ice_volume = HI · A_J · sea_ice_fraction_J misses 5/2."""
    f = _toy_fields(*_toy_artifact(tmp_path, monkeypatch, z_row=True))
    output = _toy_extensive_output(f)
    output["sea_ice_volume"] = output["HI"] * f["area"] * output["sea_ice_fraction"]
    with pytest.raises(AssertionError, match="sea_ice_volume"):
        _check_toy_extensive(f, output)


def test_extensive_identity_raises_on_hfds_total_area(tmp_path, monkeypatch):
    f = _toy_fields(*_toy_artifact(tmp_path, monkeypatch, z_row=True))
    output = _toy_extensive_output(f)
    output["hfds_total_area"] = f["per_ocean_area"]["zeta"]
    with pytest.raises(AssertionError, match="hfds_total_area"):
        _check_toy_extensive(f, output)


def _perturb_areacello_coords(url, ulps: int, factor: float = 1.0) -> None:
    """Rewrite the artifact's ``areacello.nc`` with ``lat``, ``lon`` moved
    ``ulps`` ulp up, then scaled by ``factor`` (the F90 artifact's lat)."""
    path = os.path.join(url, weights.AREACELLO_FILENAME)
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    for name in ("lat", "lon"):
        x = ds[name].values
        for _ in range(ulps):
            x = np.nextafter(x, np.inf)
        ds = ds.assign_coords({name: x * factor})
    ds.to_netcdf(path)


def test_open_areacello_on_regridder_coords(tmp_path, monkeypatch):
    """A_J stored with coords a few ulp off keeps every row of A_J · regridder(x)."""
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    _perturb_areacello_coords(url, ulps=4)
    area = weights.open_areacello(url, "toy")
    fraction = regridder(_toy_wetmask().isel({LEVEL_DIM: 0}, drop=True) * 1.0)
    assert (area * fraction).sizes == {"lat": 1, "lon": 3}
    xr.align(area, fraction, join="exact")


def test_open_areacello_raises_off_target_grid(tmp_path, monkeypatch):
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    _perturb_areacello_coords(url, ulps=0, factor=1.001)
    with pytest.raises(AssertionError, match="lon vs target grid"):
        weights.open_areacello(url, "toy")


def test_native_area_statics_raises_on_area_off_grid(tmp_path, monkeypatch):
    """An A_J off the regridder-output coords raises instead of dropping rows."""
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    area = weights.open_areacello(url, "toy")
    area = area.assign_coords(lat=np.nextafter(area.lat.values, np.inf))
    with pytest.raises(ValueError, match="exact"):
        native_area_statics(
            regridder,
            area,
            _toy_wetmask(),
            level_thickness(_native(TOY_DEPTHO, "deptho"), TOY_INTERFACES),
            _native(TOY_SFTOF, "sftof"),
        )


@pytest.mark.parametrize(
    "deptho, areacello, sftof, match",
    [
        ([3.5, 2.0, np.nan], A_NATIVE, TOY_SFTOF, "bottom interface"),
        (TOY_DEPTHO, [1.0, 0.0, 2.0], TOY_SFTOF, "areacello_i ≤ 0"),
        (TOY_DEPTHO, A_NATIVE, [100.0, 50.0, np.nan], "sftof_i = 100 m_i0"),
        ([3.0, 1.0, np.nan], A_NATIVE, TOY_SFTOF, "wetmask ≠ 1\\[deptho > z_k\\]"),
    ],
)
def test_native_statics_asserts_raise(deptho, areacello, sftof, match):
    wetmask = _toy_wetmask()
    assert_native_statics(
        wetmask,
        _native(TOY_DEPTHO, "deptho"),
        TOY_INTERFACES,
        _native(A_NATIVE, "areacello"),
        _native(TOY_SFTOF, "sftof"),
    )
    with pytest.raises(AssertionError, match=match):
        assert_native_statics(
            wetmask,
            _native(deptho, "deptho"),
            TOY_INTERFACES,
            _native(areacello, "areacello"),
            _native(sftof, "sftof"),
        )


def _set(da: xr.DataArray, j: int, value: float) -> None:
    da.values[..., j] = value


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (lambda s, d: _set(s.stored["h_prime_1"], 0, np.nan), "non-finite"),
        (lambda s, d: _set(s.stored["mask_0"], 2, 1.0), "Z cells have mask_J0 = 1"),
        (lambda s, d: _set(s.stored["mask_1"], 1, 0.0), "mask_1"),
        (lambda s, d: _set(d, 0, 2.5), "Σ_k V'_Jk = W'_J0 deptho_J"),
        (lambda s, d: _set(s.stored["areacello"], 0, 1.4), "Σ_J A_J = Σ_i a_i"),
        (lambda s, d: _set(s.stored["sftof"], 0, 99.0), "sftof_J = 100"),
        (lambda s, d: _set(s.stored["sftof"], 1, -1.0), "lower bound"),
        (
            lambda s, d: _set(s.stored["sea_surface_fraction"], 1, 0.3),
            "sea_surface_fraction_J",
        ),
    ],
)
def test_static_identities_raise(tmp_path, monkeypatch, corrupt, match):
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    statics, wetmask, deptho, deptho_J = _toy_statics(regridder, url)
    corrupt(statics, deptho_J)
    with pytest.raises(AssertionError, match=match):
        _check_toy_statics(statics, regridder, wetmask, deptho, deptho_J)


def test_static_identities_raise_on_w_hat_column_sum(tmp_path, monkeypatch):
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    statics, wetmask, deptho, deptho_J = _toy_statics(regridder, url)
    a_other = _native([1.0, 1.1, 1.9], "areacello")
    with pytest.raises(AssertionError, match="Σ_J ŵ_Ji A_J = a_i"):
        check_static_identities(statics, regridder, wetmask, a_other, deptho, deptho_J)


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


# z* (PRD ### z* thickness) on table 3: thetao x = (1, −1, ·), so = 35 at
# both levels. η = (3/10, 0, ·) m: η/H varies within J1.
TOY_SSH = [0.3, 0.0, np.nan]
TOY_SO = [35.0, 35.0, np.nan]


def _toy_z_star(regridder, url, ssh=TOY_SSH):
    wetmask = _toy_wetmask()
    deptho = _native(TOY_DEPTHO, "deptho")
    level = np.broadcast_to(np.asarray([TOY_X, TOY_SO])[:, None, None, :], (2, 2, 1, 3))
    fields = xr.Dataset(
        {
            name: ((LEVEL_DIM, "lat", "lon"), level[n])
            for n, name in enumerate(["thetao", "so"])
        },
        coords={LEVEL_DIM: wetmask[LEVEL_DIM]},
    )
    return {
        "regridder": regridder,
        "area": weights.open_areacello(url, "toy"),
        "wetmask": wetmask,
        "dz": level_thickness(deptho, TOY_INTERFACES),
        "native_areacello": _native(A_NATIVE, "areacello"),
        "deptho": deptho,
        "ssh": _native(ssh, "SSH"),
        "fields": fields,
    }


def test_z_star_residual_toy(tmp_path, monkeypatch):
    """h_0 = (11/10, 1), h_1 = (11/5, 1); σ_J = (43/40, 1); per-level
    native volumes (8/5, 1/2), (27/10, 1/2); V'_Jk = (3/2, 1/2), (5/2, 1/2)."""
    z = _toy_z_star(*_toy_artifact(tmp_path, monkeypatch, z_row=True))
    report = z_star_residual(**z)
    vol = {"sigma_J": [F(1, 128), F(1, 216)], "sigma_1": [F(1, 16), F(2, 27)]}
    cont = {
        "sigma_J": {"thetao": [F(5, 168), F(7, 256)], "so": [F(1, 168), F(1, 256)]},
        "sigma_1": {"thetao": [F(1, 21), F(1, 16)], "so": [F(1, 21), F(1, 16)]},
    }
    for key, tabled in vol.items():
        _assert_tabled(report[key]["vol"], tabled, f"{key} vol_k")
        for name, values in cont[key].items():
            _assert_tabled(report[key]["cont"][name], values, f"{key} cont_k {name}")


def test_z_star_column_volume_exact(tmp_path, monkeypatch):
    """Σ_k σ_J V'_Jk = Σ_k Σ_i O_Ji m_ik h_ik on mask_J0 = 1 (σ_J from the
    stored SSH, deptho), though vol_k ≠ 0 per level."""
    z = _toy_z_star(*_toy_artifact(tmp_path, monkeypatch, z_row=True))
    regridder, area, m = z["regridder"], z["area"], z["wetmask"]
    sigma = z_star_stretch(regridder, m, z["deptho"], z["ssh"])
    _assert_tabled(sigma.values.ravel(), [F(43, 40), 1, 1], "σ_J", 1.0)
    volume = area * _regrid_levels(regridder, (m * z["dz"]).fillna(0.0))
    h = z_star_thickness(z["dz"], m, z["deptho"], z["ssh"])
    native_volume = area * _regrid_levels(regridder, h)
    _assert_tabled(
        (sigma * volume).sum(LEVEL_DIM).values.ravel()[:2],
        native_volume.sum(LEVEL_DIM).values.ravel()[:2],
        "Σ_k σ_J V'_Jk = Σ_k Σ_i O_Ji m_ik h_ik",
    )


def _regrid_levels(regridder, x: xr.DataArray) -> xr.DataArray:
    return regridder(x.astype(np.float64), keep_attrs=False)


@pytest.mark.parametrize("r", [F(1, 10), F(-1, 20)])
def test_z_star_residual_uniform_ratio(tmp_path, monkeypatch, r):
    """η_i = r H_i: σ_J removes the residual; σ_J = 1 leaves vol_k = |r|/(1+r)
    and cont_1 of thetao = |r|/(1+r) · |Σ a m_1 Δz_1 x| / Σ a m_1 Δz_1 |x|."""
    ssh = [float(r) * h for h in TOY_DEPTHO]
    regridder, url = _toy_artifact(tmp_path, monkeypatch, z_row=True)
    z = _toy_z_star(regridder, url, ssh=ssh)
    report = z_star_residual(**z)
    zero = np.zeros(2)
    for name in ("thetao", "so"):
        assert_close(
            report["sigma_J"]["cont"][name],
            zero,
            rtol=F64_RTOL,
            scale=1.0,
            context=f"σ_J cont_k {name}",
        )
    assert_close(
        report["sigma_J"]["vol"], zero, rtol=F64_RTOL, scale=1.0, context="σ_J vol_k"
    )
    stretch = abs(r) / (1 + r)
    _assert_tabled(report["sigma_1"]["vol"], [stretch, stretch], "σ = 1 vol_k")
    _assert_tabled(
        report["sigma_1"]["cont"]["thetao"], [0, stretch / 3], "σ = 1 cont_k", 1.0
    )
