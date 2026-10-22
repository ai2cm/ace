"""Conservative regrid weights rebuilt in native (MOM6) cell areas.

i native cell, J target cell; areas in sr are on the unit sphere, R the
sphere radius:

    w_Ji          stored ESMF conservative weight, NaN entries dropped
    A^esmf_J      target cell area ESMF divided overlaps by (sr)
    ã^esmf_i      ESMF's own source cell area (sr)
    a_i           native areacello (m²)
    ã_i  = R² Σ_J w_Ji A^esmf_J                       native area the weights imply (m²)
    O_Ji = R² w_Ji A^esmf_J a_i / ã_i                 overlap in native areas (m²)
    A_J  = Σ_i O_Ji                                   0 on Z, rows ESMF left unmapped
    ŵ_Ji = O_Ji / A_J                                 published weights; empty rows on Z

so Σ_J O_Ji = a_i. xESMF marks each unmapped target row with one NaN entry
at source column 0 (``unmapped_to_nan``); those entries are dropped.

Every check uses one pass rule, ``|lhs − rhs| ≤ rtol · S``, never ``==``.
"""

import dataclasses
from typing import Mapping

import esmpy
import numpy as np
import scipy.sparse
import xarray as xr
from xesmf.frontend import ds_to_ESMFgrid

from .ocean_emulators_port import OCEAN_FRACTION_THRESHOLD, regrid_normalized
from .postprocess import SEA_ICE_FRACTION

# analysis/2026-09-30-1956-native-area-regrid/05-esmpy-areas.yaml::acceptance_rtol
ESMPY_AREA_RTOL = 1e-9
# analysis/2026-10-05-2201-ra-hemispheric-closure/04_ice_area_ladder.py::F64_RTOL
F64_RTOL = 1e-12
# analysis/2026-10-05-2201-ra-hemispheric-closure/04_ice_area_ladder.py::AREA_RTOL
AREA_RTOL = 1e-6

ARTIFACT_ATTRS = (
    "source_weights_url",
    "a_source_url",
    "esmpy_area_rtol",
    "esmpy_max_rel_residual",
    "a_over_a_tilde_spread",
)

# xESMF-format triplets (S, row, col), 0-based, NaN markers included.
Triplets = tuple[np.ndarray, np.ndarray, np.ndarray]


def _fail(context: str, failed: np.ndarray, excess: np.ndarray) -> None:
    n_failed = int(failed.sum())
    if n_failed:
        worst = (
            np.nanmax(excess[failed]) if np.isfinite(excess[failed]).any() else np.nan
        )
        raise AssertionError(
            f"{context}: {n_failed} of {failed.size} values fail the rule "
            f"(max excess over rtol·S: {worst}; NaN counts as a failure)"
        )


def assert_close(lhs, rhs, *, rtol: float, scale, context: str) -> None:
    """Raise unless ``|lhs − rhs| ≤ rtol · scale`` everywhere."""
    lhs, rhs, scale = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (lhs, rhs, scale))
    )
    excess = np.abs(lhs - rhs) - rtol * scale
    _fail(context, ~(excess <= 0), excess)


def assert_bound(
    lhs, *, lower=None, upper=None, rtol: float, scale, context: str
) -> None:
    """Raise unless ``lower − rtol·scale ≤ lhs ≤ upper + rtol·scale``."""
    lhs = np.asarray(lhs, dtype=np.float64)
    tol = rtol * np.asarray(scale, dtype=np.float64)
    if upper is not None:
        excess = np.broadcast_to(lhs - upper - tol, lhs.shape)
        _fail(f"{context} (upper bound {upper})", ~(excess <= 0), excess)
    if lower is not None:
        excess = np.broadcast_to(lower - tol - lhs, lhs.shape)
        _fail(f"{context} (lower bound {lower})", ~(excess <= 0), excess)


def esmf_cell_areas(grid: xr.Dataset) -> np.ndarray:
    """ESMF's own cell areas (sr) of ``grid`` (``lon``, ``lat``, ``lon_b``,
    ``lat_b``), raveled in the cell order of xESMF weights.

    ``periodic=False``, as xESMF builds conservative grids."""
    esmf_grid, shape, _ = ds_to_ESMFgrid(grid, need_bounds=True, periodic=False)
    field = esmpy.Field(esmf_grid, staggerloc=esmpy.StaggerLoc.CENTER)
    field.get_area()
    return np.asarray(field.data).T.reshape(shape).astype(np.float64).ravel()


@dataclasses.dataclass
class NativeAreaWeights:
    w_hat: scipy.sparse.csr_matrix  # ŵ_Ji, (target, source), no entries on Z rows
    area: np.ndarray  # A_J (m²), 0 on Z
    max_rel_residual: float  # max_i |Σ_J w_Ji A^esmf_J − ã^esmf_i| / ã^esmf_i
    a_over_a_tilde_spread: float  # max_{J ∉ Z} |ρ_J / ρ̄ − 1|


def _finite_matrix(w: Triplets, shape: tuple[int, int]) -> scipy.sparse.csr_matrix:
    S, row, col = (np.asarray(x) for x in w)
    finite = np.isfinite(S)
    return scipy.sparse.coo_matrix(
        (S[finite].astype(np.float64), (row[finite], col[finite])), shape=shape
    ).tocsr()


def native_area_weights(
    w: Triplets,
    target_area_esmf: np.ndarray,
    source_area_esmf: np.ndarray,
    a: np.ndarray,
    radius: float,
) -> NativeAreaWeights:
    """Rebuild ESMF weights ``w`` in native areas ``a`` (m²).

    Raises if ESMF's implied source areas miss ``source_area_esmf`` by more
    than ``ESMPY_AREA_RTOL``, or if any artifact invariant fails."""
    A_e = np.asarray(target_area_esmf, dtype=np.float64).ravel()
    a_e = np.asarray(source_area_esmf, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.shape != a_e.shape:
        raise ValueError(f"a has {a.size} cells, source grid has {a_e.size}")
    W = _finite_matrix(w, (A_e.size, a_e.size))

    implied = W.T @ A_e  # Σ_J w_Ji A^esmf_J (sr)
    assert_close(
        implied,
        a_e,
        rtol=ESMPY_AREA_RTOL,
        scale=a_e,
        context="ESMPy acceptance Σ_J w_Ji A^esmf_J = ã^esmf_i",
    )
    max_rel_residual = float(np.max(np.abs(implied - a_e) / a_e))

    a_tilde = radius**2 * implied
    scale_i = np.divide(a, a_tilde, out=np.zeros_like(a), where=a_tilde > 0)
    overlap = (
        scipy.sparse.diags(radius**2 * A_e) @ W @ scipy.sparse.diags(scale_i)
    ).tocsr()
    overlap.eliminate_zeros()
    area = np.asarray(overlap.sum(axis=1)).ravel()
    inv_area = np.divide(1.0, area, out=np.zeros_like(area), where=area > 0)
    w_hat = (scipy.sparse.diags(inv_area) @ overlap).tocsr()
    w_hat.eliminate_zeros()

    assert_artifact_invariants(w, w_hat, area)

    mapped = area > 0
    esmf_overlap = A_e * np.asarray(W.sum(axis=1)).ravel()  # Σ_i w_Ji A^esmf_J
    rho = area[mapped] / (radius**2 * esmf_overlap[mapped])
    rho_bar = a.sum() / a_tilde.sum()
    spread = float(np.max(np.abs(rho / rho_bar - 1.0)))
    return NativeAreaWeights(w_hat, area, max_rel_residual, spread)


def _support(m: scipy.sparse.spmatrix) -> set[tuple[int, int]]:
    m = m.tocoo()
    nonzero = m.data != 0
    return set(zip(m.row[nonzero].tolist(), m.col[nonzero].tolist()))


def assert_artifact_invariants(
    w: Triplets, w_hat: scipy.sparse.spmatrix, area: np.ndarray
) -> None:
    """Raise unless the artifact-build asserts hold on source ``w`` and the
    new ``ŵ``, ``A_J``:

        {i : ã_i = 0} = ∅;  D ⊆ {0};  {J : row J of w holds a NaN} = {J : A_J = 0};
        ŵ ≥ 0;  support(ŵ) = support(finite w)
    """
    S, row, col = (np.asarray(x) for x in w)
    area = np.asarray(area, dtype=np.float64).ravel()
    W = _finite_matrix(w, w_hat.shape)
    nan = ~np.isfinite(S)

    # With A^esmf_J > 0 and w ≥ 0, ã_i = 0 exactly where column i holds no nonzero w.
    W_coo = W.tocoo()
    n_mapped_source = np.unique(W_coo.col[W_coo.data != 0]).size
    n_unmapped_source = W.shape[1] - n_mapped_source
    if n_unmapped_source:
        raise AssertionError(f"{n_unmapped_source} source cells have ã_i = 0")

    D = set(np.unique(col[nan]).tolist())
    if not D <= {0}:
        raise AssertionError(
            f"NaN weights outside source column 0: columns {sorted(D)}"
        )

    marker_rows = set(np.unique(row[nan]).tolist())
    zero_rows = set(np.flatnonzero(area == 0).tolist())
    if marker_rows != zero_rows:
        raise AssertionError(
            f"NaN-marker rows ≠ {{J : A_J = 0}}: "
            f"{len(marker_rows - zero_rows)} marker rows with A_J ≠ 0, "
            f"{len(zero_rows - marker_rows)} rows with A_J = 0 and no marker"
        )

    w_hat_data = w_hat.tocoo().data
    if not (w_hat_data >= 0).all():
        raise AssertionError(
            f"{int((~(w_hat_data >= 0)).sum())} entries of ŵ are < 0 or NaN"
        )

    if _support(w_hat) != _support(W):
        raise AssertionError("support(ŵ) ≠ support(finite w)")


# ---------------------------------------------------------------------------
# Statics (step 2): with ŵ loaded, Σ_i O_Ji y_i = A_J · regridder(y)_J
# ---------------------------------------------------------------------------

LEVEL_DIM = "z_l"


def level_thickness(deptho: xr.DataArray, interfaces: np.ndarray) -> xr.DataArray:
    """Δz_ik = clip(H_i, z_k, z_{k+1}) − z_k (m), partial bottom cells; H NaN → 0."""
    H = deptho.fillna(0.0).astype(np.float64)
    top = xr.DataArray(np.asarray(interfaces[:-1], dtype=np.float64), dims=LEVEL_DIM)
    bottom = xr.DataArray(np.asarray(interfaces[1:], dtype=np.float64), dims=LEVEL_DIM)
    return (H.clip(top, bottom) - top).transpose(LEVEL_DIM, ...)


@dataclasses.dataclass
class NativeAreaStatics:
    stored: (
        xr.Dataset
    )  # areacello, sftof, sea_surface_fraction, mask_<k>[, h_prime_<k>]
    ocean_area: xr.DataArray  # W'_J0 (m²), 0 on land and Z
    ocean_volume: xr.DataArray | None = (
        None  # V'_Jk (m³), (z_l, *target dims), 0 off mask
    )


def _regrid(regridder, x: xr.DataArray) -> xr.DataArray:
    return regridder(x.astype(np.float64), keep_attrs=False)


def native_area_statics(
    regridder,
    area: xr.DataArray,
    wetmask: xr.DataArray,
    dz: xr.DataArray,
    sftof_native: xr.DataArray,
) -> NativeAreaStatics:
    """Step-2 statics in float64 from ŵ (``regridder``) and A_J (``area``):

        W'_J0 = A_J regridder(m_0)          V'_Jk = A_J regridder(m_k Δz_k)
        sftof_J = regridder(sftof_i m_i0)   sea_surface_fraction_J = sftof_J / 100
        mask_Jk = 1[V'_Jk > 0]              h'_Jk = V'_Jk / W'_J0 on mask, else 0

    each 0 on land and on Z (``fillna(0)``)."""
    m = wetmask.astype(np.float64)
    m0 = wetmask.isel({LEVEL_DIM: 0}, drop=True)
    surface_fraction = _regrid(regridder, m.isel({LEVEL_DIM: 0}, drop=True))
    # A_J must carry regridder-output coords (weights.open_areacello);
    # an inner join would drop rows.
    area, _ = xr.align(area.astype(np.float64), surface_fraction, join="exact")
    ocean_area = (area * surface_fraction).fillna(0.0)
    ocean_volume = (area * _regrid(regridder, (m * dz).fillna(0.0))).fillna(0.0)
    ocean_volume = ocean_volume.transpose(LEVEL_DIM, ...)
    sftof = _regrid(regridder, sftof_native.where(m0).fillna(0.0)).fillna(0.0)
    mask = ocean_volume > 0
    h_prime = xr.where(mask, ocean_volume / ocean_area.where(ocean_area > 0), 0.0)

    stored = xr.Dataset(
        {
            "areacello": area.fillna(0.0),
            "sftof": sftof,
            "sea_surface_fraction": sftof / 100.0,
        }
    )
    for k in range(wetmask.sizes[LEVEL_DIM]):
        level = {LEVEL_DIM: k}
        stored[f"mask_{k}"] = mask.isel(level, drop=True).astype(np.float64)
        stored[f"h_prime_{k}"] = h_prime.isel(level, drop=True)
    return NativeAreaStatics(stored, ocean_area, ocean_volume)


def assert_native_statics(
    wetmask: xr.DataArray,
    deptho: xr.DataArray,
    interfaces: np.ndarray,
    areacello: xr.DataArray,
    sftof: xr.DataArray,
) -> None:
    """Per-store asserts on the native inputs (step 5):

    m_ik = 1[H_i > z_k];  max_i H_i ≤ z_bottom;  assert_native_area_statics
    """
    H = deptho.fillna(0.0).astype(np.float64)
    top = xr.DataArray(np.asarray(interfaces[:-1], dtype=np.float64), dims=LEVEL_DIM)
    mismatch = (wetmask != (H > top)).sum([d for d in wetmask.dims if d != LEVEL_DIM])
    if mismatch.any():
        raise AssertionError(
            "wetmask ≠ 1[deptho > z_k]; mismatches per level: "
            f"{mismatch.values.tolist()}"
        )
    bottom = float(interfaces[-1])
    assert_bound(
        float(H.max()),
        upper=bottom,
        rtol=F64_RTOL,
        scale=bottom,
        context="max_i deptho_i ≤ bottom interface",
    )
    assert_native_area_statics(
        wetmask.isel({LEVEL_DIM: 0}, drop=True), areacello, sftof
    )


def assert_native_area_statics(
    m0: xr.DataArray, areacello: xr.DataArray, sftof: xr.DataArray
) -> None:
    """Per-store asserts on native areas: a_i > 0;  sftof_i = 100 m_i0"""
    n_bad_area = int((~(areacello > 0)).sum())
    if n_bad_area:
        raise AssertionError(f"{n_bad_area} native areacello_i ≤ 0 or NaN")
    assert_close(
        sftof.fillna(0.0).astype(np.float64),
        100.0 * m0.astype(np.float64),
        rtol=F64_RTOL,
        scale=100.0,
        context="sftof_i = 100 m_i0",
    )


def _finite_weights(regridder) -> scipy.sparse.csr_matrix:
    """ŵ as loaded (target, source), the NaN markers on Z dropped."""
    W = regridder.weights.data.to_scipy_sparse().tocsr()
    W.data[~np.isfinite(W.data)] = 0.0
    W.eliminate_zeros()
    return W


def check_static_identities(
    statics: NativeAreaStatics,
    regridder,
    wetmask: xr.DataArray,
    areacello_native: xr.DataArray,
    deptho_native: xr.DataArray | None = None,
    deptho: xr.DataArray | None = None,
) -> None:
    """Step-5 per-store asserts on the coarse statics, and static identities:

    Z ⊂ {mask_J0 = 0};  A_J ≥ 0;  mask_Jk = 1[regridder(m_k) > 0];  statics finite
    Σ_k V'_Jk = W'_J0 deptho_J                       mask_J0 = 1, with depths given
    Σ_J A_J = Σ_i a_i;   Σ_J ŵ_Ji A_J = a_i          every i
    sftof_J = 100 where every i with O_Ji > 0 is ocean
    0 ≤ sftof_J ≤ 100;   sea_surface_fraction_J = W'_J0 / A_J    J ∉ Z

    ``wetmask`` has dims ``(z_l, ...)``; one level for a surface-only store."""
    stored = statics.stored
    area = stored["areacello"]
    n_levels = wetmask.sizes[LEVEL_DIM]
    for name in stored.data_vars:
        n_nonfinite = int((~np.isfinite(stored[name])).sum())
        if n_nonfinite:
            raise AssertionError(f"{name}: {n_nonfinite} non-finite values")
    if int((area < 0).sum()):
        raise AssertionError("A_J < 0")
    Z = (area == 0).values
    ocean = (stored["mask_0"] > 0).values
    if (Z & ocean).any():
        raise AssertionError(f"{int((Z & ocean).sum())} Z cells have mask_J0 = 1")

    m = wetmask.astype(np.float64)
    for k in range(n_levels):
        fraction = _regrid(regridder, m.isel({LEVEL_DIM: k}, drop=True)).fillna(0.0)
        today = (fraction > 0).values
        n_mismatch = int((today != (stored[f"mask_{k}"] > 0).values).sum())
        if n_mismatch:
            raise AssertionError(
                f"mask_{k} = 1[V'_Jk > 0] differs from 1[regridder(m_k) > 0] "
                f"on {n_mismatch} cells"
            )

    m0 = m.isel({LEVEL_DIM: 0}, drop=True)
    volume = statics.ocean_volume
    if deptho_native is not None and deptho is not None and volume is not None:
        H = deptho_native.fillna(0.0).astype(np.float64)
        # Σ_i O_Ji m_i0 H_i
        column_scale = (area * _regrid(regridder, m0 * H)).fillna(0.0)
        assert_close(
            volume.sum(LEVEL_DIM).values[ocean],
            (statics.ocean_area * deptho).values[ocean],
            rtol=F64_RTOL,
            scale=column_scale.values[ocean],
            context="Σ_k V'_Jk = W'_J0 deptho_J",
        )

    a = areacello_native.astype(np.float64)
    assert_close(
        float(area.sum()),
        float(a.sum()),
        rtol=F64_RTOL,
        scale=float(a.sum()),
        context="Σ_J A_J = Σ_i a_i",
    )
    a_i = a.transpose(*regridder.in_horiz_dims).values.ravel()
    A_J = area.transpose(*regridder.out_horiz_dims).values.ravel()
    assert_close(
        _finite_weights(regridder).T @ A_J,
        a_i,
        rtol=F64_RTOL,
        scale=a_i,
        context="Σ_J ŵ_Ji A_J = a_i",
    )

    sftof = stored["sftof"].values
    # Exact zero: a sum of ŵ_Ji > 0 times (1 − m_i0) ∈ {0, 1} is 0 iff no land i.
    all_ocean = (_regrid(regridder, 1.0 - m0) == 0).values
    assert_close(
        sftof[all_ocean], 100.0, rtol=F64_RTOL, scale=100.0, context="sftof_J = 100"
    )
    assert_bound(
        sftof, lower=0.0, upper=100.0, rtol=F64_RTOL, scale=100.0, context="sftof_J"
    )
    mapped = ~Z
    surface = (statics.ocean_area / area.where(area > 0)).values[mapped]
    assert_close(
        stored["sea_surface_fraction"].values[mapped],
        surface,
        rtol=F64_RTOL,
        scale=surface,
        context="sea_surface_fraction_J = W'_J0 / A_J",
    )


# ---------------------------------------------------------------------------
# Fields (step 3) and their identities (step 5), per (time, level) chunk
# ---------------------------------------------------------------------------


def regrid_thickness_weighted(
    ds: xr.Dataset, regridder, wetmask_k: xr.DataArray, dz_k: xr.DataArray
) -> tuple[xr.Dataset, xr.DataArray]:
    """Per-ocean-volume regrid of every variable of one level ``k``, float64:

        X^tw_Jk = regridder((m_k Δz_k X).fillna(0)) / regridder(m_k Δz_k)

    NaN where ``regridder(m_k Δz_k)`` is 0 or NaN (``mask_Jk = 0``, Z).
    Returns the regridded dataset and ``regridder(m_k).fillna(0)``."""
    m = wetmask_k.astype(np.float64)
    weight = (m * dz_k.astype(np.float64)).fillna(0.0)
    divisor = _regrid(regridder, weight)
    divisor = divisor.where(divisor > 0)
    out = xr.Dataset(attrs=ds.attrs)
    for name, da in ds.data_vars.items():
        numerator = (da.astype(np.float64).where(wetmask_k) * weight).fillna(0.0)
        out[name] = regridder(numerator, keep_attrs=True) / divisor
        out[name].attrs = da.attrs
    return out, _regrid(regridder, m).fillna(0.0)


def regrid_full_cell(
    regridder, da: xr.DataArray, wetmask: xr.DataArray, ocean_fraction: xr.DataArray
) -> xr.DataArray:
    """X_J = regridder(X.where(m_0).fillna(0)), float64, NaN where
    ``ocean_fraction ≤ OCEAN_FRACTION_THRESHOLD`` (``mask_J0 = 0``, Z)."""
    full = regridder(da.astype(np.float64).where(wetmask).fillna(0.0), keep_attrs=True)
    return full.where(ocean_fraction > OCEAN_FRACTION_THRESHOLD)


def _native_sum(
    weight: xr.DataArray, wetmask: xr.DataArray, x: xr.DataArray
) -> xr.DataArray:
    """Σ_i weight_i m_i x_i, per time (and level)."""
    weight, wetmask, x = xr.align(weight, wetmask, x, join="exact")
    return (weight * x.where(wetmask, 0.0)).sum(wetmask.dims, skipna=False)


def _target_sum(
    weight: xr.DataArray, x: xr.DataArray, where: xr.DataArray
) -> xr.DataArray:
    """Σ_J weight_J x_J over ``where``, per time (and level)."""
    weight, x, where = xr.align(weight, x, where, join="exact")
    return (weight * x).where(where, 0.0).sum(where.dims, skipna=False)


def _assert_sums_close(
    target: xr.DataArray, native: xr.DataArray, scale: xr.DataArray, context: str
) -> None:
    target, native, scale = xr.broadcast(target, native, scale)
    dims = target.dims
    assert_close(
        target.transpose(*dims).values,
        native.transpose(*dims).values,
        rtol=F64_RTOL,
        scale=scale.transpose(*dims).values,
        context=context,
    )


def _check_sums(
    fields: xr.Dataset,
    native: xr.Dataset,
    target_weight: xr.DataArray,
    where: xr.DataArray,
    native_weight: xr.DataArray,
    wetmask: xr.DataArray,
    label: str,
    context: str,
) -> None:
    for name, x in fields.data_vars.items():
        _assert_sums_close(
            _target_sum(target_weight, x, where),
            _native_sum(native_weight, wetmask, native[name]),
            _native_sum(native_weight, wetmask, abs(native[name])),
            f"{context}: {label}, {name}",
        )


def check_field_identities(
    native: xr.Dataset,
    regridder,
    wetmask: xr.DataArray,
    area: xr.DataArray,
    ocean_fraction: xr.DataArray,
    native_areacello: xr.DataArray,
    context: str,
    *,
    per_volume: xr.Dataset | None = None,
    per_ocean_area: xr.Dataset | None = None,
    full_cell: xr.Dataset | None = None,
    dz: xr.DataArray | None = None,
) -> None:
    """Step-5 field identities on one chunk, per time, before postprocess:

        Σ_J V'_Jk X^tw_Jk = Σ_i a_i m_ik Δz_ik X_ik    V'_Jk = A_J regridder(m_k Δz_k)
        Σ_J W'_J0 X^a_J   = Σ_i a_i m_i0 X_i           W'_J0 = A_J ocean_fraction_J
        Σ_J A_J X_J       = Σ_i a_i m_i0 X_i           J ∉ Z, mask_J0 = 1
        c_i ≤ 1 on m_i0 = 1;  sea_ice_fraction_J ≤ 1 on J ∉ Z, mask_J0 = 1

    ``native`` holds the native fields (float64) under the names the
    regridded datasets use; ``wetmask``, ``dz`` are the chunk's level."""
    a = native_areacello.astype(np.float64)
    if per_volume is not None and per_volume.data_vars:
        if dz is None:
            raise ValueError(f"{context}: the 3-D identity needs Δz")
        weight = (wetmask.astype(np.float64) * dz.astype(np.float64)).fillna(0.0)
        volume = (area * _regrid(regridder, weight)).fillna(0.0)
        _check_sums(
            per_volume,
            native,
            volume,
            volume > 0,
            a * dz,
            wetmask,
            "Σ_J V'_Jk X^tw_Jk = Σ_i a_i m_ik Δz_ik X_ik",
            context,
        )
    surface = ocean_fraction > OCEAN_FRACTION_THRESHOLD
    covered_ocean = surface & (area > 0)
    if per_ocean_area is not None:
        _check_sums(
            per_ocean_area,
            native,
            area * ocean_fraction,
            surface,
            a,
            wetmask,
            "Σ_J W'_J0 X^a_J = Σ_i a_i m_i0 X_i",
            context,
        )
    if full_cell is None:
        return
    _check_sums(
        full_cell,
        native,
        area,
        covered_ocean,
        a,
        wetmask,
        "Σ_J A_J X_J = Σ_i a_i m_i0 X_i",
        context,
    )
    if SEA_ICE_FRACTION in full_cell:
        assert_bound(
            native[SEA_ICE_FRACTION].where(wetmask, 0.0),
            upper=1.0,
            rtol=F64_RTOL,
            scale=1.0,
            context=f"{context}: native c_i ≤ 1 on m_i0 = 1",
        )
        assert_bound(
            full_cell[SEA_ICE_FRACTION].where(covered_ocean, 0.0),
            upper=1.0,
            rtol=F64_RTOL,
            scale=1.0,
            context=f"{context}: sea_ice_fraction_J ≤ 1",
        )


# Target weight of each checked postprocess output: A_J (per full cell) or 1
# (extensive, a per-cell total).
EXTENSIVE_OUTPUTS = {"hfds_total_area": True, "sea_ice_volume": False}


def check_extensive_identities(
    output: xr.Dataset,
    native: Mapping[str, xr.DataArray],
    wetmask: xr.DataArray,
    area: xr.DataArray,
    ocean_fraction: xr.DataArray,
    native_areacello: xr.DataArray,
    context: str,
) -> None:
    """Step-5 identities on postprocess outputs (A6), J ∉ Z, mask_J0 = 1:

        Σ_J A_J hfds_total_area_J = Σ_i a_i m_i0 hfds_i
        Σ_J sea_ice_volume_J      = Σ_i a_i m_i0 HI_i c_i

    ``native`` maps each checked output name to its native counterpart."""
    covered_ocean = (ocean_fraction > OCEAN_FRACTION_THRESHOLD) & (area > 0)
    a = native_areacello.astype(np.float64)
    for name, x_native in native.items():
        if name not in EXTENSIVE_OUTPUTS:
            raise ValueError(f"{context}: no identity for postprocess output {name}")
        weight = area if EXTENSIVE_OUTPUTS[name] else xr.ones_like(area)
        _assert_sums_close(
            _target_sum(weight, output[name].astype(np.float64), covered_ocean),
            _native_sum(a, wetmask, x_native),
            _native_sum(a, wetmask, abs(x_native)),
            f"{context}: {name}, Σ_J over J ∉ Z = Σ_i a_i m_i0 (native)",
        )


# ---------------------------------------------------------------------------
# z* residual (step 5): report only, no gate (D6)
# ---------------------------------------------------------------------------

# Store root-attr key of the report-only items (a/ã spread, z* residual).
NATIVE_AREA_CHECKS = "native_area_checks"
Z_STAR_DEFINITION = (
    "per snapshot t and level k (list index): h_ik = Δz_ik (1 + η_i/H_i) where "
    "m_ik = 1, else 0 (η = SSH, H = deptho, native); sigma_J: σ_J = 1 + η_J/H_J "
    "where mask_J0 = 1, else 1 (η_J, H_J: stored SSH, deptho); sigma_1: σ_J = 1; "
    "vol = max_J |σ_J V'_Jk − Σ_i O_Ji m_ik h_ik| / max_J Σ_i O_Ji m_ik h_ik; "
    "cont[X] = |Σ_J σ_J V'_Jk X_Jk − Σ_i a_i m_ik h_ik X_ik| "
    "/ Σ_i a_i m_ik h_ik |X_ik|"
)


def z_star_thickness(
    dz: xr.DataArray, wetmask: xr.DataArray, deptho: xr.DataArray, ssh: xr.DataArray
) -> xr.DataArray:
    """h_ik = Δz_ik (1 + η_i/H_i) where m_ik = 1, else 0 (m); η = SSH."""
    H = deptho.fillna(0.0).astype(np.float64)
    stretch = 1.0 + ssh.astype(np.float64) / H.where(H > 0)
    h = xr.where(wetmask, dz.astype(np.float64) * stretch, 0.0)
    return h.transpose(LEVEL_DIM, ...)


def z_star_stretch(
    regridder, wetmask: xr.DataArray, deptho: xr.DataArray, ssh: xr.DataArray
) -> xr.DataArray:
    """σ_J = 1 + η_J/H_J where mask_J0 = 1, else 1; η_J and H_J = deptho_J
    are the per-ocean-area regrids stored as SSH, deptho (float64)."""
    m0 = wetmask.isel({LEVEL_DIM: 0}, drop=True)
    coarse, fraction = regrid_normalized(
        xr.Dataset({"SSH": ssh, "deptho": deptho}), regridder, m0
    )
    sigma = 1.0 + coarse["SSH"] / coarse["deptho"]
    return sigma.where(fraction > OCEAN_FRACTION_THRESHOLD, 1.0)


def z_star_residual(
    regridder,
    area: xr.DataArray,
    wetmask: xr.DataArray,
    dz: xr.DataArray,
    native_areacello: xr.DataArray,
    deptho: xr.DataArray,
    ssh: xr.DataArray,
    fields: xr.Dataset,
) -> dict:
    """z* residual of one snapshot, one list entry per level k, for σ_J and
    σ_J = 1 (``Z_STAR_DEFINITION``):

        vol_k  = max_J |σ_J V'_Jk − Σ_i O_Ji m_ik h_ik| / max_J Σ_i O_Ji m_ik h_ik
        cont_k = |Σ_J σ_J V'_Jk X^tw_Jk − Σ_i a_i m_ik h_ik X_ik|
                 / Σ_i a_i m_ik h_ik |X_ik|

    Σ_i O_Ji y_i = A_J regridder(y)_J; V'_Jk = A_J regridder(m_k Δz_k);
    X^tw from regrid_thickness_weighted. ``fields``: native X, dims
    ``(z_l, ...)``; ``ssh``: native η."""
    native_inputs = xr.align(
        wetmask, dz, native_areacello, deptho, ssh, fields, join="exact"
    )
    wetmask, dz, native_areacello, deptho, ssh, fields = native_inputs
    a = native_areacello.astype(np.float64)
    h = z_star_thickness(dz, wetmask, deptho, ssh)
    sigmas = {
        "sigma_J": z_star_stretch(regridder, wetmask, deptho, ssh),
        "sigma_1": 1.0,
    }
    vol: dict[str, list[float]] = {key: [] for key in sigmas}
    cont: dict[str, dict[str, list[float]]] = {
        key: {str(name): [] for name in fields.data_vars} for key in sigmas
    }
    for k in range(wetmask.sizes[LEVEL_DIM]):
        level = {LEVEL_DIM: k}
        m_k = wetmask.isel(level, drop=True)
        dz_k = dz.isel(level, drop=True)
        h_k = h.isel(level, drop=True)
        volume = (area * _regrid(regridder, (m_k * dz_k).fillna(0.0))).fillna(0.0)
        native_volume = (area * _regrid(regridder, h_k)).fillna(0.0)
        x_tw, _ = regrid_thickness_weighted(
            fields.isel(level, drop=True), regridder, m_k, dz_k
        )
        for key, sigma in sigmas.items():
            coarse = sigma * volume
            vol[key].append(
                float(abs(coarse - native_volume).max() / native_volume.max())
            )
            for name, values in cont[key].items():
                x = fields[name].isel(level, drop=True).astype(np.float64)
                x = x.where(m_k, 0.0)
                native = float((a * h_k * x).sum())
                scale = float((a * h_k * abs(x)).sum())
                target = float((coarse * x_tw[name]).where(volume > 0, 0.0).sum())
                values.append(abs(target - native) / scale)
    return {key: {"vol": vol[key], "cont": cont[key]} for key in sigmas}
