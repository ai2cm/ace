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

import esmpy
import numpy as np
import scipy.sparse
import xarray as xr
from xesmf.frontend import ds_to_ESMFgrid

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
