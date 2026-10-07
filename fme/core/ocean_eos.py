"""Wright (1997) equation of state, reduced-range fit, in the anomaly form.

    al0(T,S)    = a0 + a1*T + a2*S                               [m3 kg-1]
    p0(T,S)     = b0 + b4*S + T*(b1 + T*(b2 + b3*T) + b5*S)      [Pa]
    lambda(T,S) = c0 + c4*S + T*(c1 + T*(c2 + c3*T) + c5*S)      [m2 s-2]
    rho(S,T,p)  = (p + p0) / (lambda + al0*(p + p0))             [kg m-3]

T potential temperature [degC], S practical salinity [PSU], p gauge pressure
[Pa]. MOM6 ``EQN_OF_STATE = "WRIGHT"``,
``MOM_EOS_Wright.F90::calculate_density_array_wright``; the anomaly form is
that routine's ``rho_ref`` branch, which keeps float32 precision by not forming
``rho`` itself.
"""

import torch

from fme.core.constants import DENSITY_OF_SEA_WATER_CM4, GRAVITY

# Boussinesq constants of the CM4 runs (MOM6 ``RHO_0``, ``G_EARTH``).
RHO_0 = DENSITY_OF_SEA_WATER_CM4  # [kg m-3]
G_EARTH = GRAVITY  # [m s-2]

# Density-threshold mixed layer depth defaults of ``OceanData.mld_wright97``.
DELTA_RHO_THRESHOLD = 0.03  # [kg m-3], as MOM6's mlotst diagnostic:
# https://github.com/NOAA-GFDL/MOM6/blob/89921f1c0a6486f4bd7f6d5428fd101e2610a05f/src/parameterizations/vertical/MOM_diabatic_driver.F90#L3499-L3502
MLD_REF_LAYER = 1  # reference layer index, dimensionless

# Reduced-range fit, -2<T<30 degC, 28<S<38 PSU, 0<p<5e7 Pa.
_A0, _A1, _A2 = 7.057924e-4, 3.480336e-7, -1.112733e-7
_B0, _B1, _B2, _B3, _B4, _B5 = (
    5.790749e8, 3.516535e6, -4.002714e4, 2.084372e2, 5.944068e5, -9.643486e3,
)  # fmt: skip
_C0, _C1, _C2, _C3, _C4, _C5 = (
    1.704853e5, 7.904722e2, -7.984422, 5.140652e-2, -2.302158e2, -3.079464,
)  # fmt: skip


def wright97_anomaly(
    S: torch.Tensor, theta: torch.Tensor, p: torch.Tensor, rho_ref: float = RHO_0
) -> torch.Tensor:
    """In-situ density anomaly ``rho - rho_ref`` via Wright (1997).

    Elementwise with broadcasting, differentiable, computed in the inputs'
    dtype (Python-float coefficients do not promote a float32 tensor).

    Args:
        S: Practical salinity (PSU).
        theta: Potential temperature (degC).
        p: Gauge pressure (Pa).
        rho_ref: Reference density (kg/m^3).

    Returns:
        rho - rho_ref (kg/m^3).
    """
    # A Python float, so its cancellation happens in float64.
    pa_000 = _B0 * (1.0 - _A0 * rho_ref) - rho_ref * _C0
    al_TS = _A1 * theta + _A2 * S
    al0 = _A0 + al_TS
    p_TSp = p + (_B4 * S + theta * (_B1 + (theta * (_B2 + _B3 * theta) + _B5 * S)))
    lam_TS = _C4 * S + theta * (_C1 + (theta * (_C2 + _C3 * theta) + _C5 * S))
    num = pa_000 + (p_TSp - rho_ref * (p_TSp * al0 + (_B0 * al_TS + lam_TS)))
    return num / ((_C0 + lam_TS) + al0 * (_B0 + p_TSp))


def boussinesq_pressure(
    depth: torch.Tensor, rho_0: float = RHO_0, g_earth: float = G_EARTH
) -> torch.Tensor:
    """``p = rho_0 * g_earth * depth`` [Pa], depth [m] positive down."""
    return rho_0 * g_earth * depth


def interface_to_center_depth(idepth: torch.Tensor) -> torch.Tensor:
    """Level-centre depth ``(idepth[:-1] + idepth[1:]) / 2`` from interfaces."""
    return 0.5 * (idepth[:-1] + idepth[1:])


def _sea_floor_depth(
    idepth: torch.Tensor, mask: torch.Tensor, deptho: torch.Tensor | None
) -> torch.Tensor:
    """Sea floor depth [m]: ``deptho`` when given, else
    ``max_k(mask_k * idepth_{k+1})``, in ``idepth``'s dtype.
    """
    if deptho is not None:
        return deptho.to(idepth.dtype)
    return (mask * idepth[1:]).max(dim=-1).values


def _mixed_layer_depth(
    thetao: torch.Tensor,
    so: torch.Tensor,
    idepth: torch.Tensor,
    mask: torch.Tensor,
    deptho: torch.Tensor,
    delta_rho_threshold: float,
    ref_layer: int,
) -> torch.Tensor:
    """Density-threshold mixed layer depth [m], positive down.

    ``rho = wright97_anomaly(so, thetao, p=0)``; the MLD is where
    ``rho_k - rho_ref`` first exceeds ``delta_rho_threshold`` below
    ``ref_layer``, linearly interpolated between level centres, and
    ``deptho`` where it never does.

    Args:
        thetao: Potential temperature [degC], ``(..., nz)``.
        so: Practical salinity [PSU], ``(..., nz)``.
        idepth: Interface depths [m], ``(nz + 1,)``.
        mask: Ocean mask, broadcastable to ``thetao``.
        deptho: Sea floor depth [m], broadcastable to ``thetao.shape[:-1]``.
        delta_rho_threshold: Density threshold [kg/m**3].
        ref_layer: Reference layer index.
    """
    n_levels = thetao.shape[-1]
    if not 0 <= ref_layer < n_levels - 1:
        raise ValueError(
            f"mld_ref_layer must be in [0, {n_levels - 2}], got {ref_layer}"
        )
    # NaN inputs (land, below the bottom) are replaced by finite values before
    # the EOS and the interpolation so that the zero gradient torch.where passes
    # to unselected entries is never multiplied by NaN; d keeps its NaN there so
    # the forward selection is unchanged.
    finite = torch.isfinite(thetao) & torch.isfinite(so)
    rho = wright97_anomaly(
        torch.where(finite, so, torch.zeros_like(so)),
        torch.where(finite, thetao, torch.zeros_like(thetao)),
        torch.zeros_like(thetao),
    )
    rho = torch.where(finite, rho, torch.full_like(rho, float("nan")))
    zc = interface_to_center_depth(idepth)
    d = rho - rho[..., ref_layer : ref_layer + 1]
    d_safe = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    mask = mask.expand(d.shape)
    mld = torch.full_like(d[..., 0], float("nan"))
    notset = torch.ones_like(mld, dtype=torch.bool)
    for k in range(ref_layer + 1, n_levels):
        lm = (d[..., k] > delta_rho_threshold) & notset & (mask[..., k] > 0)
        if k == ref_layer + 1:
            dprev = torch.zeros_like(d[..., k])
            dprev_safe = dprev
        else:
            dprev = d[..., k - 1]
            dprev_safe = d_safe[..., k - 1]
        zprev = zc[ref_layer] if k == ref_layer + 1 else zc[k - 1]
        denom = d_safe[..., k] - dprev_safe + 1e-8
        denom = torch.where(lm, denom, torch.ones_like(denom))
        frac = (delta_rho_threshold - dprev_safe) / denom
        value = zprev + frac * (zc[k] - zprev)
        value = torch.where(
            torch.isfinite(dprev), value, torch.full_like(value, float("nan"))
        )
        mld = torch.where(lm, value, mld)
        notset = notset & ~lm
    return torch.where(torch.isnan(mld), deptho.expand(mld.shape), mld)


def _density_anomaly(
    thetao: torch.Tensor,
    so: torch.Tensor,
    idepth: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """In-situ density anomaly ``rho_k - RHO_0`` [kg m-3] at the Boussinesq
    pressure of each level centre, NaN where ``mask_k == 0`` or an input is NaN.

    Args:
        thetao: Potential temperature [degC], ``(..., nz)``.
        so: Practical salinity [PSU], ``(..., nz)``.
        idepth: Interface depths [m], ``(nz + 1,)``.
        mask: Ocean mask, broadcastable to ``thetao``.

    Returns:
        ``(..., nz)``, level ``k`` on the last dim.
    """
    p = boussinesq_pressure(interface_to_center_depth(idepth.to(torch.float64)))
    p = p.to(dtype=thetao.dtype, device=thetao.device)
    valid = (mask.to(thetao.device) > 0) & thetao.isfinite() & so.isfinite()
    return torch.where(valid, wright97_anomaly(so, thetao, p), torch.nan)


def _column_density_integral(rho: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    """``C = sum_k rho_k * dz_k`` [kg m-2] over the last dim, NaN ``rho_k``
    contributing 0.
    """
    dz = dz.to(dtype=rho.dtype, device=rho.device)
    return torch.where(rho.isfinite(), rho * dz, 0.0).sum(dim=-1)
