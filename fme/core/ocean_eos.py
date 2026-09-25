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

Port of ``analysis/2026-09-23-1523-wright97-rhoinsitu/wright97_eos.py`` in the
ai2cm workspace (``wright97_anomaly``, ``boussinesq_pressure``,
``interface_to_center_depth``).
"""

import torch

from fme.core.constants import DENSITY_OF_SEA_WATER_CM4, GRAVITY

# Boussinesq constants of the CM4 runs (MOM6 ``RHO_0``, ``G_EARTH``).
RHO_0 = DENSITY_OF_SEA_WATER_CM4  # [kg m-3]
G_EARTH = GRAVITY  # [m s-2]

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
