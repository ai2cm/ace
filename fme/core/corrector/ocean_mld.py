"""Mixed layer depth (MLD) helpers for ocean budget corrections.

Provides JMD95 equation of state, MLD computation, MLD-based vertical
weighting, and geothermal bottom-cell correction.
"""

import torch

from fme.core.constants import (
    DELTA_RHO_THRESHOLD,
    DENSITY_OF_WATER_CM4,
    MLD_REF_LAYER,
    SPECIFIC_HEAT_OF_WATER_CM4,
)
from fme.core.ocean_data import HasOceanDepthIntegral, OceanData


@torch.jit.script
def jmd95_potential_density(S: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Potential density at surface pressure via the UNESCO / JMD95 polynomial.

    Reference:
        Jackett, D. R. and T. J. McDougall, 1995: Minimal adjustment of
        hydrographic profiles to achieve static stability. *J. Atmos. Oceanic
        Technol.*, **12**, 381--389, doi:10.1175/1520-0426(1995)012<0381:
        MAOHPT>2.0.CO;2.

    Args:
        S: Practical salinity (PSU).
        theta: Potential temperature (degC).

    Returns:
        Potential density (kg/m^3).
    """
    t2 = theta**2
    t3 = theta**3
    t4 = theta**4
    t5 = theta**5

    rho_w = (
        999.842594
        + 6.793952e-2 * theta
        - 9.095290e-3 * t2
        + 1.001685e-4 * t3
        - 1.120083e-6 * t4
        + 6.536332e-9 * t5
    )

    A = (
        8.244930e-1
        - 4.089900e-3 * theta
        + 7.643800e-5 * t2
        - 8.246700e-7 * t3
        + 5.387500e-9 * t4
    )

    B = -5.724660e-3 + 1.022700e-4 * theta - 1.654600e-6 * t2

    C = 4.831400e-4

    return rho_w + A * S + B * S**1.5 + C * S**2


def compute_mld(
    density: torch.Tensor,
    idepth: torch.Tensor,
    deptho: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = DELTA_RHO_THRESHOLD,
    ref_layer: int = MLD_REF_LAYER,
) -> torch.Tensor:
    """Compute mixed layer depth by linearly interpolating a density threshold.

    MLD is the shallowest depth where the potential density exceeds
    ``density[ref_layer] + threshold``.  Linear interpolation between
    layer centres is used.  Where the threshold is never exceeded, MLD
    falls back to ``deptho`` (sea-floor depth).

    Args:
        density: Potential density, shape ``(B, Y, X, Z)``.
        idepth: Interface depths (1-D, length ``Z + 1``).
        deptho: Sea-floor depth, shape broadcastable to ``(B, Y, X)``.
        mask: Ocean mask, shape broadcastable to ``(B, Y, X, Z)``.
        threshold: Density difference threshold (kg/m^3).
        ref_layer: Reference layer index for the density difference.

    Returns:
        MLD in metres, shape ``(B, Y, X)``.
    """
    lev_center = (idepth[:-1] + idepth[1:]) / 2.0
    nlev = density.shape[-1]

    rho_ref = density[..., ref_layer : ref_layer + 1]
    delta_rho = density - rho_ref

    mld = torch.full_like(density[..., 0], float("nan"))
    not_yet_set = torch.ones_like(mld, dtype=torch.bool)

    for k in range(ref_layer + 1, nlev):
        exceeded = delta_rho[..., k] > threshold
        layer_valid = mask[..., k] > 0
        layer_mask = exceeded & not_yet_set & layer_valid

        if k == ref_layer + 1:
            drho_prev = torch.zeros_like(delta_rho[..., k])
            z_prev = lev_center[ref_layer]
        else:
            drho_prev = delta_rho[..., k - 1]
            z_prev = lev_center[k - 1]

        drho_curr = delta_rho[..., k]
        z_curr = lev_center[k]

        frac = (threshold - drho_prev) / (drho_curr - drho_prev + 1e-8)
        interp_depth = z_prev + frac * (z_curr - z_prev)

        mld = torch.where(layer_mask, interp_depth, mld)
        not_yet_set = not_yet_set & ~layer_mask

    mld = torch.where(torch.isnan(mld), deptho, mld)
    return mld


def compute_mld_active_thickness(
    mld_2d: torch.Tensor,
    idepth: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-layer thickness that falls within the mixed layer.

    Args:
        mld_2d: Mixed layer depth, shape ``(B, Y, X)``.
        idepth: Interface depths (1-D, length ``Z + 1``).
        mask: Ocean mask, shape broadcastable to ``(B, Y, X, Z)``.

    Returns:
        Active thickness ``h`` with shape ``(B, Y, X, Z)``, zeroed outside
        the mask and where MLD is NaN.
    """
    z_top = idepth[:-1]  # (Z,)
    z_bot = idepth[1:]  # (Z,)
    mld = mld_2d.unsqueeze(-1)  # (B, Y, X, 1)

    active = torch.clamp(torch.clamp(z_bot, max=mld) - z_top, min=0.0)

    valid = ~torch.isnan(mld)
    active = torch.where(valid, active, torch.zeros_like(active))
    active = active * mask

    return active


def compute_mld_weights_from_ocean_data(
    gen: OceanData,
    forcing: OceanData,
    vertical_coordinate: HasOceanDepthIntegral,
) -> torch.Tensor:
    """Compute MLD active-thickness weights from generated ocean state.

    Computes JMD95 potential density from ``gen`` temperature and salinity,
    derives MLD, and returns the per-layer active thickness within the
    mixed layer.

    Args:
        gen: Generated ocean data (must contain ``thetao`` and ``so``).
        forcing: Forcing data (must contain ``deptho``).
        vertical_coordinate: Provides interface depths and ocean mask.

    Returns:
        Active thickness tensor ``h`` with shape ``(B, Y, X, Z)``.
    """
    theta = gen.sea_water_potential_temperature  # (B, Y, X, Z)
    S = gen.sea_water_salinity  # (B, Y, X, Z)
    density = jmd95_potential_density(S, theta)

    idepth = vertical_coordinate.get_idepth()
    mask = vertical_coordinate.get_mask()
    deptho = forcing.sea_floor_depth

    mld_2d = compute_mld(density, idepth, deptho, mask)
    return compute_mld_active_thickness(mld_2d, idepth, mask)


def apply_geothermal_bottom_correction(
    gen: OceanData,
    forcing: OceanData,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
) -> None:
    """Apply geothermal heat flux correction to the bottom ocean cell.

    Modifies ``gen.data`` in place by adding a temperature increment to
    each column's deepest valid layer proportional to the local
    geothermal heat flux.

    Args:
        gen: Generated ocean data (modified in place).
        forcing: Forcing data (must contain ``hfgeou`` and
            ``sea_surface_fraction``).
        vertical_coordinate: Provides interface depths and ocean mask.
        timestep_seconds: Model timestep in seconds.
    """
    mask = vertical_coordinate.get_mask()  # broadcastable to (B, Y, X, Z)
    dz = vertical_coordinate.get_idepth().diff(dim=-1)  # (Z,)

    wet_count = mask.sum(dim=-1, keepdim=True)
    cumulative = mask.cumsum(dim=-1)
    bottom_mask = (cumulative == wet_count) & (mask > 0)

    hfgeou = forcing.geothermal_heat_flux
    ssf = forcing.sea_surface_fraction

    dT_geo = (
        (hfgeou * ssf).unsqueeze(-1)
        * timestep_seconds
        / (DENSITY_OF_WATER_CM4 * SPECIFIC_HEAT_OF_WATER_CM4 * dz)
    )
    dT_geo = dT_geo * bottom_mask

    nlev = gen.sea_water_potential_temperature.shape[-1]
    for k in range(nlev):
        gen.data[f"thetao_{k}"] = gen.data[f"thetao_{k}"] + dT_geo[..., k]
