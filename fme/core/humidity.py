import torch

from fme.core.constants import FREEZING_TEMPERATURE_KELVIN, RDGAS, RVGAS

# Ratio of the gas constant of dry air to that of water vapor (~0.622).
EPSILON_WATER_VAPOR = RDGAS / RVGAS


def bolton_saturation_vapor_pressure(temperature_kelvin: torch.Tensor) -> torch.Tensor:
    """Saturation vapor pressure over liquid water from the Bolton (1980) formula.

    Returns ``e_s = 6.112 * exp(17.67 * T_C / (T_C + 243.5))`` in hPa, where
    ``T_C`` is temperature in degrees Celsius.

    Args:
        temperature_kelvin: Temperature in Kelvin.

    Returns:
        Saturation vapor pressure in hPa, same shape as input.
    """
    t_c = temperature_kelvin - FREEZING_TEMPERATURE_KELVIN
    return 6.112 * torch.exp(17.67 * t_c / (t_c + 243.5))


def saturation_specific_humidity(
    temperature_kelvin: torch.Tensor,
    pressure_pa: torch.Tensor,
) -> torch.Tensor:
    """Saturation specific humidity over liquid water in kg/kg.

    Derived from the Bolton (1980) saturation vapor pressure ``e_s`` and the
    ambient pressure ``p`` as

        q_sat = eps * e_s / (p - (1 - eps) * e_s),

    where ``eps = R_dry / R_vapor``.  ``e_s`` is converted from hPa (the unit
    returned by :func:`bolton_saturation_vapor_pressure`) to Pa so it matches
    ``pressure_pa``.

    Args:
        temperature_kelvin: Temperature in Kelvin.
        pressure_pa: Ambient pressure in Pa, broadcastable to the temperature.

    Returns:
        Saturation specific humidity in kg/kg, broadcast to the common shape.
    """
    e_s_pa = bolton_saturation_vapor_pressure(temperature_kelvin) * 100.0
    return (
        EPSILON_WATER_VAPOR
        * e_s_pa
        / (pressure_pa - (1.0 - EPSILON_WATER_VAPOR) * e_s_pa)
    )
