import torch

from fme.core.constants import FREEZING_TEMPERATURE_KELVIN


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
