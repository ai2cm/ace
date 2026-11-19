"""
Step-derived input: clear-sky outgoing longwave flux in the CO2 15 micron band.

The estimate follows the exponential-band, emission-level model of Jeevanjee
et al. (2021, J. Climate 34(23), "An Analytical Model for Spatially Varying
Clear-Sky CO2 Forcing"), eqs. (1)-(3). The CO2 reference absorption
coefficient decays exponentially away from band center,

    kappa_ref(nu) = kappa_0 * exp(-|nu - nu_0| / l),

and with pressure broadening each wavenumber emits to space from the pressure

    p_em(nu, q) = p_0(q) * exp(|nu - nu_0| / (2 l)),
    p_0(q) = sqrt(2 tau_em g p_ref / (D q kappa_0)),

where q is the CO2 mass concentration. Wavenumbers whose emission pressure
exceeds the surface pressure see the surface. Changing variables from
wavenumber to ln(p_em) (d nu = 2 l d ln p on each side of band center), the
flux over a fixed interval of half-width W about nu_0 is

    F = 4 l * integral_{ln p_0}^{ln p_s} pi B(nu_0, T(p)) d ln p
        + 2 (W - 2 l ln(p_s / p_0)) * pi B(nu_0, T_s).

Differentiating in ln q recovers the paper's forcing,
-2 l [pi B(nu_0, T_s) - pi B(nu_0, T(p_0))] per unit ln q. Water vapor is not
represented. Temperature is taken as constant within each model layer, so the
flux is continuous in every input and no level is selected.
"""

import dataclasses
import math

import torch

from fme.core.coordinates import HybridSigmaPressureCoordinate, VerticalCoordinate
from fme.core.typing_ import TensorMapping

# Jeevanjee et al. (2021), section 2 and Table 1.
BAND_CENTER_WAVENUMBER = 667.5  # nu_0, cm^-1
SPECTROSCOPIC_DECAY = 10.2  # l, cm^-1
REFERENCE_ABSORPTION = 50.0  # kappa_0, m^2 kg^-1, at (250 K, 100 hPa)
REFERENCE_PRESSURE = 1.0e4  # p_ref, Pa
DIFFUSIVITY_FACTOR = 1.5  # D
EMISSION_OPTICAL_DEPTH = 0.5  # tau_em
GRAVITY = 9.80665  # m s^-2
CO2_MOLAR_MASS_RATIO = 44.01 / 28.97  # CO2 / dry air

_PLANCK_C1 = 1.191042972e-16  # 2 h c^2, W m^2 sr^-1
_PLANCK_C2 = 1.438776877e-2  # h c / k, m K


def planck_flux_density(temperature: torch.Tensor) -> torch.Tensor:
    """
    Hemispheric Planck flux density pi * B(nu_0, T) at the CO2 band center.

    Args:
        temperature: Temperature in K.

    Returns:
        Flux density in W m^-2 (cm^-1)^-1.
    """
    wavenumber = BAND_CENTER_WAVENUMBER * 100.0  # m^-1
    radiance = (
        _PLANCK_C1 * wavenumber**3 / torch.expm1(_PLANCK_C2 * wavenumber / temperature)
    )  # W m^-2 sr^-1 (m^-1)^-1
    return math.pi * 100.0 * radiance


def band_center_emission_pressure(co2_volume_mixing_ratio: torch.Tensor):
    """
    Emission pressure p_0 at the CO2 band center, in Pa.

    Args:
        co2_volume_mixing_ratio: CO2 volume mixing ratio (mol/mol).
    """
    q = co2_volume_mixing_ratio * CO2_MOLAR_MASS_RATIO
    return torch.sqrt(
        2.0
        * EMISSION_OPTICAL_DEPTH
        * GRAVITY
        * REFERENCE_PRESSURE
        / (DIFFUSIVITY_FACTOR * q * REFERENCE_ABSORPTION)
    )


def co2_band_olr(
    air_temperature: torch.Tensor,
    surface_temperature: torch.Tensor,
    interface_pressure: torch.Tensor,
    co2_volume_mixing_ratio: torch.Tensor,
    band_half_width: float,
) -> torch.Tensor:
    """
    Clear-sky outgoing longwave flux in the CO2 band, per column.

    Args:
        air_temperature: Layer temperatures in K, shape [..., n_layers],
            ordered top to bottom.
        surface_temperature: Surface temperature in K, shape [...].
        interface_pressure: Layer interface pressures in Pa, shape
            [..., n_layers + 1], ordered top to bottom.
        co2_volume_mixing_ratio: CO2 volume mixing ratio (mol/mol), shape [...].
        band_half_width: Half-width W of the spectral interval about band
            center, in cm^-1.

    Returns:
        Flux in W m^-2, shape [...].
    """
    log_p0 = torch.log(band_center_emission_pressure(co2_volume_mixing_ratio))
    log_p = torch.log(interface_pressure)
    log_p_surface = log_p[..., -1]
    layer_top = torch.maximum(log_p[..., :-1], log_p0.unsqueeze(-1))
    emitting_thickness = (log_p[..., 1:] - layer_top).clamp(min=0.0)
    atmosphere = (
        4.0
        * SPECTROSCOPIC_DECAY
        * (planck_flux_density(air_temperature) * emitting_thickness).sum(dim=-1)
    )
    absorbing_half_width = 2.0 * SPECTROSCOPIC_DECAY * (log_p_surface - log_p0)
    surface = (
        2.0
        * (band_half_width - absorbing_half_width)
        * planck_flux_density(surface_temperature)
    )
    return atmosphere + surface


@dataclasses.dataclass
class CO2BandOLRInputConfig:
    """
    Replace the CO2 network input with a Step-derived estimate of the
    clear-sky outgoing longwave flux in the CO2 band.

    The channel named ``co2_name`` stays in the step's ``in_names`` (so the
    data requirements and channel count are unchanged), but the value the
    network receives in that channel is the normalized band flux, computed
    each step from the step's input temperatures, surface pressure and CO2.

    Parameters:
        mean: Mean of the band flux over training data, in W m^-2, used to
            normalize the channel.
        std: Standard deviation of the band flux over training data, in
            W m^-2, used to normalize the channel.
        co2_name: Name of the CO2 volume mixing ratio (mol/mol) input.
        surface_temperature_name: Name of the surface temperature input.
        surface_pressure_name: Name of the surface pressure input.
        air_temperature_prefix: Prefix of the layer temperature inputs, which
            are named ``<prefix><k>`` for layer k, ordered top to bottom.
        band_half_width: Half-width of the spectral interval about the band
            center, in cm^-1. Must exceed the half-width over which CO2 is
            opaque, about 80 cm^-1 at 280 ppmv and 94 cm^-1 at 1120 ppmv for
            a 1000 hPa surface.
    """

    mean: float
    std: float
    co2_name: str = "global_mean_co2"
    surface_temperature_name: str = "surface_temperature"
    surface_pressure_name: str = "PRESsfc"
    air_temperature_prefix: str = "air_temperature_"
    band_half_width: float = 100.0

    def __post_init__(self):
        if self.std <= 0:
            raise ValueError(f"std must be positive, got {self.std}")

    def air_temperature_names(self, n_layers: int) -> list[str]:
        return [f"{self.air_temperature_prefix}{k}" for k in range(n_layers)]

    def validate_names(self, in_names: list[str]) -> None:
        for name in (
            self.co2_name,
            self.surface_temperature_name,
            self.surface_pressure_name,
        ):
            if name not in in_names:
                raise ValueError(
                    f"co2_band_olr_input requires '{name}' to be in in_names"
                )

    def build(
        self, vertical_coordinate: VerticalCoordinate, in_names: list[str]
    ) -> "CO2BandOLRInput":
        if not isinstance(vertical_coordinate, HybridSigmaPressureCoordinate):
            raise ValueError(
                "co2_band_olr_input requires a hybrid sigma-pressure vertical "
                f"coordinate, got {type(vertical_coordinate).__name__}"
            )
        n_layers = len(vertical_coordinate) - 1
        for name in self.air_temperature_names(n_layers):
            if name not in in_names:
                raise ValueError(
                    f"co2_band_olr_input requires '{name}' to be in in_names"
                )
        return CO2BandOLRInput(self, vertical_coordinate)


class CO2BandOLRInput:
    """Computes the normalized CO2-band flux that replaces the CO2 input."""

    def __init__(
        self,
        config: CO2BandOLRInputConfig,
        vertical_coordinate: HybridSigmaPressureCoordinate,
    ):
        self._config = config
        self._vertical_coordinate = vertical_coordinate
        self._air_temperature_names = config.air_temperature_names(
            len(vertical_coordinate) - 1
        )

    def flux(self, input: TensorMapping) -> torch.Tensor:
        """Band flux in W m^-2 from denormalized step inputs."""
        surface_pressure = input[self._config.surface_pressure_name]
        coordinate = self._vertical_coordinate.to(str(surface_pressure.device))
        air_temperature = torch.stack(
            [input[name] for name in self._air_temperature_names], dim=-1
        )
        return co2_band_olr(
            air_temperature=air_temperature,
            surface_temperature=input[self._config.surface_temperature_name],
            interface_pressure=coordinate.interface_pressure(surface_pressure),
            co2_volume_mixing_ratio=input[self._config.co2_name],
            band_half_width=self._config.band_half_width,
        )

    def normalized_overrides(self, input: TensorMapping) -> dict[str, torch.Tensor]:
        """
        Normalized-input entries to substitute, keyed by input name.

        Args:
            input: Denormalized step inputs.
        """
        normalized = (self.flux(input) - self._config.mean) / self._config.std
        return {self._config.co2_name: normalized}
