import dataclasses
import datetime
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import dacite
import torch

import fme
from fme.core.atmosphere_data import (
    AtmosphereData,
    HasAtmosphereVerticalIntegral,
    compute_layer_thickness,
)
from fme.core.constants import GRAVITY, SPECIFIC_HEAT_OF_DRY_AIR_CONST_VOLUME
from fme.core.corrector.registry import CorrectorABC
from fme.core.corrector.utils import force_positive
from fme.core.gridded_ops import GriddedOperations
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class EnergyBudgetConfig:
    """Configuration for total energy budget correction.

    Parameters:
        method: Method to use for total energy budget correction. The available option
            is "constant_temperature", which enforces conservation of total energy by
            imposing a vertically and horizontally uniform air temperature correction.
        constant_unaccounted_heating: Column-integrated heating in W/m**2 to be added
            to the energy flux into the atmosphere when conserving total energy.
            This can be useful for correcting errors in energy budget in target data.
            The same additional heating is imposed at all time steps and grid cells.
    """

    method: Literal["constant_temperature"]
    constant_unaccounted_heating: float = 0.0


@CorrectorSelector.register("atmosphere_corrector")
@dataclasses.dataclass
class AtmosphereCorrectorConfig:
    r"""
    Configuration for the post-step state corrector.

    ``conserve_dry_air`` enforces the constraint that:

    .. math::

        global\_dry\_air = global\_mean(ps -
        sum_k((ak\_diff + bk\_diff \* ps) \* wat_k))

    in the generated data is equal to its value in the input data. This is done
    by adding a globally-constant correction to the surface pressure in each
    column. As per-mass values such as mixing ratios of water are unchanged,
    this can cause changes in total water or energy. Note all global means here
    are area-weighted.

    ``zero_global_mean_moisture_advection`` enforces the constraint that:

    .. math::

        global\_mean(tendency\_of\_total\_water\_path\_due\_to\_advection) = 0

    in the generated data. This is done by adding a globally-constant correction
    to the moisture advection tendency in each column.

    ``moisture_budget_correction`` enforces closure of the moisture budget equation:

    .. math::

        tendency\_of\_total\_water\_path = (evaporation\_rate - precipitation\_rate
        \\\\ + tendency\_of\_total\_water\_path\_due\_to\_advection)

    in the generated data, where ``tendency_of_total_water_path`` is the difference
    between the total water path at the current timestep and the previous
    timestep divided by the time difference. This is done by modifying the
    precipitation, evaporation, and/or moisture advection tendency fields as
    described in the ``moisture_budget_correction`` attribute. When
    advection tendency is modified, this budget equation is enforced in each
    column, while when only precipitation or evaporation are modified, only
    the global mean of the budget equation is enforced.

    When enforcing moisture budget closure, we assume the global mean moisture
    advection is zero. Therefore ``zero_global_mean_moisture_advection`` must be
    True if using a ``moisture_budget_correction`` option other than ``None``.

    Parameters:
        conserve_dry_air: If True, force the generated data to conserve dry air
            by subtracting a constant offset from the surface pressure of each
            column. This can cause changes in per-mass values such as total water
            or energy.
        zero_global_mean_moisture_advection: If True, force the generated data to
            have zero global mean moisture advection by subtracting a constant
            offset from the moisture advection tendency of each column.
        moisture_budget_correction: If not "None", force the generated data to
            conserve global or column-local moisture by modifying budget fields.
            Options are:

            - ``precipitation``: multiply precipitation by a scale factor
              to close the global moisture budget.
            - ``evaporation``: multiply evaporation by a scale factor
              to close the global moisture budget.
            - ``advection_and_precipitation``: after applying the "precipitation"
              global-mean correction above, recompute the column-integrated
              advective tendency as the budget residual,
              ensuring column budget closure.
            - ``advection_and_evaporation``: after applying the "evaporation"
              global-mean correction above, recompute the column-integrated
              advective tendency as the budget residual,
              ensuring column budget closure.

        force_positive_names: Names of fields that should be forced to be greater
            than or equal to zero. This is useful for fields like precipitation.
        total_energy_budget_correction: If not None, force the generated data to
            conserve an idealized version of total energy using the provided
            configuration.
    """

    conserve_dry_air: bool = False
    zero_global_mean_moisture_advection: bool = False
    moisture_budget_correction: (
        Literal[
            "precipitation",
            "evaporation",
            "advection_and_precipitation",
            "advection_and_evaporation",
        ]
        | None
    ) = None
    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    total_energy_budget_correction: EnergyBudgetConfig | None = None

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "AtmosphereCorrectorConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )


class AtmosphereCorrector(CorrectorABC):
    def __init__(
        self,
        config: AtmosphereCorrectorConfig,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasAtmosphereVerticalIntegral | None,
        timestep: datetime.timedelta,
    ):
        self._config = config
        self._gridded_operations = gridded_operations
        self._vertical_coordinate = vertical_coordinate

        self._timestep_seconds = timestep.total_seconds()
        if fme.get_device() == torch.device("mps", 0):
            self._dry_air_precision = torch.float32
        else:
            self._dry_air_precision = torch.float64

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorDict:
        """Apply corrections to the generated data.

        Args:
            input_data: The input time step data.
            gen_data: The data generated by the model, to be corrected.
            forcing_data: The forcing data for the same time step as gen_data.

        Returns:
            The corrected data.
        """
        gen_data = dict(gen_data)
        if len(self._config.force_positive_names) > 0:
            # do this step before imposing other conservation correctors, since
            # otherwise it could end up creating violations of those constraints.
            gen_data = force_positive(gen_data, self._config.force_positive_names)
        if self._config.conserve_dry_air:
            if self._vertical_coordinate is None:
                raise ValueError(
                    "conserve_dry_air is set to True, but no vertical coordinate is "
                    "available."
                )
            gen_data = _force_conserve_dry_air(
                input_data=input_data,
                gen_data=gen_data,
                area_weighted_mean=self._gridded_operations.area_weighted_mean,
                vertical_coordinate=self._vertical_coordinate,
                precision=self._dry_air_precision,
            )
        if self._config.zero_global_mean_moisture_advection:
            gen_data = _force_zero_global_mean_moisture_advection(
                gen_data=gen_data,
                area_weighted_mean=self._gridded_operations.area_weighted_mean,
            )
        if self._config.moisture_budget_correction is not None:
            if self._vertical_coordinate is None:
                raise ValueError(
                    "Moisture budget correction is turned on, but no vertical "
                    "coordinate is available."
                )
            gen_data = _force_conserve_moisture(
                input_data=input_data,
                gen_data=gen_data,
                area_weighted_mean=self._gridded_operations.area_weighted_mean,
                vertical_coordinate=self._vertical_coordinate,
                timestep_seconds=self._timestep_seconds,
                terms_to_modify=self._config.moisture_budget_correction,
            )
        if self._config.total_energy_budget_correction is not None:
            if self._vertical_coordinate is None:
                raise ValueError(
                    "Energy budget correction is turned on, but no vertical coordinate"
                    " is available."
                )
            gen_data = _force_conserve_total_energy(
                input_data=input_data,
                gen_data=gen_data,
                forcing_data=forcing_data,
                area_weighted_mean=self._gridded_operations.area_weighted_mean,
                vertical_coordinate=self._vertical_coordinate,
                timestep_seconds=self._timestep_seconds,
                method=self._config.total_energy_budget_correction.method,
                unaccounted_heating=self._config.total_energy_budget_correction.constant_unaccounted_heating,
            )
        return gen_data


class AreaWeightedMean(Protocol):
    def __call__(
        self, data: torch.Tensor, keepdim: bool, name: str | None = None
    ) -> torch.Tensor: ...


def _force_conserve_dry_air(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasAtmosphereVerticalIntegral,
    precision: torch.dtype = torch.float64,
) -> TensorDict:
    """
    Update the generated data to conserve dry air.

    This is done by adding a constant correction to the dry air pressure of
    each column, and may result in changes in per-mass values such as
    total water or energy.

    We first compute the target dry air pressure by computing the globally
    averaged difference in dry air pressure between the input_data and gen_data,
    and then add this offset to the fully-resolved gen_data dry air pressure.
    We can then solve for the surface pressure corresponding to this new dry air
    pressure.

    We start from the expression for dry air pressure:

        dry_air = ps - sum_k((ak_diff + bk_diff * ps) * wat_k)

    To update the dry air, we compute and update the surface pressure:

        ps = (
            dry_air + sum_k(ak_diff * wat_k)
        ) / (
            1 - sum_k(bk_diff * wat_k)
        )
    """
    input = AtmosphereData(input_data, vertical_coordinate)
    if input.surface_pressure is None:
        raise ValueError("surface_pressure is required to force dry air conservation")
    gen = AtmosphereData(gen_data, vertical_coordinate)
    gen_dry_air = gen.surface_pressure_due_to_dry_air
    global_gen_dry_air = area_weighted_mean(gen_dry_air.to(precision), keepdim=True)
    global_target_gen_dry_air = area_weighted_mean(
        input.surface_pressure_due_to_dry_air.to(precision),
        keepdim=True,
    )
    error = global_gen_dry_air - global_target_gen_dry_air
    new_gen_dry_air = gen_dry_air.to(precision) - error
    try:
        wat = gen.specific_total_water.to(precision)
    except KeyError:
        raise ValueError("specific_total_water is required for conservation")
    ak_diff = vertical_coordinate.get_ak().diff().to(precision)
    bk_diff = vertical_coordinate.get_bk().diff().to(precision)
    new_pressure = (new_gen_dry_air + (ak_diff * wat).sum(-1)) / (
        1 - (bk_diff * wat).sum(-1)
    )
    gen.set_surface_pressure(new_pressure.to(dtype=input.surface_pressure.dtype))
    return gen.data


def _force_zero_global_mean_moisture_advection(
    gen_data: TensorMapping,
    area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
) -> TensorDict:
    """
    Update the generated data so advection conserves moisture.

    Does so by adding a constant offset to the moisture advective tendency.

    Args:
        gen_data: The generated data.
        area_weighted_mean: Computes an area-weighted mean,
            removing horizontal dimensions.
    """
    gen = AtmosphereData(gen_data)

    mean_moisture_advection = area_weighted_mean(
        gen.tendency_of_total_water_path_due_to_advection,
    )
    gen.set_tendency_of_total_water_path_due_to_advection(
        gen.tendency_of_total_water_path_due_to_advection
        - mean_moisture_advection[..., None, None]
    )
    return gen.data


def _force_conserve_moisture(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasAtmosphereVerticalIntegral,
    timestep_seconds: float,
    terms_to_modify: Literal[
        "precipitation",
        "evaporation",
        "advection_and_precipitation",
        "advection_and_evaporation",
    ],
) -> TensorDict:
    """
    Update the generated data to conserve moisture.

    Does so while conserving total dry air in each column.

    Assumes the global mean advective tendency of moisture is zero. This assumption
    means any existing global mean advective tendency will be set to zero
    if the advective tendency is re-computed.

    Args:
        input_data: The input data.
        gen_data: The generated data one timestep after the input data.
        area_weighted_mean: Computes an area-weighted mean,
            removing horizontal dimensions.
        vertical_coordinate: The sigma coordinates.
        timestep_seconds: Timestep of the model in seconds.
        terms_to_modify: Which terms to modify, in addition to modifying surface
            pressure to conserve dry air mass. One of:
            - "precipitation": modify precipitation only
            - "evaporation": modify evaporation only
            - "advection_and_precipitation": modify advection and precipitation
            - "advection_and_evaporation": modify advection and evaporation
    """
    input = AtmosphereData(input_data, vertical_coordinate)
    gen = AtmosphereData(gen_data, vertical_coordinate)

    gen_total_water_path = gen.total_water_path
    twp_total_tendency = (
        gen_total_water_path - input.total_water_path
    ) / timestep_seconds
    twp_tendency_global_mean = area_weighted_mean(twp_total_tendency, keepdim=True)
    evaporation_global_mean = area_weighted_mean(gen.evaporation_rate, keepdim=True)
    precipitation_global_mean = area_weighted_mean(gen.precipitation_rate, keepdim=True)
    if terms_to_modify.endswith("precipitation"):
        # We want to achieve
        #     global_mean(twp_total_tendency) = (
        #         global_mean(evaporation_rate)
        #         - global_mean(precipitation_rate)
        #     )
        # so we modify precipitation_rate to achieve this. Note we have
        # assumed the global mean advection tendency is zero.
        # First, we find the required global-mean precipitation rate
        #     new_global_precip_rate = (
        #         global_mean(evaporation_rate)
        #         - global_mean(twp_total_tendency)
        #     )
        new_precipitation_global_mean = (
            evaporation_global_mean - twp_tendency_global_mean
        )
        # Because scalar multiplication commutes with summation, we can
        # achieve this by multiplying each gridcell's precipitation rate
        # by the ratio of the new global mean to the current global mean.
        #    new_precip_rate = (
        #        new_global_precip_rate / current_global_precip_rate
        #    ) * current_precip_rate
        gen.set_precipitation_rate(
            gen.precipitation_rate
            * (new_precipitation_global_mean / precipitation_global_mean)
        )
    elif terms_to_modify.endswith("evaporation"):
        # Derived similarly as for "precipitation" case.
        new_evaporation_global_mean = (
            twp_tendency_global_mean + precipitation_global_mean
        )
        gen.set_evaporation_rate(
            gen.evaporation_rate
            * (new_evaporation_global_mean / evaporation_global_mean)
        )
    if terms_to_modify.startswith("advection"):
        # Having already corrected the global-mean budget, we recompute
        # advection based on assumption that the columnwise
        # moisture budget closes. Correcting the global mean budget first
        # is important to ensure the resulting advection has zero global mean.
        new_advection = twp_total_tendency - (
            gen.evaporation_rate - gen.precipitation_rate
        )
        gen.set_tendency_of_total_water_path_due_to_advection(new_advection)
    return gen.data


def _force_conserve_total_energy(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasAtmosphereVerticalIntegral,
    timestep_seconds: float,
    method: Literal["constant_temperature"] = "constant_temperature",
    unaccounted_heating: float = 0.0,
) -> TensorDict:
    """Apply a correction to the generated data to conserve total energy.

    This function also inserts the unaccounted heating into the generated data.
    """
    if method != "constant_temperature":
        raise NotImplementedError(
            f"Method {method} not implemented for total energy conservation"
        )
    input = AtmosphereData(input_data, vertical_coordinate)
    forcing = AtmosphereData(forcing_data)
    required_forcing = {
        "DSWRFtoa": forcing.toa_down_sw_radiative_flux,
        "HGTsfc": forcing.surface_height,
    }
    atmosphere_data = dict(gen_data)
    for name, tensor in required_forcing.items():
        atmosphere_data[name] = tensor
    gen = AtmosphereData(atmosphere_data, vertical_coordinate)

    gen_energy_path = gen.total_energy_ace2_path
    input_energy_path = input.total_energy_ace2_path
    predicted_energy_flux_into_atmosphere = gen.net_energy_flux_into_atmosphere

    gen_energy_path_global_mean = area_weighted_mean(gen_energy_path, keepdim=True)
    input_energy_path_global_mean = area_weighted_mean(input_energy_path, keepdim=True)
    energy_flux_global_mean = area_weighted_mean(
        predicted_energy_flux_into_atmosphere, keepdim=True
    )

    desired_energy_path_global_mean = (
        input_energy_path_global_mean
        + (energy_flux_global_mean + unaccounted_heating) * timestep_seconds
    )

    energy_correction = desired_energy_path_global_mean - gen_energy_path_global_mean
    energy_to_temperature_factor = _energy_correction_factor(gen, vertical_coordinate)
    # take global mean to impose a spatially uniform temperature correction
    energy_to_temp_factor_gm = area_weighted_mean(energy_to_temperature_factor, True)
    temperature_correction = energy_correction / energy_to_temp_factor_gm

    # apply same temperature correction to all vertical layers
    n_levels = gen.air_temperature.shape[-1]
    for k in range(n_levels):
        name = f"air_temperature_{k}"
        gen.data[name] = gen.data[name] + torch.nan_to_num(
            temperature_correction, nan=0.0
        )

    # filter required here because we merged forcing data into gen above
    return {k: v for k, v in gen.data.items() if k in gen_data}


def _energy_correction_factor(
    gen: AtmosphereData, vertical_coordinate: HasAtmosphereVerticalIntegral
) -> torch.Tensor:
    """
    Compute the factor to get a vertically-uniform temperature correction that
    will lead to a desired change in the globally-averaged total energy.

    See https://www.overleaf.com/read/dqjjcvzxnfvn#d525aa.
    """
    interface_pressure = vertical_coordinate.interface_pressure(gen.surface_pressure)
    q_times_dlogp = (
        compute_layer_thickness(
            interface_pressure, gen.air_temperature, gen.specific_total_water
        )
        * GRAVITY
        / gen.air_temperature
    )
    cumulative = torch.cumsum(q_times_dlogp.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
    total_integrand = (
        SPECIFIC_HEAT_OF_DRY_AIR_CONST_VOLUME - 0.5 * q_times_dlogp + cumulative
    )
    correction_factor = vertical_coordinate.vertical_integral(
        total_integrand, gen.surface_pressure
    )
    return correction_factor
