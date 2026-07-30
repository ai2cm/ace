import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import torch

from fme.core.atmosphere_data import AtmosphereData
from fme.core.constants import (
    FREEZING_TEMPERATURE_KELVIN,
    LATENT_HEAT_OF_VAPORIZATION,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.corrector.registry import (
    Correction,
    CorrectionSequence,
    CorrectorConfigABC,
)
from fme.core.corrector.state import CorrectorState
from fme.core.corrector.utils import ForcePositive, replace_value_keep_gradient
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import GriddedOperations
from fme.core.ocean_data import HasOceanDepthIntegral, OceanData
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorDict, TensorMapping


class AreaWeightedMean(Protocol):
    def __call__(
        self, data: torch.Tensor, keepdim: bool = False, name: str | None = None
    ) -> torch.Tensor: ...


@dataclasses.dataclass
class SeaIceFractionConfig:
    """Correct predicted sea_ice_fraction to ensure it is always in 0-1, and
    land_fraction + sea_ice_fraction + ocean_fraction = 1. After
    sea_ice_fraction is corrected, all variables listed in
    zero_where_ice_free_names will be set to 0 everywhere
    sea_ice_fraction is 0.

    Parameters:
        sea_ice_fraction_name: Name of the sea ice fraction variable.
        land_fraction_name: Name of the land fraction variable.
        zero_where_ice_free_names: List of variable names to set to 0
            wherever sea_ice_fraction is 0.
        remove_negative_ocean_fraction: If True, reduce sea_ice_fraction
            to prevent ocean_fraction (1 - sea_ice_fraction - land_fraction)
            from being negative.
    """

    sea_ice_fraction_name: str
    land_fraction_name: str
    zero_where_ice_free_names: list[str] = dataclasses.field(default_factory=list)
    remove_negative_ocean_fraction: bool = True

    def __call__(
        self,
        gen_data: TensorMapping,
        input_data: TensorMapping,
        keep_gradient: bool = False,
    ) -> TensorDict:
        """
        Returns:
            A ``TensorDict`` containing only the fields modified by this
            correction (the sea ice fraction and the fields zeroed where
            ice-free).
        """
        out: TensorDict = {}
        sif = gen_data[self.sea_ice_fraction_name]
        clamped_sif = torch.clamp(sif, min=0.0, max=1.0)
        if keep_gradient:
            clamped_sif = replace_value_keep_gradient(sif, clamped_sif)
        out[self.sea_ice_fraction_name] = clamped_sif
        if self.remove_negative_ocean_fraction:
            negative_ocean_fraction = (
                1
                - out[self.sea_ice_fraction_name]
                - input_data[self.land_fraction_name]
            )
            negative_ocean_fraction = negative_ocean_fraction.clip(max=0)
            rebalanced_sif = out[self.sea_ice_fraction_name] + negative_ocean_fraction
            if keep_gradient:
                rebalanced_sif = replace_value_keep_gradient(
                    out[self.sea_ice_fraction_name], rebalanced_sif
                )
            out[self.sea_ice_fraction_name] = rebalanced_sif
        for name in self.zero_where_ice_free_names:
            out[name] = gen_data[name] * (out[self.sea_ice_fraction_name] > 0.0)
        return out


@dataclasses.dataclass
class OceanHeatContentBudgetConfig:
    """Configuration for ocean heat content budget correction.

    Parameters:
        method: Method to use for OHC budget correction. The available option is
            "scaled_temperature", which enforces conservation of heat content
            by scaling the predicted potential temperature by a vertically and
            horizontally uniform correction factor.
        constant_unaccounted_heating: Area-weighted global mean
            column-integrated heating in W/m**2 to be added to the energy flux
            into the ocean when conserving the heat content. This can be useful
            for correcting errors in heat budget in target data. The same
            additional heating is imposed at all time steps and grid cells.

    """

    method: Literal["scaled_temperature"]
    constant_unaccounted_heating: float = 0.0


@dataclasses.dataclass
class SurfaceEnergyFluxCorrectionConfig:
    """Configuration for correcting the generated hfds using
    atmosphere-derived surface energy fluxes and ocean_fraction.

    The net_flux is the net surface energy flux computed from atmospheric
    forcing variables and generated SST. The ocean_fraction naturally zeroes
    out the correction on land and reduces it under sea ice.

    Available options are:
      - "residual_prediction": corrected_hfds = gen_hfds + ocean_fraction * net_flux.
        The network predicts a residual that is added to the forcing-derived flux.
      - "prescribed": corrected_hfds = net_flux * ocean_fraction + gen_hfds *
        (1 - ocean_fraction). Open-ocean hfds is prescribed from forcings; the
        network prediction is retained under sea ice and on land.

    Parameters:
        method: Method to use for the correction.

    """

    method: Literal["residual_prediction", "prescribed"]


@dataclasses.dataclass
class SeaIceFractionCorrection:
    """Correction that enforces sea-ice-fraction constraints.

    Wraps ``SeaIceFractionConfig`` so the corrector applies the operation
    without reading config fields. ``forcing_data`` and ``corrector_state`` are
    unused and passed through.

    If ``keep_gradient`` is True, the clamp and rebalance are applied with a
    straight-through estimator so out-of-range cells still get a learning signal.
    """

    config: SeaIceFractionConfig
    keep_gradient: bool = False

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only the fields modified by
            this correction (the sea ice fraction and the fields zeroed where
            ice-free). ``SeaIceFractionConfig.__call__`` already returns only
            those fields, preserving the straight-through estimator when
            ``keep_gradient`` is set.
        """
        corrected = self.config(gen_data, input_data, keep_gradient=self.keep_gradient)
        return corrected, corrector_state


@dataclasses.dataclass
class SurfaceEnergyFluxCorrection:
    """Correction that adjusts hfds using atmosphere-derived surface fluxes."""

    method: Literal["residual_prediction", "prescribed"]

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only the field modified by this
            correction (the net downward surface heat flux, ``hfds``).
        """
        corrected = _correct_hfds(
            input_data,
            gen_data,
            forcing_data,
            method=self.method,
        )
        return corrected, corrector_state


@dataclasses.dataclass
class OceanHeatContentCorrection:
    """Correction that conserves ocean heat content."""

    area_weighted_mean: AreaWeightedMean
    vertical_coordinate: HasOceanDepthIntegral | None
    timestep_seconds: float
    method: Literal["scaled_temperature"]
    unaccounted_heating: float

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only the fields modified by
            this correction (the potential temperature at every depth level, and
            the sea surface temperature when present).
        """
        if self.vertical_coordinate is None:
            raise ValueError(
                "Ocean heat content correction is turned on, but no vertical "
                "coordinate is available."
            )
        corrected = _force_conserve_ocean_heat_content(
            input_data,
            gen_data,
            forcing_data,
            self.area_weighted_mean,
            self.vertical_coordinate,
            self.timestep_seconds,
            self.method,
            self.unaccounted_heating,
        )
        return corrected, corrector_state


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig(CorrectorConfigABC):
    """Configuration for corrections applied to generated ocean data.

    Parameters:
        force_positive_names: Names of fields that should be forced to be greater
            than or equal to zero.
        sea_ice_fraction_correction: Optional configuration for a sea-ice-fraction
            correction (bounds sea_ice_fraction to 0-1 and keeps the land, ocean,
            and sea-ice fractions summing to one).
        surface_energy_flux_correction: Optional configuration for a surface energy
            flux correction to the generated hfds.
        ocean_heat_content_correction: Optional configuration for an ocean heat
            content correction.
        keep_gradient_through_clamps: If True, apply the corrector's hard clamps
            (the ``force_positive_names`` clamp and the
            ``sea_ice_fraction_correction`` bound/rebalance) with a straight-through
            estimator: the forward value is still clamped, but gradient flows as if
            the clamp had not happened, so out-of-range cells still get a learning
            signal.
    """

    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: SeaIceFractionConfig | None = None
    surface_energy_flux_correction: SurfaceEnergyFluxCorrectionConfig | None = None
    ocean_heat_content_correction: OceanHeatContentBudgetConfig | None = None
    keep_gradient_through_clamps: bool = False

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        state_copy = dict(state)
        if "masking" in state_copy:
            del state_copy["masking"]
        if "ocean_heat_content_correction" in state_copy and isinstance(
            state_copy["ocean_heat_content_correction"], bool
        ):
            if state_copy["ocean_heat_content_correction"]:
                state_copy["ocean_heat_content_correction"] = (
                    OceanHeatContentBudgetConfig(method="scaled_temperature")
                )
            else:
                state_copy["ocean_heat_content_correction"] = None
        if "sea_ice_fraction_correction" in state_copy:
            sif = state_copy["sea_ice_fraction_correction"]
            if isinstance(sif, dict) and "sea_ice_thickness_name" in sif:
                thickness_name = sif.pop("sea_ice_thickness_name")
                if thickness_name is not None:
                    sif.setdefault("zero_where_ice_free_names", []).append(
                        thickness_name
                    )
        return state_copy

    def _get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "OceanCorrector":
        return self._build(
            dataset_info.gridded_operations,
            dataset_info.ocean_vertical_coordinate,
            dataset_info.timestep,
        )

    def _build(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasOceanDepthIntegral | None,
        timestep: datetime.timedelta,
    ) -> "OceanCorrector":
        area_weighted_mean = gridded_operations.area_weighted_mean
        timestep_seconds = timestep.total_seconds()
        corrections: list[Correction] = []
        if len(self.force_positive_names) > 0:
            corrections.append(
                ForcePositive(
                    self.force_positive_names,
                    keep_gradient=self.keep_gradient_through_clamps,
                )
            )
        if self.sea_ice_fraction_correction is not None:
            corrections.append(
                SeaIceFractionCorrection(
                    self.sea_ice_fraction_correction,
                    keep_gradient=self.keep_gradient_through_clamps,
                )
            )
        if self.surface_energy_flux_correction is not None:
            corrections.append(
                SurfaceEnergyFluxCorrection(self.surface_energy_flux_correction.method)
            )
        if self.ocean_heat_content_correction is not None:
            corrections.append(
                OceanHeatContentCorrection(
                    area_weighted_mean,
                    vertical_coordinate,
                    timestep_seconds,
                    self.ocean_heat_content_correction.method,
                    self.ocean_heat_content_correction.constant_unaccounted_heating,
                )
            )
        return OceanCorrector(corrections)


class OceanCorrector(CorrectionSequence):
    pass


def _compute_ocean_net_surface_energy_flux(
    forcing_data: TensorMapping,
    sst: torch.Tensor,
) -> torch.Tensor:
    """Compute the net surface energy flux into the ocean from atmospheric
    forcing variables and the sea surface temperature.

    This extends the atmosphere net surface energy flux with SST-dependent
    heat transport by precipitation and evaporation.
    """
    atmos = AtmosphereData(forcing_data)
    base_flux = (
        atmos.net_surface_energy_flux
    )  # missing: - calving * LATENT_HEAT_OF_FREEZING
    mass_heat_flux = (
        SPECIFIC_HEAT_OF_SEA_WATER_CM4
        * (
            atmos.precipitation_rate
            + atmos.frozen_precipitation_rate
            - (atmos.latent_heat_flux / LATENT_HEAT_OF_VAPORIZATION)
        )  # missing: + river runoff + calving
        * (sst - FREEZING_TEMPERATURE_KELVIN)
    )
    return base_flux + mass_heat_flux


def _correct_hfds(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    method: Literal["residual_prediction", "prescribed"],
) -> TensorDict:
    """Apply surface energy flux correction to the generated hfds.

    The ocean_fraction naturally zeroes the correction on land and reduces
    it under sea ice.

    Methods:
        residual_prediction: gen_hfds + ocean_fraction * net_flux
        prescribed: net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
    """
    input = OceanData(input_data)
    forcing = OceanData(forcing_data)
    ocean_fraction = input.ocean_fraction
    net_flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, input.sea_surface_temperature
    )
    out: TensorDict = {}
    if "hfds" in gen_data:
        hfds_name = "hfds"
    else:
        hfds_name = "hfds_total_area"
        net_flux = net_flux * forcing.sea_surface_fraction
    gen_hfds = gen_data[hfds_name]
    if method == "residual_prediction":
        out[hfds_name] = net_flux * ocean_fraction + gen_hfds
    elif method == "prescribed":
        out[hfds_name] = net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
    else:
        raise NotImplementedError(
            f"Method {method!r} not implemented for surface energy flux correction"
        )
    return out


def _force_conserve_ocean_heat_content(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
    method: Literal["scaled_temperature"] = "scaled_temperature",
    unaccounted_heating: float = 0.0,
) -> TensorDict:
    if method != "scaled_temperature":
        raise NotImplementedError(
            f"Method {method!r} not implemented for ocean heat content conservation"
        )
    if "hfds" in gen_data and "hfds" in forcing_data:
        raise ValueError(
            "Net downward surface heat flux cannot be present in both gen_data and "
            "forcing_data."
        )
    input = OceanData(input_data, vertical_coordinate)
    if input.ocean_heat_content is None:
        raise ValueError(
            "ocean_heat_content is required to force ocean heat content conservation"
        )
    gen = OceanData(gen_data, vertical_coordinate)
    forcing = OceanData(forcing_data)
    global_gen_ocean_heat_content = area_weighted_mean(
        gen.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    global_input_ocean_heat_content = area_weighted_mean(
        input.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    try:
        # First priority: pre-weighted heat flux in gen_data
        net_energy_flux_into_ocean = (
            gen.net_downward_surface_heat_flux_total_area
            + forcing.geothermal_heat_flux * forcing.sea_surface_fraction
        )
    except KeyError:
        try:
            # Second priority: standard heat flux in gen_data
            net_energy_flux_into_ocean = (
                gen.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction
        except KeyError:
            # Third priority: standard heat flux in input_data
            net_energy_flux_into_ocean = (
                input.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction
    energy_flux_global_mean = area_weighted_mean(
        net_energy_flux_into_ocean,
        keepdim=True,
        name="ocean_heat_content",
    )
    expected_change_ocean_heat_content = (
        energy_flux_global_mean + unaccounted_heating
    ) * timestep_seconds
    heat_content_correction_ratio = (
        global_input_ocean_heat_content + expected_change_ocean_heat_content
    ) / global_gen_ocean_heat_content
    # apply same temperature correction to all vertical layers
    out: TensorDict = {}
    n_levels = gen.sea_water_potential_temperature.shape[-1]
    for k in range(n_levels):
        name = f"thetao_{k}"
        out[name] = gen.data[name] * heat_content_correction_ratio
    if "sst" in gen.data:
        out["sst"] = (  # assuming sst in Kelvin
            gen.data["sst"] - FREEZING_TEMPERATURE_KELVIN
        ) * heat_content_correction_ratio + FREEZING_TEMPERATURE_KELVIN
    return out
