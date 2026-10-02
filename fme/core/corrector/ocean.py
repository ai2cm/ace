import dataclasses
import datetime
import functools
import warnings
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import torch

from fme.core.atmosphere_data import AtmosphereData
from fme.core.constants import (
    DENSITY_OF_SEA_ICE,
    DENSITY_OF_SEA_WATER_CM4,
    FREEZING_TEMPERATURE_KELVIN,
    LATENT_HEAT_OF_VAPORIZATION,
    REFERENCE_SALINITY_PSU,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
    SPHERE_AREA_M2,
)
from fme.core.corrector.registry import (
    Correction,
    CorrectionSequence,
    CorrectorConfigABC,
)
from fme.core.corrector.state import CorrectorState
from fme.core.corrector.utils import ForcePositive, replace_value_keep_gradient
from fme.core.dataset_info import DatasetInfo, MissingDatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.ocean_data import (
    OCEAN_FIELD_NAME_PREFIXES,
    HasOceanDepthIntegral,
    OceanData,
)
from fme.core.registry.corrector import CorrectorSelector
from fme.core.spatial_mask_provider import (
    NullSpatialMaskProvider,
    SpatialMaskProviderABC,
)
from fme.core.typing_ import TensorDict, TensorMapping


class AreaWeightedMean(Protocol):
    def __call__(
        self, data: torch.Tensor, keepdim: bool = False, name: str | None = None
    ) -> torch.Tensor: ...


class AreaWeightedSum(Protocol):
    def __call__(
        self, data: torch.Tensor, keepdim: bool = False, name: str | None = None
    ) -> torch.Tensor: ...


GlobalTotal = Callable[[torch.Tensor], torch.Tensor]
"""Sum of a per-cell field times the cell area over the ocean, keeping the
horizontal dimensions."""


class SaltBudget(Protocol):
    """Expected change of the total ocean salt content over one step, in
    psu m**3, keeping the horizontal dimensions.
    """

    def __call__(
        self,
        input: OceanData,
        gen: OceanData,
        forcing: OceanData,
        global_total: GlobalTotal,
        timestep_seconds: float,
        dtype: torch.dtype,
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


def _sea_ice_volume_output_mask(
    spatial_mask_provider: SpatialMaskProviderABC,
) -> torch.Tensor | None:
    """Cells where the stepper keeps the sea_ice_volume prediction, or None for
    every cell.
    """
    mask = spatial_mask_provider.get_mask_tensor_for("sea_ice_volume")
    if mask is None:
        return None
    # rounded as the stepper's output masking does, so the two agree
    return torch.round(mask.to(get_device())) != 0


def _require_sea_surface_fraction_weighting(
    budget_type: str, weight_by_sea_surface_fraction: bool
) -> None:
    if not weight_by_sea_surface_fraction:
        raise ValueError(
            f"The {budget_type!r} salt budget requires "
            "weight_by_sea_surface_fraction=True."
        )


@dataclasses.dataclass
class IceVolumeSaltBudgetConfig:
    """Salt budget from the change of the total sea ice volume.

    Parameters:
        slope_psu: Change in total salt content (psu m**3) per change in total
            sea ice volume (m**3).
        type: Selects this budget.
    """

    slope_psu: float
    type: Literal["ice_volume"] = "ice_volume"

    def validate(self, weight_by_sea_surface_fraction: bool) -> None:
        pass

    def build(self, spatial_mask_provider: SpatialMaskProviderABC) -> SaltBudget:
        Distributed.get_instance().require_no_spatial_parallelism(
            "The ice volume salt budget sums sea_ice_volume over the local "
            "spatial chunk only."
        )
        return IceVolumeSaltBudget(
            self.slope_psu, _sea_ice_volume_output_mask(spatial_mask_provider)
        )


@dataclasses.dataclass
class SeaSurfaceHeightSaltBudgetConfig:
    """Salt budget from the change of the sea surface height. Requires ``SSH``,
    the height including its global mean, in the inputs and outputs.

    Parameters:
        reference_salinity_psu: Salinity at which the added water dilutes the
            salt, in psu.
        type: Selects this budget.
    """

    reference_salinity_psu: float = REFERENCE_SALINITY_PSU
    type: Literal["sea_surface_height"] = "sea_surface_height"

    def validate(self, weight_by_sea_surface_fraction: bool) -> None:
        _require_sea_surface_fraction_weighting(
            self.type, weight_by_sea_surface_fraction
        )

    def build(self, spatial_mask_provider: SpatialMaskProviderABC) -> SaltBudget:
        return SeaSurfaceHeightSaltBudget(self.reference_salinity_psu)


@dataclasses.dataclass
class WaterFluxRegimesConfig:
    """Regimes of the water flux salt budget, from the input and generated
    states.

    Parameters:
        sea_ice_fraction_threshold: Cells whose sea ice fraction exceeds this at
            either step are ice covered.
        coast_sea_surface_fraction_threshold: Cells that are not ice covered and
            whose sea surface fraction is below this are coast; the rest is
            open water.
    """

    sea_ice_fraction_threshold: float = 0.0
    coast_sea_surface_fraction_threshold: float = 1.0


@dataclasses.dataclass
class WaterFluxSaltBudgetConfig:
    """Salt budget from the surface water flux and the sea ice basal salt
    flux, (DT / rho_0) * sum((-S_ref * wfo + 1000 * sfdsi) * ssf * A), with
    the enabled terms in place of wfo and sfdsi in their regimes. wfo and
    sfdsi are read from the generated data, or from the forcing data if they
    are not predicted.

    Parameters:
        open_water_precipitation_minus_evaporation: Use the forcing
            precipitation minus evaporation in place of wfo over open water.
        sea_ice_mass: Use -rho_ice * delta(sea_ice_volume) / DT in place of
            wfo where ice covered.
        sea_ice_salinity_psu: With ``sea_ice_mass``, use
            -rho_ice * S_ice * delta(sea_ice_volume) / DT in place of sfdsi
            where ice covered. None keeps sfdsi there.
        brine_rejection: Include the sea ice salt flux.
        reference_salinity_psu: Reference salinity of the virtual salt flux,
            in psu.
        sea_ice_density_kg_m3: Density of sea ice.
        regimes: Thresholds defining the regimes.
        type: Selects this budget.
    """

    open_water_precipitation_minus_evaporation: bool = False
    sea_ice_mass: bool = False
    sea_ice_salinity_psu: float | None = None
    brine_rejection: bool = True
    reference_salinity_psu: float = REFERENCE_SALINITY_PSU
    sea_ice_density_kg_m3: float = DENSITY_OF_SEA_ICE
    regimes: WaterFluxRegimesConfig = dataclasses.field(
        default_factory=WaterFluxRegimesConfig
    )
    type: Literal["water_flux"] = "water_flux"

    def validate(self, weight_by_sea_surface_fraction: bool) -> None:
        _require_sea_surface_fraction_weighting(
            self.type, weight_by_sea_surface_fraction
        )
        if self.sea_ice_salinity_psu is not None and not self.sea_ice_mass:
            raise ValueError("sea_ice_salinity_psu requires sea_ice_mass.")
        if self.sea_ice_salinity_psu is not None and not self.brine_rejection:
            raise ValueError("sea_ice_salinity_psu requires brine_rejection.")
        if self.sea_ice_density_kg_m3 <= 0.0:
            raise ValueError(
                "sea_ice_density_kg_m3 must be positive, got "
                f"{self.sea_ice_density_kg_m3}."
            )

    def build(self, spatial_mask_provider: SpatialMaskProviderABC) -> SaltBudget:
        ice_volume_mask = None
        if self.sea_ice_mass:
            Distributed.get_instance().require_no_spatial_parallelism(
                "The sea ice mass term of the water flux salt budget sums "
                "sea_ice_volume over the local spatial chunk only."
            )
            ice_volume_mask = _sea_ice_volume_output_mask(spatial_mask_provider)
        return WaterFluxSaltBudget(self, ice_volume_mask)


SaltBudgetConfig = (
    IceVolumeSaltBudgetConfig
    | SeaSurfaceHeightSaltBudgetConfig
    | WaterFluxSaltBudgetConfig
)


@dataclasses.dataclass
class OceanSaltContentBudgetConfig:
    """Configuration for ocean salt content budget correction.

    Assumes area weights normalized over the whole sphere.

    Parameters:
        method: Method to use for salt budget correction. "scaled_salinity"
            scales the predicted salinity by a vertically and horizontally
            uniform factor.
        budget_config: Budget for the expected change of the salt content
            over a step. None holds the salt content fixed, up to the constant
            term.
        constant_unaccounted_salting: Area-weighted global mean rate of column
            salt content change added at every step, in psu m / s.
        use_float64: Compute the global sums, budget and correction ratio in
            float64 instead of the data's dtype.
        weight_by_sea_surface_fraction: Weight the column salt content, and
            the area the constant term applies over, by the sea surface
            fraction.
    """

    method: Literal["scaled_salinity"]
    budget_config: SaltBudgetConfig | None = None
    constant_unaccounted_salting: float = 0.0
    use_float64: bool = True
    weight_by_sea_surface_fraction: bool = True

    def __post_init__(self):
        # validated here rather than in each budget config's __post_init__,
        # since dacite discards errors raised while matching a union member
        if self.budget_config is not None:
            self.budget_config.validate(self.weight_by_sea_surface_fraction)


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
      - "prescribed_open_ocean": corrected_hfds = net_flux where ocean_fraction
        == 1, and gen_hfds elsewhere. hfds is prescribed from forcings only on
        cells that are entirely ice-free ocean, and the network prediction
        passes through unweighted everywhere else.

    Parameters:
        method: Method to use for the correction.

    """

    method: Literal["residual_prediction", "prescribed", "prescribed_open_ocean"]


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

    method: Literal["residual_prediction", "prescribed", "prescribed_open_ocean"]

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


@dataclasses.dataclass
class OceanSaltContentCorrection:
    """Correction that conserves ocean salt content."""

    area_weighted_sum: AreaWeightedSum
    vertical_coordinate: HasOceanDepthIntegral | None
    timestep_seconds: float
    method: Literal["scaled_salinity"]
    budget: SaltBudget | None
    unaccounted_salting: float
    weight_by_sea_surface_fraction: bool
    use_float64: bool

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
            this correction (the salinity at every depth level).
        """
        if self.vertical_coordinate is None:
            raise ValueError(
                "Ocean salt content correction is turned on, but no vertical "
                "coordinate is available."
            )
        corrected = _force_conserve_ocean_salt_content(
            input_data,
            gen_data,
            forcing_data,
            self.area_weighted_sum,
            self.vertical_coordinate,
            self.timestep_seconds,
            self.method,
            self.budget,
            self.unaccounted_salting,
            self.weight_by_sea_surface_fraction,
            self.use_float64,
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
        ocean_salt_content_correction: Optional configuration for an ocean salt
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
    ocean_salt_content_correction: OceanSaltContentBudgetConfig | None = None
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
        salt = state_copy.get("ocean_salt_content_correction")
        if isinstance(salt, dict) and "ice_volume_salt_slope_psu" in salt:
            warnings.warn(
                "ocean_salt_content_correction.ice_volume_salt_slope_psu is "
                "deprecated; use budget_config with type 'ice_volume'.",
                DeprecationWarning,
            )
            salt = dict(salt)
            slope = salt.pop("ice_volume_salt_slope_psu")
            if slope != 0.0:
                salt["budget_config"] = {"type": "ice_volume", "slope_psu": slope}
            # slopes from before budget_config were fit to the unweighted content
            salt.setdefault("weight_by_sea_surface_fraction", False)
            state_copy["ocean_salt_content_correction"] = salt
        return state_copy

    def _get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "OceanCorrector":
        spatial_mask_provider: SpatialMaskProviderABC
        try:
            spatial_mask_provider = dataset_info.spatial_mask_provider
        except MissingDatasetInfo:
            spatial_mask_provider = NullSpatialMaskProvider
        return self._build(
            dataset_info.gridded_operations,
            dataset_info.ocean_vertical_coordinate,
            dataset_info.timestep,
            spatial_mask_provider=spatial_mask_provider,
        )

    def _build(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasOceanDepthIntegral | None,
        timestep: datetime.timedelta,
        spatial_mask_provider: SpatialMaskProviderABC = NullSpatialMaskProvider,
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
        if self.ocean_salt_content_correction is not None:
            salt_config = self.ocean_salt_content_correction
            corrections.append(
                OceanSaltContentCorrection(
                    gridded_operations.area_weighted_sum,
                    vertical_coordinate,
                    timestep_seconds,
                    salt_config.method,
                    (
                        None
                        if salt_config.budget_config is None
                        else salt_config.budget_config.build(spatial_mask_provider)
                    ),
                    salt_config.constant_unaccounted_salting,
                    salt_config.weight_by_sea_surface_fraction,
                    salt_config.use_float64,
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
    method: Literal["residual_prediction", "prescribed", "prescribed_open_ocean"],
) -> TensorDict:
    """Apply surface energy flux correction to the generated hfds.

    The ocean_fraction naturally zeroes the correction on land and reduces
    it under sea ice.

    Methods:
        residual_prediction: gen_hfds + ocean_fraction * net_flux
        prescribed: net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
        prescribed_open_ocean: net_flux where ocean_fraction == 1, else gen_hfds
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
    elif method == "prescribed_open_ocean":
        out[hfds_name] = torch.where(ocean_fraction == 1, net_flux, gen_hfds)
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


def _total_sea_ice_volume_change(
    input: OceanData,
    gen: OceanData,
    ice_volume_mask: torch.Tensor | None,
    dtype: torch.dtype,
    required_by: str,
) -> torch.Tensor:
    """Change of the total sea ice volume over the step, in m**3.

    Args:
        input: Ocean data at the previous step.
        gen: Generated ocean data at the current step.
        ice_volume_mask: Cells to count; None counts every cell.
        dtype: dtype of the sum.
        required_by: Names the caller in the error for a missing field.
    """
    try:
        gen_ice_volume = gen.sea_ice_volume.to(dtype)
        input_ice_volume = input.sea_ice_volume.to(dtype)
    except KeyError as err:
        raise ValueError(f"sea_ice_volume is required by {required_by}.") from err
    ice_volume_change = gen_ice_volume - input_ice_volume
    if ice_volume_mask is not None:
        ice_volume_change = torch.where(
            ice_volume_mask,
            ice_volume_change,
            torch.zeros_like(ice_volume_change),
        )
    # sea_ice_volume is per cell (m**3), so its total is a plain sum over the
    # local grid; the configs using it reject spatial parallelism.
    return ice_volume_change.sum(dim=(-2, -1), keepdim=True)


@dataclasses.dataclass
class IceVolumeSaltBudget:
    """Salt budget from the change of the total sea ice volume.

    Parameters:
        slope_psu: Change in total salt content (psu m**3) per change in total
            sea ice volume (m**3).
        ice_volume_mask: Cells whose sea_ice_volume prediction the stepper
            keeps; None counts every cell.
    """

    slope_psu: float
    ice_volume_mask: torch.Tensor | None

    def __call__(
        self,
        input: OceanData,
        gen: OceanData,
        forcing: OceanData,
        global_total: GlobalTotal,
        timestep_seconds: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        total_ice_volume_change = _total_sea_ice_volume_change(
            input,
            gen,
            self.ice_volume_mask,
            dtype,
            required_by="the ice volume salt budget",
        )
        return self.slope_psu * total_ice_volume_change


@dataclasses.dataclass
class SeaSurfaceHeightSaltBudget:
    """Salt budget from the change of the sea surface height,
    -S_ref * sum((SSH_gen - SSH_input) * ssf * A).

    Parameters:
        reference_salinity_psu: Salinity at which the added water dilutes the
            salt, in psu.
    """

    reference_salinity_psu: float

    def __call__(
        self,
        input: OceanData,
        gen: OceanData,
        forcing: OceanData,
        global_total: GlobalTotal,
        timestep_seconds: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        try:
            gen_height = gen.sea_surface_height.to(dtype)
            input_height = input.sea_surface_height.to(dtype)
        except KeyError as err:
            raise ValueError(
                "The sea surface height salt budget requires SSH (not zos) in the "
                "input and generated data."
            ) from err
        # NaN over land
        height_change = torch.nan_to_num(gen_height - input_height)  # m
        sea_surface_fraction = forcing.sea_surface_fraction.to(dtype)
        added_water_volume = global_total(height_change * sea_surface_fraction)  # m**3
        return -self.reference_salinity_psu * added_water_volume


def _generated_or_forcing(
    standard_name: str, gen: OceanData, forcing: OceanData
) -> OceanData | None:
    """The data holding a field: the generated data, or the forcing data if
    the field is not predicted. None if neither has it.
    """
    names = OCEAN_FIELD_NAME_PREFIXES[standard_name]
    in_gen = any(name in gen.data for name in names)
    in_forcing = any(name in forcing.data for name in names)
    if in_gen and in_forcing:
        raise ValueError(
            f"{names[0]} cannot be in both the generated and forcing data."
        )
    if in_gen:
        return gen
    if in_forcing:
        return forcing
    return None


def _sea_ice_fraction(data: OceanData) -> torch.Tensor | None:
    """ocean_sea_ice_fraction if present, else sea_ice_fraction, with NaN as
    zero; None if neither is available.
    """
    for name in ("ocean_sea_ice_fraction", "sea_ice_fraction"):
        try:
            return torch.nan_to_num(data[name])
        except KeyError:
            continue
    return None


@dataclasses.dataclass
class WaterFluxSaltBudget:
    """Salt budget from the surface water flux and the sea ice basal salt
    flux, with terms in place of wfo and sfdsi in their regimes. See
    ``WaterFluxSaltBudgetConfig``.

    Parameters:
        config: The budget configuration.
        ice_volume_mask: Cells whose sea_ice_volume prediction the stepper
            keeps; None counts every cell.
    """

    config: WaterFluxSaltBudgetConfig
    ice_volume_mask: torch.Tensor | None

    def _ice_covered(self, input: OceanData, gen: OceanData) -> torch.Tensor:
        fractions = [
            fraction
            for fraction in (_sea_ice_fraction(input), _sea_ice_fraction(gen))
            if fraction is not None
        ]
        if len(fractions) == 0:
            raise ValueError(
                "The water flux salt budget regimes need the sea ice fraction in "
                "the input or generated data."
            )
        threshold = self.config.regimes.sea_ice_fraction_threshold
        return functools.reduce(
            torch.logical_or, [fraction > threshold for fraction in fractions]
        )

    def __call__(
        self,
        input: OceanData,
        gen: OceanData,
        forcing: OceanData,
        global_total: GlobalTotal,
        timestep_seconds: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        config = self.config
        sea_surface_fraction = forcing.sea_surface_fraction.to(dtype)
        wfo_data = _generated_or_forcing("water_flux_into_sea_water", gen, forcing)
        if wfo_data is None:
            raise ValueError(
                "The water flux salt budget needs wfo in the generated or forcing "
                "data."
            )
        # fluxes per unit total cell area; wfo is NaN over land
        water = torch.nan_to_num(wfo_data.water_flux_into_sea_water.to(dtype))
        water = water * sea_surface_fraction  # kg/m**2/s
        salt = torch.zeros_like(water)  # g/m**2/s
        if config.brine_rejection:
            sfdsi_data = _generated_or_forcing(
                "downward_sea_ice_basal_salt_flux", gen, forcing
            )
            if sfdsi_data is None:
                raise ValueError(
                    "The water flux salt budget needs sfdsi in the generated or "
                    "forcing data; set brine_rejection to False to omit it."
                )
            sfdsi = sfdsi_data.downward_sea_ice_basal_salt_flux.to(dtype)
            salt = 1000.0 * sfdsi * sea_surface_fraction
        ice_mass_change_kg_per_s: torch.Tensor | float = 0.0
        ice_salt_change_g_per_s: torch.Tensor | float = 0.0
        if config.open_water_precipitation_minus_evaporation or config.sea_ice_mass:
            ice_covered = self._ice_covered(input, gen)
        if config.open_water_precipitation_minus_evaporation:
            try:
                precipitation_minus_evaporation = (
                    forcing.precipitation_minus_evaporation.to(dtype)
                )
            except KeyError as err:
                raise ValueError(
                    "The water flux salt budget needs PRATEsfc and LHTFLsfc in the "
                    "forcing data."
                ) from err
            coast = (
                sea_surface_fraction
                < config.regimes.coast_sea_surface_fraction_threshold
            )
            water = torch.where(
                ~ice_covered & ~coast,
                precipitation_minus_evaporation * sea_surface_fraction,
                water,
            )
        if config.sea_ice_mass:
            water = torch.where(ice_covered, torch.zeros_like(water), water)
            ice_volume_mask = (
                ice_covered
                if self.ice_volume_mask is None
                else ice_covered & self.ice_volume_mask
            )
            ice_mass_change_kg_per_s = (
                config.sea_ice_density_kg_m3
                * _total_sea_ice_volume_change(
                    input,
                    gen,
                    ice_volume_mask,
                    dtype,
                    required_by="the sea ice mass term of the water flux salt budget",
                )
                / timestep_seconds
            )
            if config.sea_ice_salinity_psu is not None:
                salt = torch.where(ice_covered, torch.zeros_like(salt), salt)
                ice_salt_change_g_per_s = (
                    config.sea_ice_salinity_psu * ice_mass_change_kg_per_s
                )
        water_kg_per_s = global_total(water) - ice_mass_change_kg_per_s
        salt_g_per_s = global_total(salt) - ice_salt_change_g_per_s
        salt_g_per_s = salt_g_per_s - config.reference_salinity_psu * water_kg_per_s
        # g/s * s / (kg/m**3) = (g/kg) m**3 = psu m**3
        return salt_g_per_s * timestep_seconds / DENSITY_OF_SEA_WATER_CM4


def _force_conserve_ocean_salt_content(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    area_weighted_sum: AreaWeightedSum,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
    method: Literal["scaled_salinity"],
    budget: SaltBudget | None,
    unaccounted_salting: float,
    weight_by_sea_surface_fraction: bool,
    use_float64: bool,
) -> TensorDict:
    if method != "scaled_salinity":
        raise NotImplementedError(
            f"Method {method!r} not implemented for ocean salt content conservation"
        )
    input = OceanData(input_data, vertical_coordinate)
    gen = OceanData(gen_data, vertical_coordinate)
    forcing = OceanData(forcing_data)
    dtype = torch.float64 if use_float64 else gen.data["so_0"].dtype

    def global_total(data: torch.Tensor) -> torch.Tensor:
        # area weights are cell areas as a fraction of the sphere, so scaling
        # the weighted sum by 4 pi R**2 gives the total over the ocean
        weighted_sum = area_weighted_sum(data, keepdim=True, name="ocean_salt_content")
        return weighted_sum * SPHERE_AREA_M2

    if weight_by_sea_surface_fraction:
        sea_surface_fraction = forcing.sea_surface_fraction.to(dtype)
    else:
        sea_surface_fraction = torch.ones(
            (), dtype=dtype, device=gen.data["so_0"].device
        )

    def global_salt_content(data: OceanData) -> torch.Tensor:
        column = vertical_coordinate.depth_integral(data.sea_water_salinity.to(dtype))
        return global_total(column * sea_surface_fraction)  # psu m**3

    global_gen_salt_content = global_salt_content(gen)
    global_input_salt_content = global_salt_content(input)
    ocean_area = global_total(
        torch.ones_like(gen.data["so_0"], dtype=dtype) * sea_surface_fraction
    )  # m**2
    expected_change = unaccounted_salting * timestep_seconds * ocean_area
    if budget is not None:
        expected_change = expected_change + budget(
            input, gen, forcing, global_total, timestep_seconds, dtype
        )
    salt_content_correction_ratio = (
        global_input_salt_content + expected_change
    ) / global_gen_salt_content
    # apply same salinity correction to all vertical layers
    out: TensorDict = {}
    n_levels = gen.sea_water_salinity.shape[-1]
    for k in range(n_levels):
        name = f"so_{k}"
        salinity = gen.data[name]
        out[name] = (salinity.to(dtype) * salt_content_correction_ratio).to(
            salinity.dtype
        )
    return out
