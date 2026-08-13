import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Protocol

import torch

from fme.core.atmosphere_data import AtmosphereData
from fme.core.constants import (
    DENSITY_OF_SEA_WATER_CM4,
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
from fme.core.corrector.utils import (
    ForceBounded,
    ForcePositive,
    replace_value_keep_gradient,
)
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
        method: Method to use for OHC budget correction. All options enforce
            the same column heat content budget and differ only in the vertical
            direction along which the correction acts:

            - "scaled_temperature": multiply the predicted potential
              temperature by a uniform factor, depositing heat in proportion
              to ``T_k * dz_k``. Because it multiplies about 0 degrees Celsius
              it contracts every vertical mode, which is what anchors a
              residual-prediction stepper, but it also biases heat toward the
              warm upper ocean.
            - "uniform_temperature": add a uniform temperature increment,
              depositing heat in proportion to ``dz_k`` alone. This gives
              better heat placement but only translates the profile, so it
              constrains the column mean and leaves every other vertical mode
              free. Do not use it with residual temperature prediction.
            - "anomaly_scaled_temperature": contract the temperature anomaly
              about ``reference_temperature`` -- ``T_k -> Tbar_k + r *
              (T_k - Tbar_k)``. Like "scaled_temperature" this acts on every
              vertical mode, so it anchors a residual stepper; unlike it, the
              deposition follows the anomaly rather than the absolute
              temperature, so it carries no warm-upper-ocean bias. ``r`` is
              clamped to ``1 +/- max_anomaly_contraction`` and any imbalance
              left over is closed with a uniform increment, so the budget is
              always satisfied exactly.
        reference_temperature: Per-level climatological potential temperature
            in degrees Celsius, required by "anomaly_scaled_temperature" and
            ignored otherwise. Must have one entry per depth level.
        max_anomaly_contraction: Largest fractional contraction of the anomaly
            permitted in a single step by "anomaly_scaled_temperature". Bounds
            the correction when the column anomaly is near zero, which would
            otherwise make the solved factor blow up.
        shape_restoring_rate: Fraction of the global-mean vertical *shape*
            anomaly removed per step, restoring toward
            ``reference_temperature``. Composes with any method and defaults to
            off.

            This exists so "uniform_temperature" can be used with residual
            temperature prediction. A residual update carries no climatological
            mean, so each level's global mean is a free integrator, and a
            uniform increment constrains only the column mean -- the remaining
            vertical modes drift unchecked. The shape anomaly is the global-mean
            anomaly profile with its thickness-weighted mean removed, so it
            carries no column heat: damping it touches none of the heat budget
            and leaves the chosen method's deposition profile, and therefore its
            heat placement, exactly as it was. The budget closure runs after and
            closes exactly, so conservation does not depend on this term being
            heat-neutral to machine precision.

            Only the global-mean profile is restored; horizontal structure at
            every level is untouched. A rate of 0.01 with a 5-day step is a
            restoring timescale of about 500 days.
        constant_unaccounted_heating: Area-weighted global mean
            column-integrated heating in W/m**2 to be added to the energy flux
            into the ocean when conserving the heat content. This can be useful
            for correcting errors in heat budget in target data. The same
            additional heating is imposed at all time steps and grid cells.

    """

    method: Literal[
        "scaled_temperature", "uniform_temperature", "anomaly_scaled_temperature"
    ]
    constant_unaccounted_heating: float = 0.0
    reference_temperature: list[float] | None = None
    max_anomaly_contraction: float = 0.1
    shape_restoring_rate: float = 0.0

    def __post_init__(self):
        if self.shape_restoring_rate < 0.0 or self.shape_restoring_rate > 1.0:
            raise ValueError(
                "shape_restoring_rate must be in [0, 1], got "
                f"{self.shape_restoring_rate}."
            )
        if self.shape_restoring_rate > 0.0 and self.reference_temperature is None:
            raise ValueError(
                "reference_temperature is required when shape_restoring_rate "
                "is greater than zero."
            )
        if self.method == "anomaly_scaled_temperature":
            if self.reference_temperature is None:
                raise ValueError(
                    "reference_temperature is required when method is "
                    "'anomaly_scaled_temperature'."
                )
            if not 0.0 < self.max_anomaly_contraction <= 1.0:
                raise ValueError(
                    "max_anomaly_contraction must be in (0, 1], got "
                    f"{self.max_anomaly_contraction}."
                )
        elif (
            self.reference_temperature is not None and self.shape_restoring_rate == 0.0
        ):
            raise ValueError(
                "reference_temperature is only meaningful for method "
                "'anomaly_scaled_temperature' or with a nonzero "
                f"shape_restoring_rate, not {self.method!r} alone."
            )


@dataclasses.dataclass
class OceanSaltContentBudgetConfig:
    """Configuration for ocean salt content budget correction.

    Unlike heat, global ocean salt content has no surface flux source:
    precipitation, evaporation and runoff move water, not salt. The only
    genuine exchange is with the sea-ice reservoir, and the model predicts
    the ice volume, so the expected change is computable from the model's
    own outputs with no external forcing data.

    Parameters:
        method: The available option is "scaled_salinity", which enforces
            the salt budget by scaling the predicted salinity by a
            vertically and horizontally uniform correction factor.
        ice_volume_salt_slope_psu: Empirical slope of the global column
            salt content change against the change in global-mean sea ice
            volume: ``expected_change = slope * delta_ice_volume``. Simple
            bookkeeping (ice exporting salt at its own salinity) suggests a
            small negative slope, but CM4's brine-rejection plumbing
            measures +5.85 psu on the 1pctCO2 dataset - calibrate against
            the target data rather than assuming. Set to 0 to ignore the
            ice exchange and hold salt content fixed.
        constant_unaccounted_salting: Area-weighted global mean rate of
            column salt content change, in psu m / s, added to the expected
            change at every step. Useful when the target data's salt budget
            has a small measured residual.
    """

    method: Literal["scaled_salinity"]
    ice_volume_salt_slope_psu: float = 0.0
    constant_unaccounted_salting: float = 0.0


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
    method: Literal[
        "scaled_temperature", "uniform_temperature", "anomaly_scaled_temperature"
    ]
    unaccounted_heating: float
    reference_temperature: list[float] | None = None
    max_anomaly_contraction: float = 0.1
    shape_restoring_rate: float = 0.0

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
            self.reference_temperature,
            self.max_anomaly_contraction,
            self.shape_restoring_rate,
        )
        return corrected, corrector_state


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig(CorrectorConfigABC):
    """Configuration for corrections applied to generated ocean data.

    Parameters:
        force_positive_names: Names of fields that should be forced to be greater
            than or equal to zero.
        variable_bounds: Mapping from field name to a (lower, upper) pair the
            generated field is clamped to after each step; either side may be
            null to leave it unbounded.
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
    variable_bounds: Mapping[str, tuple[float | None, float | None]] = (
        dataclasses.field(default_factory=dict)
    )
    sea_ice_fraction_correction: SeaIceFractionConfig | None = None
    surface_energy_flux_correction: SurfaceEnergyFluxCorrectionConfig | None = None
    ocean_heat_content_correction: OceanHeatContentBudgetConfig | None = None
    ocean_salt_content_correction: OceanSaltContentBudgetConfig | None = None
    keep_gradient_through_clamps: bool = False

    # keep_gradient_through_clamps only changes how gradient flows through the
    # clamps, not whether they are applied, so it is not a correction that
    # disable_corrections can switch off (it is also a no-op under no_grad).
    NON_CORRECTION_OPTIONS: ClassVar[frozenset[str]] = (
        CorrectorConfigABC.NON_CORRECTION_OPTIONS | {"keep_gradient_through_clamps"}
    )

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
        if len(self.variable_bounds) > 0:
            corrections.append(
                ForceBounded(
                    self.variable_bounds,
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
                    self.ocean_heat_content_correction.reference_temperature,
                    self.ocean_heat_content_correction.max_anomaly_contraction,
                    self.ocean_heat_content_correction.shape_restoring_rate,
                )
            )
        if self.ocean_salt_content_correction is not None:
            corrections.append(
                OceanSaltContentCorrection(
                    area_weighted_mean,
                    vertical_coordinate,
                    timestep_seconds,
                    self.ocean_salt_content_correction.method,
                    self.ocean_salt_content_correction.ice_volume_salt_slope_psu,
                    self.ocean_salt_content_correction.constant_unaccounted_salting,
                )
            )
        return OceanCorrector(corrections)


class OceanCorrector(CorrectionSequence):
    pass


@dataclasses.dataclass
class OceanSaltContentCorrection:
    """Correction that conserves ocean salt content."""

    area_weighted_mean: AreaWeightedMean
    vertical_coordinate: HasOceanDepthIntegral | None
    timestep_seconds: float
    method: Literal["scaled_salinity"]
    ice_volume_salt_slope_psu: float
    unaccounted_salting: float

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
            self.area_weighted_mean,
            self.vertical_coordinate,
            self.timestep_seconds,
            self.method,
            self.ice_volume_salt_slope_psu,
            self.unaccounted_salting,
        )
        return corrected, corrector_state


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
    method: Literal[
        "scaled_temperature", "uniform_temperature", "anomaly_scaled_temperature"
    ] = "scaled_temperature",
    unaccounted_heating: float = 0.0,
    reference_temperature: list[float] | None = None,
    max_anomaly_contraction: float = 0.1,
    shape_restoring_rate: float = 0.0,
) -> TensorDict:
    if method not in (
        "scaled_temperature",
        "uniform_temperature",
        "anomaly_scaled_temperature",
    ):
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
    target_ocean_heat_content = (
        global_input_ocean_heat_content + expected_change_ocean_heat_content
    )
    out: TensorDict = {}
    gen_potential_temperature = gen.sea_water_potential_temperature
    n_levels = gen_potential_temperature.shape[-1]

    if shape_restoring_rate > 0.0:
        # Damp the drift in the vertical modes the column budget cannot see.
        # The shape anomaly is the global-mean anomaly profile minus its
        # thickness-weighted mean, so it carries no column heat and removing a
        # fraction of it leaves the budget, and hence the deposition profile of
        # whichever method runs below, untouched. Only the global-mean profile
        # moves; horizontal structure at each level is left alone.
        if reference_temperature is None:
            raise ValueError(
                "reference_temperature is required when shape_restoring_rate "
                "is greater than zero."
            )
        if len(reference_temperature) != n_levels:
            raise ValueError(
                f"reference_temperature has {len(reference_temperature)} entries "
                f"but the data has {n_levels} depth levels."
            )
        shape_reference = torch.tensor(
            reference_temperature,
            dtype=gen_potential_temperature.dtype,
            device=gen_potential_temperature.device,
        )
        anomaly = gen_potential_temperature - shape_reference.reshape(
            *([1] * (gen_potential_temperature.ndim - 1)), n_levels
        )
        # thickness-weighted mean anomaly, taken through depth_integral so it
        # uses the same columns and mask the heat content itself uses
        global_anomaly_heat = area_weighted_mean(
            vertical_coordinate.depth_integral(
                anomaly * SPECIFIC_HEAT_OF_SEA_WATER_CM4 * DENSITY_OF_SEA_WATER_CM4
            ),
            keepdim=True,
            name="ocean_heat_content",
        )
        column_heat_capacity = area_weighted_mean(
            vertical_coordinate.depth_integral(
                torch.ones_like(gen_potential_temperature)
                * SPECIFIC_HEAT_OF_SEA_WATER_CM4
                * DENSITY_OF_SEA_WATER_CM4
            ),
            keepdim=True,
            name="ocean_heat_content",
        )
        thickness_weighted_mean_anomaly = global_anomaly_heat / column_heat_capacity
        restored: TensorDict = {}
        for k in range(n_levels):
            name = f"thetao_{k}"
            level_mean_anomaly = area_weighted_mean(
                anomaly.select(-1, k), keepdim=True, name=name
            )
            shape_anomaly = level_mean_anomaly - thickness_weighted_mean_anomaly
            restored[name] = gen.data[name] - shape_restoring_rate * shape_anomaly
        if "sst" in gen.data:
            sst_anomaly = area_weighted_mean(
                gen.data["sst"]
                - FREEZING_TEMPERATURE_KELVIN
                - float(reference_temperature[0]),
                keepdim=True,
                name="sst",
            )
            restored["sst"] = gen.data["sst"] - shape_restoring_rate * (
                sst_anomaly - thickness_weighted_mean_anomaly
            )
        # everything below reads the restored state
        gen = OceanData({**dict(gen.data), **restored}, vertical_coordinate)
        gen_potential_temperature = gen.sea_water_potential_temperature
        global_gen_ocean_heat_content = area_weighted_mean(
            gen.ocean_heat_content,
            keepdim=True,
            name="ocean_heat_content",
        )
        out.update(restored)

    if method == "scaled_temperature":
        # Multiplying about 0 degrees Celsius contracts every vertical mode,
        # which is what anchors a residual stepper, at the cost of depositing
        # heat in proportion to the absolute temperature.
        heat_content_correction_ratio = (
            target_ocean_heat_content / global_gen_ocean_heat_content
        )
        for k in range(n_levels):
            name = f"thetao_{k}"
            out[name] = gen.data[name] * heat_content_correction_ratio
        if "sst" in gen.data:
            out["sst"] = (  # assuming sst in Kelvin
                gen.data["sst"] - FREEZING_TEMPERATURE_KELVIN
            ) * heat_content_correction_ratio + FREEZING_TEMPERATURE_KELVIN
        return out

    # Both remaining methods need the column heat capacity, which must be a
    # depth_integral over the same columns as the heat content itself or the
    # budget is off by the difference.
    heat_capacity_per_area = area_weighted_mean(
        vertical_coordinate.depth_integral(
            torch.ones_like(gen_potential_temperature)
            * SPECIFIC_HEAT_OF_SEA_WATER_CM4
            * DENSITY_OF_SEA_WATER_CM4
        ),
        keepdim=True,
        name="ocean_heat_content",
    )
    mask = getattr(vertical_coordinate, "mask", None)
    if mask is None:
        raise ValueError(
            f"Method {method!r} needs the vertical coordinate's wet-cell mask "
            "to place an increment, but this vertical coordinate has none."
        )
    # Every cell the store marks valid is shifted, including those the
    # bathymetry puts at zero thickness: they hold real data and are scored,
    # and shifting them adds no heat. Cells outside the mask hold fill values
    # and are left alone. ``> 0`` mirrors depth_integral's own mask test.
    is_masked_valid = (mask > 0.0).to(dtype=gen_potential_temperature.dtype)

    if method == "uniform_temperature":
        temperature_increment = (
            target_ocean_heat_content - global_gen_ocean_heat_content
        ) / heat_capacity_per_area
    else:  # anomaly_scaled_temperature
        if reference_temperature is None:
            raise ValueError(
                "reference_temperature is required for " "'anomaly_scaled_temperature'."
            )
        if len(reference_temperature) != n_levels:
            raise ValueError(
                f"reference_temperature has {len(reference_temperature)} entries "
                f"but the data has {n_levels} depth levels."
            )
        reference = torch.tensor(
            reference_temperature,
            dtype=gen_potential_temperature.dtype,
            device=gen_potential_temperature.device,
        ).reshape(*([1] * (gen_potential_temperature.ndim - 1)), n_levels)
        reference_field = reference.expand_as(gen_potential_temperature)
        global_reference_ocean_heat_content = area_weighted_mean(
            vertical_coordinate.depth_integral(
                reference_field
                * SPECIFIC_HEAT_OF_SEA_WATER_CM4
                * DENSITY_OF_SEA_WATER_CM4
            ),
            keepdim=True,
            name="ocean_heat_content",
        )
        # Heat held in the anomaly about the reference profile, and the heat the
        # anomaly would have to hold for the budget to close.
        gen_anomaly_heat = (
            global_gen_ocean_heat_content - global_reference_ocean_heat_content
        )
        target_anomaly_heat = (
            target_ocean_heat_content - global_reference_ocean_heat_content
        )
        # Contract the anomaly toward the reference. Clamping bounds the factor
        # when the column anomaly passes through zero, where the exact solution
        # is unbounded; whatever the clamp leaves unclosed is handled by the
        # uniform increment below, so the budget still closes exactly.
        safe_denominator = torch.where(
            gen_anomaly_heat == 0.0,
            torch.ones_like(gen_anomaly_heat),
            gen_anomaly_heat,
        )
        anomaly_contraction = torch.where(
            gen_anomaly_heat == 0.0,
            torch.ones_like(gen_anomaly_heat),
            target_anomaly_heat / safe_denominator,
        ).clamp(1.0 - max_anomaly_contraction, 1.0 + max_anomaly_contraction)
        for k in range(n_levels):
            name = f"thetao_{k}"
            out[name] = reference[..., k] + anomaly_contraction * (
                gen.data[name] - reference[..., k]
            )
        if "sst" in gen.data:
            # sst is in Kelvin and is not in the heat content integral, so it
            # follows thetao_0's reference as a consistency choice.
            sst_reference = reference[..., 0] + FREEZING_TEMPERATURE_KELVIN
            out["sst"] = sst_reference + anomaly_contraction * (
                gen.data["sst"] - sst_reference
            )
        contracted_ocean_heat_content = (
            global_reference_ocean_heat_content + anomaly_contraction * gen_anomaly_heat
        )
        temperature_increment = (
            target_ocean_heat_content - contracted_ocean_heat_content
        ) / heat_capacity_per_area

    # "uniform_temperature" leaves ``out`` empty above and applies the whole
    # correction here; "anomaly_scaled_temperature" has already written the
    # contracted field and this adds the increment on top of it.
    contracted: TensorDict = dict(out) if out else dict(gen.data)
    for k in range(n_levels):
        name = f"thetao_{k}"
        out[name] = contracted[name] + temperature_increment * is_masked_valid.select(
            -1, k
        )
    if "sst" in gen.data:
        # An increment needs no Kelvin offset, unlike the multiplicative path.
        out["sst"] = contracted["sst"] + temperature_increment * is_masked_valid.select(
            -1, 0
        )
    return out


def _force_conserve_ocean_salt_content(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
    method: Literal["scaled_salinity"] = "scaled_salinity",
    ice_volume_salt_slope_psu: float = 0.0,
    unaccounted_salting: float = 0.0,
) -> TensorDict:
    """Scale the predicted salinity so global column salt content changes
    only by the sea-ice exchange plus any constant unaccounted rate.

    Salt has no surface flux source (freshwater fluxes move water, not
    salt), so no forcing data is required: the ice term uses the model's
    own predicted sea ice volume, mirroring how the heat correction anchors
    to the predicted surface heat flux.
    """
    if method != "scaled_salinity":
        raise NotImplementedError(
            f"Method {method!r} not implemented for ocean salt content " "conservation"
        )
    input = OceanData(input_data, vertical_coordinate)
    gen = OceanData(gen_data, vertical_coordinate)
    salinity = gen.sea_water_salinity
    global_gen_salt = area_weighted_mean(
        vertical_coordinate.depth_integral(salinity),
        keepdim=True,
        name="ocean_salt_content",
    )
    global_input_salt = area_weighted_mean(
        vertical_coordinate.depth_integral(input.sea_water_salinity),
        keepdim=True,
        name="ocean_salt_content",
    )
    expected_change = torch.zeros_like(global_input_salt)
    if ice_volume_salt_slope_psu != 0.0:
        try:
            ice_change = area_weighted_mean(
                gen.sea_ice_volume - input.sea_ice_volume,
                keepdim=True,
                name="ocean_salt_content",
            )
            expected_change = expected_change + ice_volume_salt_slope_psu * ice_change
        except KeyError:
            pass  # no sea ice volume in this model: no ice exchange term
    expected_change = expected_change + unaccounted_salting * timestep_seconds
    salt_content_correction_ratio = (
        global_input_salt + expected_change
    ) / global_gen_salt
    out: TensorDict = {}
    n_levels = salinity.shape[-1]
    for k in range(n_levels):
        name = f"so_{k}"
        out[name] = gen.data[name] * salt_content_correction_ratio
    return out
