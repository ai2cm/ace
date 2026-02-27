import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import dacite
import torch

from fme.core.constants import (
    DENSITY_OF_WATER_CM4,
    FREEZING_TEMPERATURE_KELVIN,
    REFERENCE_SALINITY_PSU,
    SPECIFIC_HEAT_OF_WATER_CM4,
)
from fme.core.corrector.ocean_mld import (
    apply_geothermal_bottom_correction,
    compute_mld_soft_weights_from_ocean_data,
    compute_mld_weights_from_ocean_data,
)
from fme.core.corrector.registry import CorrectorABC
from fme.core.corrector.utils import force_positive
from fme.core.gridded_ops import GriddedOperations
from fme.core.ocean_data import HasOceanDepthIntegral, OceanData
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorDict, TensorMapping


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
        self, gen_data: TensorMapping, input_data: TensorMapping
    ) -> TensorDict:
        out = {**gen_data}
        out[self.sea_ice_fraction_name] = torch.clamp(
            out[self.sea_ice_fraction_name], min=0.0, max=1.0
        )
        if self.remove_negative_ocean_fraction:
            negative_ocean_fraction = (
                1
                - out[self.sea_ice_fraction_name]
                - input_data[self.land_fraction_name]
            )
            negative_ocean_fraction = negative_ocean_fraction.clip(max=0)
            out[self.sea_ice_fraction_name] += negative_ocean_fraction
        for name in self.zero_where_ice_free_names:
            out[name] = gen_data[name] * (out[self.sea_ice_fraction_name] > 0.0)
        return out


@dataclasses.dataclass
class OceanHeatContentBudgetConfig:
    """Configuration for ocean heat content budget correction.

    Parameters:
        method: Method to use for OHC budget correction. Options:
            "scaled_temperature" enforces conservation by scaling the predicted
            potential temperature by a vertically and horizontally uniform
            correction factor. "mixed_layer_depth" distributes the energy
            deficit within the mixed layer using JMD95-derived MLD weights.
            "mixed_layer_depth_geo" first applies a geothermal heat flux
            correction to the bottom ocean cell and then distributes the
            remaining deficit via MLD weights. "mixed_layer_depth_soft" and
            "mixed_layer_depth_soft_geo" are differentiable variants that use
            soft-thresholded MLD (requires ``mld_soft_threshold`` on the
            parent ``OceanCorrectorConfig``).
        constant_unaccounted_heating: Area-weighted global mean
            column-integrated heating in W/m**2 to be added to the energy flux
            into the ocean when conserving the heat content. This can be useful
            for correcting errors in heat budget in target data. The same
            additional heating is imposed at all time steps and grid cells.

    """

    method: Literal[
        "scaled_temperature",
        "mixed_layer_depth",
        "mixed_layer_depth_geo",
        "mixed_layer_depth_soft",
        "mixed_layer_depth_soft_geo",
    ]
    constant_unaccounted_heating: float = 0.0


@dataclasses.dataclass
class OceanSaltContentBudgetConfig:
    """Configuration for ocean salt content budget correction.

    Parameters:
        method: Method to use for salt content budget correction. Options:
            "scaled_salinity" enforces conservation by scaling the predicted
            salinity by a vertically and horizontally uniform correction factor.
            "mixed_layer_depth" distributes the salt deficit within the mixed
            layer using JMD95-derived MLD weights. "mixed_layer_depth_soft"
            is a differentiable variant using soft-thresholded MLD (requires
            ``mld_soft_threshold`` on the parent ``OceanCorrectorConfig``).
        constant_unaccounted_salt_flux: Area-weighted global mean
            column-integrated salt flux in g/m**2/s to be added to the virtual
            salt flux when conserving the salt content. This can be useful for
            correcting errors in salt budget in target data.
    """

    method: Literal["scaled_salinity", "mixed_layer_depth", "mixed_layer_depth_soft"]
    constant_unaccounted_salt_flux: float = 0.0


_SOFT_MLD_METHODS = frozenset(
    {
        "mixed_layer_depth_soft",
        "mixed_layer_depth_soft_geo",
    }
)


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig:
    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: SeaIceFractionConfig | None = None
    ocean_heat_content_correction: OceanHeatContentBudgetConfig | None = None
    ocean_salt_content_correction: OceanSaltContentBudgetConfig | None = None
    mld_soft_threshold: float | None = None

    def __post_init__(self) -> None:
        methods = set()
        if self.ocean_heat_content_correction is not None:
            methods.add(str(self.ocean_heat_content_correction.method))
        if self.ocean_salt_content_correction is not None:
            methods.add(str(self.ocean_salt_content_correction.method))
        uses_soft = bool(methods & _SOFT_MLD_METHODS)
        if uses_soft and self.mld_soft_threshold is None:
            raise ValueError(
                "mld_soft_threshold must be set when using a soft MLD method "
                f"({methods & _SOFT_MLD_METHODS})."
            )
        if self.mld_soft_threshold is not None and not uses_soft:
            raise ValueError(
                "mld_soft_threshold is set but no budget correction uses a "
                "soft MLD method."
            )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "OceanCorrectorConfig":
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
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
        elif (
            "ocean_heat_content_correction" in state_copy
            and "method" in state_copy["ocean_heat_content_correction"]
            and state_copy["ocean_heat_content_correction"]["method"]
            == "constant_temperature"
        ):
            # FIXME: don't merge!
            state_copy["ocean_heat_content_correction"]["method"] = "scaled_temperature"
        if "sea_ice_fraction_correction" in state_copy:
            sif = state_copy["sea_ice_fraction_correction"]
            if isinstance(sif, dict) and "sea_ice_thickness_name" in sif:
                thickness_name = sif.pop("sea_ice_thickness_name")
                if thickness_name is not None:
                    sif.setdefault("zero_where_ice_free_names", []).append(
                        thickness_name
                    )
        return state_copy


class OceanCorrector(CorrectorABC):
    def __init__(
        self,
        config: OceanCorrectorConfig,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasOceanDepthIntegral | None,
        timestep: datetime.timedelta,
    ):
        self._config = config
        self._gridded_operations = gridded_operations
        self._vertical_coordinate = vertical_coordinate
        self._timestep = timestep

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorDict:
        if len(self._config.force_positive_names) > 0:
            gen_data = force_positive(gen_data, self._config.force_positive_names)
        if self._config.sea_ice_fraction_correction is not None:
            gen_data = self._config.sea_ice_fraction_correction(gen_data, input_data)
        mld_weights: torch.Tensor | None = None
        if self._config.ocean_salt_content_correction is not None:
            if self._vertical_coordinate is None:
                raise ValueError(
                    "Ocean salt content correction is turned on, but no vertical "
                    "coordinate is available."
                )
            gen_data, mld_weights = _force_conserve_ocean_salt_content(
                input_data,
                gen_data,
                forcing_data,
                self._gridded_operations.area_weighted_mean,
                self._vertical_coordinate,
                self._timestep.total_seconds(),
                self._config.ocean_salt_content_correction.method,
                self._config.ocean_salt_content_correction.constant_unaccounted_salt_flux,
                mld_soft_tau=self._config.mld_soft_threshold,
            )
        if self._config.ocean_heat_content_correction is not None:
            if self._vertical_coordinate is None:
                raise ValueError(
                    "Ocean heat content correction is turned on, but no vertical "
                    "coordinate is available."
                )
            gen_data = _force_conserve_ocean_heat_content(
                input_data,
                gen_data,
                forcing_data,
                self._gridded_operations.area_weighted_mean,
                self._vertical_coordinate,
                self._timestep.total_seconds(),
                self._config.ocean_heat_content_correction.method,
                self._config.ocean_heat_content_correction.constant_unaccounted_heating,
                mld_weights=mld_weights,
                mld_soft_tau=self._config.mld_soft_threshold,
            )
        return dict(gen_data)


class AreaWeightedMean(Protocol):
    def __call__(
        self, data: torch.Tensor, keepdim: bool, name: str | None = None
    ) -> torch.Tensor: ...


def _force_conserve_ocean_heat_content(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
    method: Literal[
        "scaled_temperature",
        "mixed_layer_depth",
        "mixed_layer_depth_geo",
        "mixed_layer_depth_soft",
        "mixed_layer_depth_soft_geo",
    ] = "scaled_temperature",
    unaccounted_heating: float = 0.0,
    mld_weights: torch.Tensor | None = None,
    mld_soft_tau: float | None = None,
) -> TensorDict:
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

    if method in ("mixed_layer_depth_geo", "mixed_layer_depth_soft_geo"):
        apply_geothermal_bottom_correction(
            gen,
            forcing,
            vertical_coordinate,
            timestep_seconds,
        )

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
    energy_flux_global_mean = _compute_energy_flux_global_mean(
        input,
        gen,
        forcing,
        area_weighted_mean,
    )
    expected_change_ocean_heat_content = (
        energy_flux_global_mean + unaccounted_heating
    ) * timestep_seconds

    if method == "scaled_temperature":
        _apply_scaled_heat_correction(
            gen,
            global_input_ocean_heat_content,
            expected_change_ocean_heat_content,
            global_gen_ocean_heat_content,
        )
    elif method in ("mixed_layer_depth", "mixed_layer_depth_geo"):
        if mld_weights is None:
            mld_weights = compute_mld_weights_from_ocean_data(
                gen,
                forcing,
                vertical_coordinate,
            )
        _apply_mld_heat_correction(
            gen,
            global_input_ocean_heat_content,
            expected_change_ocean_heat_content,
            global_gen_ocean_heat_content,
            mld_weights,
            vertical_coordinate,
            area_weighted_mean,
        )
    elif method in ("mixed_layer_depth_soft", "mixed_layer_depth_soft_geo"):
        assert mld_soft_tau is not None
        if mld_weights is None:
            mld_weights = compute_mld_soft_weights_from_ocean_data(
                gen,
                forcing,
                vertical_coordinate,
                tau=mld_soft_tau,
            )
        _apply_mld_heat_correction(
            gen,
            global_input_ocean_heat_content,
            expected_change_ocean_heat_content,
            global_gen_ocean_heat_content,
            mld_weights,
            vertical_coordinate,
            area_weighted_mean,
        )
    else:
        raise NotImplementedError(
            f"Method {method!r} not implemented for ocean heat content conservation"
        )
    return gen.data


def _compute_energy_flux_global_mean(
    input: OceanData,
    gen: OceanData,
    forcing: OceanData,
    area_weighted_mean: AreaWeightedMean,
) -> torch.Tensor:
    try:
        net_energy_flux_into_ocean = (
            gen.net_downward_surface_heat_flux_total_area
            + forcing.geothermal_heat_flux * forcing.sea_surface_fraction
        )
    except KeyError:
        try:
            net_energy_flux_into_ocean = (
                gen.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction
        except KeyError:
            net_energy_flux_into_ocean = (
                input.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction
    return area_weighted_mean(
        net_energy_flux_into_ocean,
        keepdim=True,
        name="ocean_heat_content",
    )


def _apply_scaled_heat_correction(
    gen: OceanData,
    global_input_ohc: torch.Tensor,
    expected_change: torch.Tensor,
    global_gen_ohc: torch.Tensor,
) -> None:
    ratio = (global_input_ohc + expected_change) / global_gen_ohc
    n_levels = gen.sea_water_potential_temperature.shape[-1]
    for k in range(n_levels):
        gen.data[f"thetao_{k}"] = gen.data[f"thetao_{k}"] * ratio
    if "sst" in gen.data:
        gen.data["sst"] = (
            gen.data["sst"] - FREEZING_TEMPERATURE_KELVIN
        ) * ratio + FREEZING_TEMPERATURE_KELVIN


def _apply_mld_heat_correction(
    gen: OceanData,
    global_input_ohc: torch.Tensor,
    expected_change: torch.Tensor,
    global_gen_ohc: torch.Tensor,
    mld_weights: torch.Tensor,
    vertical_coordinate: HasOceanDepthIntegral,
    area_weighted_mean: AreaWeightedMean,
) -> None:
    """Distribute energy deficit within the mixed layer (modifies *gen* in place)."""
    delta_E = global_input_ohc + expected_change - global_gen_ohc
    dz = vertical_coordinate.get_idepth().diff(dim=-1)  # (Z,)
    total_active = mld_weights.sum(dim=-1)  # (B, Y, X)
    Ah_mean = area_weighted_mean(
        total_active,
        keepdim=True,
        name="ocean_heat_content",
    )
    dT = (
        delta_E.unsqueeze(-1)
        * mld_weights
        / (
            Ah_mean.unsqueeze(-1)
            * DENSITY_OF_WATER_CM4
            * SPECIFIC_HEAT_OF_WATER_CM4
            * dz
        )
    )
    n_levels = gen.sea_water_potential_temperature.shape[-1]
    for k in range(n_levels):
        gen.data[f"thetao_{k}"] = gen.data[f"thetao_{k}"] + dT[..., k]
    if "sst" in gen.data:
        gen.data["sst"] = gen.data["sst"] + dT[..., 0]


def _force_conserve_ocean_salt_content(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
    method: Literal[
        "scaled_salinity", "mixed_layer_depth", "mixed_layer_depth_soft"
    ] = "scaled_salinity",
    unaccounted_salt_flux: float = 0.0,
    mld_soft_tau: float | None = None,
) -> tuple[TensorDict, torch.Tensor | None]:
    if "wfo" in gen_data and "wfo" in forcing_data:
        raise ValueError(
            "Water flux into sea water cannot be present in both gen_data and "
            "forcing_data."
        )
    input = OceanData(input_data, vertical_coordinate)
    gen = OceanData(gen_data, vertical_coordinate)
    forcing = OceanData(forcing_data)

    global_gen_salt_content = area_weighted_mean(
        gen.ocean_salt_content,
        keepdim=True,
        name="ocean_salt_content",
    )
    global_input_salt_content = area_weighted_mean(
        input.ocean_salt_content,
        keepdim=True,
        name="ocean_salt_content",
    )
    try:
        wfo = gen.water_flux_into_sea_water
    except KeyError:
        wfo = input.water_flux_into_sea_water
    virtual_salt_flux = -REFERENCE_SALINITY_PSU * wfo * forcing.sea_surface_fraction
    salt_flux_global_mean = area_weighted_mean(
        virtual_salt_flux,
        keepdim=True,
        name="ocean_salt_content",
    )
    expected_change_salt_content = (
        salt_flux_global_mean + unaccounted_salt_flux
    ) * timestep_seconds

    if method == "scaled_salinity":
        return _apply_scaled_salt_correction(
            gen,
            global_input_salt_content,
            expected_change_salt_content,
            global_gen_salt_content,
        )
    elif method == "mixed_layer_depth":
        mld_weights = compute_mld_weights_from_ocean_data(
            gen,
            forcing,
            vertical_coordinate,
        )
        _apply_mld_salt_correction(
            gen,
            global_input_salt_content,
            expected_change_salt_content,
            global_gen_salt_content,
            mld_weights,
            vertical_coordinate,
            area_weighted_mean,
        )
        return gen.data, mld_weights
    elif method == "mixed_layer_depth_soft":
        assert mld_soft_tau is not None
        mld_weights = compute_mld_soft_weights_from_ocean_data(
            gen,
            forcing,
            vertical_coordinate,
            tau=mld_soft_tau,
        )
        _apply_mld_salt_correction(
            gen,
            global_input_salt_content,
            expected_change_salt_content,
            global_gen_salt_content,
            mld_weights,
            vertical_coordinate,
            area_weighted_mean,
        )
        return gen.data, mld_weights
    else:
        raise NotImplementedError(
            f"Method {method!r} not implemented for ocean salt content conservation"
        )


def _apply_scaled_salt_correction(
    gen: OceanData,
    global_input_salt_content: torch.Tensor,
    expected_change: torch.Tensor,
    global_gen_salt_content: torch.Tensor,
) -> tuple[TensorDict, None]:
    ratio = (global_input_salt_content + expected_change) / global_gen_salt_content
    n_levels = gen.sea_water_salinity.shape[-1]
    for k in range(n_levels):
        gen.data[f"so_{k}"] = gen.data[f"so_{k}"] * ratio
    return gen.data, None


def _apply_mld_salt_correction(
    gen: OceanData,
    global_input_salt_content: torch.Tensor,
    expected_change: torch.Tensor,
    global_gen_salt_content: torch.Tensor,
    mld_weights: torch.Tensor,
    vertical_coordinate: HasOceanDepthIntegral,
    area_weighted_mean: AreaWeightedMean,
) -> None:
    """Distribute salt deficit within the mixed layer (modifies *gen* in place)."""
    delta_S = global_input_salt_content + expected_change - global_gen_salt_content
    dz = vertical_coordinate.get_idepth().diff(dim=-1)  # (Z,)
    total_active = mld_weights.sum(dim=-1)  # (B, Y, X)
    Ah_mean = area_weighted_mean(
        total_active,
        keepdim=True,
        name="ocean_salt_content",
    )
    dS = (
        delta_S.unsqueeze(-1)
        * mld_weights
        / (Ah_mean.unsqueeze(-1) * DENSITY_OF_WATER_CM4 * dz)
    )
    n_levels = gen.sea_water_salinity.shape[-1]
    for k in range(n_levels):
        gen.data[f"so_{k}"] = gen.data[f"so_{k}"] + dS[..., k]
