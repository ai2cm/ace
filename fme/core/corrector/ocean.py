import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import torch

from fme.core.atmosphere_data import AtmosphereData
from fme.core.cloud import open_dataset_via_inter_filesystem_copy
from fme.core.constants import (
    FREEZING_TEMPERATURE_KELVIN,
    LATENT_HEAT_OF_FREEZING,
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


SurfaceEnergyFluxMethod = Literal[
    "residual_prediction",
    "prescribed",
    "prescribed_open_ocean",
    "prescribed_cell_mean",
]


@dataclasses.dataclass
class RunoffHeatFluxConfig:
    """A static map of the heat carried into the ocean by river and iceberg
    runoff.

    The atmosphere's surface fluxes do not carry this term (runoff comes from
    the land model), but an ocean heat budget that books the heat content of
    every water stream crossing the surface (e.g. MOM6's ``hfds``) includes
    it, almost entirely in coastal cells. A time mean of the ocean model's
    own diagnostic (``hfrunoffds`` in a stats ``time-mean.nc``) is the
    intended source.

    Give either ``path`` or ``values``. A training config gives ``path``;
    ``load`` reads the map once into ``values`` and clears ``path``, so a
    checkpoint holds the map itself and inference from it reads no file.

    Parameters:
        path: Path to a netCDF file (any fsspec filesystem) holding the map on
            the model's ``(lat, lon)`` grid. NaN cells (land in the ocean
            model's diagnostic) are read as zero.
        name: Variable name within the file.
        per_unit_sea_area: If True (the MOM6 convention), the map is per unit
            sea area and is multiplied by ``sea_surface_fraction`` to give a
            flux per unit cell area. If False it is used as is.
        values: The map itself, as rows of latitude. Set by ``load``; not
            meant to be written in a config by hand.
    """

    path: str | None = None
    name: str = "hfrunoffds"
    per_unit_sea_area: bool = True
    values: list[list[float]] | None = None

    def __post_init__(self):
        if (self.path is None) == (self.values is None):
            raise ValueError(
                "runoff_heat_flux needs exactly one of path or values, got "
                f"path={self.path!r} and "
                f"{'no values' if self.values is None else 'values'}."
            )

    def load(self):
        """Read the map from ``path`` into ``values``, so the configuration no
        longer depends on the file.
        """
        if self.path is not None:
            self.values = _read_runoff_heat_map(self.path, self.name).tolist()
            self.path = None

    def build(self, img_shape: tuple[int, int] | None = None) -> "StaticRunoffHeatFlux":
        """Build the runoff heat flux, reading the map from ``path`` if it has
        not been loaded.

        Args:
            img_shape: Horizontal shape of the ocean grid, to validate the map
                against. Not validated if None.
        """
        if self.values is not None:
            runoff_map = torch.tensor(self.values, dtype=torch.float32)
        else:
            assert self.path is not None  # guaranteed by __post_init__
            runoff_map = _read_runoff_heat_map(self.path, self.name)
        if img_shape is not None and tuple(runoff_map.shape) != tuple(img_shape):
            raise ValueError(
                f"Runoff heat map has shape {tuple(runoff_map.shape)} but the "
                f"ocean grid has horizontal shape {tuple(img_shape)}."
            )
        return StaticRunoffHeatFlux(runoff_map, self.per_unit_sea_area)


def _read_runoff_heat_map(path: str, name: str) -> torch.Tensor:
    """Read a ``(lat, lon)`` map from netCDF as float32, NaN read as zero."""
    ds = open_dataset_via_inter_filesystem_copy(path)
    da = ds[name]
    if set(da.dims) != {"lat", "lon"}:
        raise ValueError(
            f"Runoff heat map {name!r} in {path!r} must have dims (lat, lon), "
            f"got {da.dims}."
        )
    values = da.transpose("lat", "lon").values
    return torch.nan_to_num(torch.as_tensor(values, dtype=torch.float32), nan=0.0)


class StaticRunoffHeatFlux:
    """A static runoff heat map, cached per device."""

    def __init__(self, runoff_map: torch.Tensor, per_unit_sea_area: bool):
        self._map = runoff_map
        self._per_unit_sea_area = per_unit_sea_area
        self._by_device: dict[torch.device, torch.Tensor] = {}

    def __call__(self, sea_surface_fraction: torch.Tensor) -> torch.Tensor:
        """Return the runoff heat flux per unit cell area, on the device and
        with the trailing ``(lat, lon)`` shape of ``sea_surface_fraction``.
        """
        device = sea_surface_fraction.device
        if device not in self._by_device:
            self._by_device[device] = self._map.to(device)
        runoff = self._by_device[device]
        if runoff.shape != sea_surface_fraction.shape[-2:]:
            raise ValueError(
                f"Runoff heat map has shape {tuple(runoff.shape)} but the "
                f"ocean fields have horizontal shape "
                f"{tuple(sea_surface_fraction.shape[-2:])}."
            )
        if self._per_unit_sea_area:
            return runoff * sea_surface_fraction
        return runoff.expand_as(sea_surface_fraction)


@dataclasses.dataclass
class UnderIceHeatFluxConfig:
    """Energy-conserving heat flux into the ocean under sea ice.

    Used by the "prescribed_cell_mean" surface energy flux correction in place
    of the network's own prediction under ice. The ice-covered part of each
    cell receives the atmosphere's net flux into the surface minus the heat the
    ice stores over the step. Only latent storage is counted: ice growth
    releases ``ice_density * latent_heat_of_fusion`` per unit volume, so the
    ocean loses less heat than the atmosphere took, and melting takes that
    heat before it reaches the water. The ice's sensible heat and the snow on
    the ice are not included.

    Parameters:
        sea_ice_volume_name: Name of the predicted sea-ice volume, the total ice
            volume in each grid cell in m**3. Must be an ocean prognostic
            variable, present in both the step's input and its output.
        ice_density: Sea-ice density in kg/m**3.
        latent_heat_of_fusion: Latent heat of fusion of ice in J/kg.
        gradient_through_ice_volume: If True, the storage term passes gradients
            to the predicted ice volume, so a loss on the heat flux also trains
            the ice. If False, the predicted volume is detached in this term.
    """

    sea_ice_volume_name: str = "sea_ice_volume"
    ice_density: float = 905.0
    latent_heat_of_fusion: float = LATENT_HEAT_OF_FREEZING
    gradient_through_ice_volume: bool = True

    def __post_init__(self):
        if self.ice_density <= 0 or self.latent_heat_of_fusion <= 0:
            raise ValueError(
                "under_ice ice_density and latent_heat_of_fusion must be positive."
            )

    def build(
        self, cell_area_m2: torch.Tensor | None, timestep_seconds: float
    ) -> "UnderIceHeatFlux":
        if cell_area_m2 is None:
            raise ValueError(
                "surface_energy_flux_correction.under_ice needs cell areas in m**2, "
                "which this grid does not provide."
            )
        return UnderIceHeatFlux(self, cell_area_m2, timestep_seconds)


class UnderIceHeatFlux:
    """Latent heat stored by the ice over one step, per unit cell area."""

    def __init__(
        self,
        config: UnderIceHeatFluxConfig,
        cell_area_m2: torch.Tensor,
        timestep_seconds: float,
    ):
        self._config = config
        self._cell_area_m2 = cell_area_m2
        self._timestep_seconds = timestep_seconds

    def storage_release(
        self, input_data: TensorMapping, gen_data: TensorMapping
    ) -> torch.Tensor:
        """Heat released into the ocean by ice growth over the step, W/m**2 of
        cell area (negative where the ice melted).
        """
        name = self._config.sea_ice_volume_name
        if name not in input_data or name not in gen_data:
            raise KeyError(
                f"surface_energy_flux_correction.under_ice needs {name!r} in both "
                "the step's input and its output."
            )
        volume_out = gen_data[name]
        if not self._config.gradient_through_ice_volume:
            volume_out = volume_out.detach()
        area = self._cell_area_m2.to(device=volume_out.device, dtype=volume_out.dtype)
        dvolume = torch.nan_to_num(volume_out - input_data[name])
        return (
            self._config.ice_density
            * self._config.latent_heat_of_fusion
            * dvolume
            / (area * self._timestep_seconds)
        )


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
      - "prescribed_cell_mean": corrected_hfds_total_area = (net_flux *
        (1 - sea_ice_fraction) + runoff_heat) * (sea_surface_fraction > 0) +
        gen_hfds_total_area * sea_ice_fraction. The atmosphere's cell-mean net
        flux, per unit cell area and not scaled by any sea fraction, is
        prescribed over the ice-free part of every sea-containing cell, land
        part included; the network prediction is retained under sea ice only.
        ``runoff_heat_flux`` adds the river-runoff heat the atmosphere does not
        carry. With ``under_ice`` set, the network's prediction is not used at
        all: the ice part also receives the atmosphere's flux, plus the heat
        released by ice growth over the step, so corrected_hfds_total_area =
        (net_flux + runoff_heat + storage_release) * (sea_surface_fraction >
        0). Only the ``hfds_total_area`` (per unit cell area) target is
        supported.

    Parameters:
        method: Method to use for the correction.
        runoff_heat_flux: Optional static river-runoff heat map, used by
            "prescribed_cell_mean" only.
        under_ice: Optional energy-conserving flux under sea ice, used by
            "prescribed_cell_mean" only. If None, the network's prediction is
            kept under ice.

    """

    method: SurfaceEnergyFluxMethod
    runoff_heat_flux: RunoffHeatFluxConfig | None = None
    under_ice: UnderIceHeatFluxConfig | None = None

    def __post_init__(self):
        for name in ("runoff_heat_flux", "under_ice"):
            if (
                getattr(self, name) is not None
                and self.method != "prescribed_cell_mean"
            ):
                raise ValueError(
                    f"surface_energy_flux_correction.{name} is only used by the "
                    f"'prescribed_cell_mean' method, got method={self.method!r}."
                )

    @property
    def requires_cell_area(self) -> bool:
        """Whether ``build`` needs the grid's cell areas."""
        return self.under_ice is not None

    @property
    def requires_img_shape(self) -> bool:
        """Whether ``build`` validates against the grid's horizontal shape."""
        return self.runoff_heat_flux is not None

    def load(self):
        """Update the configuration in place so it does not depend on external
        files.
        """
        if self.runoff_heat_flux is not None:
            self.runoff_heat_flux.load()

    def build(
        self,
        timestep_seconds: float,
        cell_area_m2: torch.Tensor | None = None,
        img_shape: tuple[int, int] | None = None,
    ) -> "SurfaceEnergyFluxCorrection":
        """Build the correction.

        Args:
            timestep_seconds: Model timestep in seconds.
            cell_area_m2: Cell areas in m**2, required if ``requires_cell_area``.
            img_shape: Horizontal shape of the ocean grid, to validate the
                runoff heat map against. Not validated if None.
        """
        return SurfaceEnergyFluxCorrection(
            self.method,
            runoff_heat_flux=(
                None
                if self.runoff_heat_flux is None
                else self.runoff_heat_flux.build(img_shape)
            ),
            under_ice=(
                None
                if self.under_ice is None
                else self.under_ice.build(cell_area_m2, timestep_seconds)
            ),
        )


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

    method: SurfaceEnergyFluxMethod
    runoff_heat_flux: StaticRunoffHeatFlux | None = None
    under_ice: UnderIceHeatFlux | None = None

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
            runoff_heat_flux=self.runoff_heat_flux,
            under_ice=self.under_ice,
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

    def load(self) -> None:
        if self.surface_energy_flux_correction is not None:
            self.surface_energy_flux_correction.load()

    def _get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "OceanCorrector":
        cell_area_m2 = None
        img_shape = None
        flux_correction = self.surface_energy_flux_correction
        if flux_correction is not None and flux_correction.requires_cell_area:
            cell_area_m2 = dataset_info.horizontal_coordinates.area_weights_m2
        if flux_correction is not None and flux_correction.requires_img_shape:
            img_shape = dataset_info.img_shape
        return self._build(
            dataset_info.gridded_operations,
            dataset_info.ocean_vertical_coordinate,
            dataset_info.timestep,
            cell_area_m2=cell_area_m2,
            img_shape=img_shape,
        )

    def _build(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasOceanDepthIntegral | None,
        timestep: datetime.timedelta,
        cell_area_m2: torch.Tensor | None = None,
        img_shape: tuple[int, int] | None = None,
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
                self.surface_energy_flux_correction.build(
                    timestep_seconds, cell_area_m2=cell_area_m2, img_shape=img_shape
                )
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
    method: SurfaceEnergyFluxMethod,
    runoff_heat_flux: StaticRunoffHeatFlux | None = None,
    under_ice: UnderIceHeatFlux | None = None,
) -> TensorDict:
    """Apply surface energy flux correction to the generated hfds.

    The ocean_fraction naturally zeroes the correction on land and reduces
    it under sea ice.

    Methods:
        residual_prediction: gen_hfds + ocean_fraction * net_flux
        prescribed: net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
        prescribed_open_ocean: net_flux where ocean_fraction == 1, else gen_hfds
        prescribed_cell_mean: (net_flux * (1 - sea_ice_fraction) + runoff_heat)
            where sea_surface_fraction > 0, plus gen_hfds * sea_ice_fraction;
            net_flux here is the unscaled cell mean (hfds_total_area only).
            With under_ice: (net_flux + runoff_heat + storage_release) where
            sea_surface_fraction > 0, and no network term.
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
    gen_hfds = gen_data[hfds_name]
    if method == "prescribed_cell_mean":
        if hfds_name != "hfds_total_area":
            raise NotImplementedError(
                "The 'prescribed_cell_mean' surface energy flux correction "
                "requires the ocean to predict hfds_total_area (per unit cell "
                "area), not hfds."
            )
        sea_surface_fraction = forcing.sea_surface_fraction
        runoff = (
            0.0 if runoff_heat_flux is None else runoff_heat_flux(sea_surface_fraction)
        )
        has_sea = (sea_surface_fraction > 0).to(net_flux.dtype)
        if under_ice is not None:
            storage_release = under_ice.storage_release(input_data, gen_data)
            out[hfds_name] = (net_flux + runoff + storage_release) * has_sea
            return out
        sea_ice_fraction = input.sea_ice_fraction
        prescribed = net_flux * (1 - sea_ice_fraction) + runoff
        out[hfds_name] = prescribed * has_sea + gen_hfds * sea_ice_fraction
        return out
    if hfds_name == "hfds_total_area":
        net_flux = net_flux * forcing.sea_surface_fraction
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
