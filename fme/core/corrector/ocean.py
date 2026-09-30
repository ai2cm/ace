import dataclasses
import datetime
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import torch

from fme.core.constants import FREEZING_TEMPERATURE_KELVIN, LATENT_HEAT_OF_FREEZING
from fme.core.corrector.registry import (
    Correction,
    CorrectionSequence,
    CorrectorConfigABC,
)
from fme.core.corrector.state import CorrectorState
from fme.core.corrector.utils import ForcePositive, replace_value_keep_gradient
from fme.core.dataset_info import DatasetInfo, MissingDatasetInfo
from fme.core.frozen_mass_budget import (
    correct_hfds,
    frozen_mass_flux_sum,
    ocean_net_surface_energy_flux,
)
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
      - "prescribed_open_ocean": corrected_hfds = net_flux where ocean_fraction
        == 1, and gen_hfds elsewhere. hfds is prescribed from forcings only on
        cells that are entirely ice-free ocean, and the network prediction
        passes through unweighted everywhere else.

    Parameters:
        method: Method to use for the correction.
        runoff_and_calving: If True, net_flux gains ``+ hfrunoffds -
            calving_residue`` (both per ocean area, read from the generated
            data) before any ``* sea_surface_fraction``. Default False leaves
            existing configs and checkpoints unchanged.

    """

    method: Literal["residual_prediction", "prescribed", "prescribed_open_ocean"]
    runoff_and_calving: bool = False


@dataclasses.dataclass
class FrozenMassBudgetConfig:
    """Configuration for the frozen-mass (sea ice + snow) budget correction,
    the ``floor_r`` or ``r_diag_floor`` form of the toy frozen-mass corrector.

    Per sample and hemisphere h (``lat >= 0``, ``lat < 0``), with
    ``<x>_h = sum(x * area * 1_h)`` and every field per total cell area::

        S      = frozen_mass_flux_sum(forcing, hfds_total_area, hfrunoffds,
                                      calving_residue)
        m_diag = m0 + (dt / L_f) (S - r_hat)
        m_pre  = m_hat                              floor_r
               = relu(m_diag)                       r_diag_floor
        D      = <m_diag - m_pre>_h                 (r_diag_floor: D <= 0)
        f      = sea_surface_fraction * sea_ice_fraction
        a      = relu(m_pre - c f),  s = m_pre - a
        dm     = D f / <f>_h                        D >= 0
               = D a / <a>_h                        -<a>_h <= D < 0
               = -a + (D + <a>_h) s / <s>_h         D < -<a>_h
        m_c    = relu(m_pre + dm)

    Each ratio is 0 where its denominator is 0. ``r_diag_floor`` discards
    ``m_hat`` and gives ``<m_c>_h = max(0, <m_diag>_h)`` in every hemisphere.
    ``m0`` is the input ``frozen_mass``; ``m_hat``, ``r_hat``,
    ``hfds_total_area``, ``hfrunoffds``, ``calving_residue`` and the sea ice
    fraction are generated; the atmosphere fluxes and
    ``sea_surface_fraction`` are forcing. Only ``frozen_mass`` is modified.

    Parameters:
        frozen_mass_name: Name of the frozen mass [kg m-2 per total area].
        residual_name: Name of the predicted budget residual [W m-2].
        sea_ice_fraction_name: Name of the sea ice fraction as a proportion of
            the sea surface.
        floor_mass_per_fraction: ``c`` [kg m-2], the mass per unit ice
            fraction kept from melting: SIS2 RHO_ICE * hLim(1).
        form: ``m_pre``, the field the increment is applied to: ``floor_r``
            (``m_hat``, the default) or ``r_diag_floor`` (``relu(m_diag)``).
    """

    frozen_mass_name: str = "frozen_mass"
    residual_name: str = "frozen_mass_energy_budget_residual"
    sea_ice_fraction_name: str = "ocean_sea_ice_fraction"
    floor_mass_per_fraction: float = 905.0 * 1.0e-10
    form: Literal["floor_r", "r_diag_floor"] = "floor_r"


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
    runoff_and_calving: bool = False

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
            runoff_and_calving=self.runoff_and_calving,
        )
        return corrected, corrector_state


def _hemisphere_masks(lat_1d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lat >= 0, lat < 0)`` as float64 ``(nlat, 1)`` masks that partition
    the grid.
    """
    lat = lat_1d.detach().to("cpu", torch.float64).reshape(-1, 1)
    return (lat >= 0).to(torch.float64), (lat < 0).to(torch.float64)


def _safe_share(
    field: torch.Tensor, total: torch.Tensor, amount: torch.Tensor
) -> torch.Tensor:
    """``amount * field / total``, 0 where ``total <= 0``; the denominator is
    replaced by 1 there so backward stays finite.
    """
    positive = total > 0
    safe_total = torch.where(positive, total, torch.ones_like(total))
    return torch.where(positive, amount * field / safe_total, torch.zeros_like(field))


def _floor_increment(
    deficit: torch.Tensor,
    ice_fraction: torch.Tensor,
    available: torch.Tensor,
    remainder: torch.Tensor,
    hemisphere_sum: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """The ``floor`` increment ``dm`` for one hemisphere, given its total
    ``D = deficit`` (shape ``(..., 1, 1)``).
    """
    total_available = hemisphere_sum(available)
    grow = _safe_share(ice_fraction, hemisphere_sum(ice_fraction), deficit)
    melt = _safe_share(available, total_available, deficit)
    melt_all = -available + _safe_share(
        remainder, hemisphere_sum(remainder), deficit + total_available
    )
    return torch.where(
        deficit >= 0,
        grow,
        torch.where(deficit >= -total_available, melt, melt_all),
    )


@dataclasses.dataclass
class FrozenMassBudgetCorrection:
    """Correction that sets each hemisphere's frozen mass to the budget
    integrated from the input with the predicted residual; see
    ``FrozenMassBudgetConfig``.

    The sums run over ``sea_surface_fraction > 0 & mask == 1`` when ``mask``
    (the dataset's ``frozen_mass`` mask) is given, else over
    ``sea_surface_fraction > 0``; elsewhere ``frozen_mass`` passes through.
    """

    config: FrozenMassBudgetConfig
    area_weighted_sum: AreaWeightedMean
    hemispheres: tuple[torch.Tensor, torch.Tensor]
    timestep_seconds: float
    mask: torch.Tensor | None = None

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only ``frozen_mass``.
        """
        c = self.config
        missing = [
            name
            for name in ("hfds_total_area", "hfrunoffds", "calving_residue")
            if name not in gen_data
        ]
        if missing:
            raise ValueError(
                f"Frozen mass budget correction needs {missing} in the generated data"
            )
        m_gen = gen_data[c.frozen_mass_name]
        out_dtype = m_gen.dtype
        ssf = torch.nan_to_num(OceanData(forcing_data).sea_surface_fraction).to(
            torch.float64
        )
        # The budget is over the ocean (sea_surface_fraction > 0) where the
        # frozen_mass mask is 1. Outside it the stored fields hold NaN and gen
        # holds network output (output masking runs after the corrector);
        # neither enters the hemisphere sums.
        region = ssf > 0
        if self.mask is not None:
            mask = self.mask.to(device=ssf.device, dtype=torch.float64)
            region = region & (mask == 1)

        def on_region(x: torch.Tensor) -> torch.Tensor:
            x = torch.nan_to_num(x.to(torch.float64))
            return torch.where(region, x, torch.zeros_like(x))

        m_hat = on_region(m_gen)
        m0 = on_region(input_data[c.frozen_mass_name])
        r_hat = on_region(gen_data[c.residual_name])
        f = ssf * on_region(gen_data[c.sea_ice_fraction_name])
        flux_sum = on_region(
            frozen_mass_flux_sum(
                forcing_data,
                gen_data["hfds_total_area"],
                gen_data["hfrunoffds"],
                gen_data["calving_residue"],
            )
        )
        m_diag = m0 + self.timestep_seconds / LATENT_HEAT_OF_FREEZING * (
            flux_sum - r_hat
        )
        m_pre = torch.relu(m_diag) if c.form == "r_diag_floor" else m_hat
        available = torch.relu(m_pre - c.floor_mass_per_fraction * f)
        remainder = m_pre - available
        dm = torch.zeros_like(m_pre)
        for hemisphere in self.hemispheres:
            if hemisphere.shape[-2] != m_hat.shape[-2]:
                raise ValueError(
                    f"Hemisphere mask has {hemisphere.shape[-2]} latitudes, data has "
                    f"{m_hat.shape[-2]}; spatial parallelism is not supported"
                )
            h = hemisphere.to(m_hat.device)

            def hemisphere_sum(x: torch.Tensor, h: torch.Tensor = h) -> torch.Tensor:
                return self.area_weighted_sum(x * h, keepdim=True)

            deficit = hemisphere_sum(m_diag - m_pre)
            dm = dm + h * _floor_increment(
                deficit, f, available, remainder, hemisphere_sum
            )
        m_c = torch.where(region, torch.relu(m_pre + dm).to(out_dtype), m_gen)
        return {c.frozen_mass_name: m_c}, corrector_state


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
        frozen_mass_budget_correction: Optional configuration for the
            frozen-mass budget (``floor``) correction, in the ``floor_r``
            (default) or ``r_diag_floor`` form (``FrozenMassBudgetConfig.form``).
            Needs latitudes, so it is not available on HEALPix grids.
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
    frozen_mass_budget_correction: FrozenMassBudgetConfig | None = None
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

    def _get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "OceanCorrector":
        lat_1d = (
            dataset_info.lat_1d
            if self.frozen_mass_budget_correction is not None
            else None
        )
        frozen_mass_mask = None
        if self.frozen_mass_budget_correction is not None:
            try:
                frozen_mass_mask = (
                    dataset_info.spatial_mask_provider.get_mask_tensor_for(
                        self.frozen_mass_budget_correction.frozen_mass_name
                    )
                )
            except MissingDatasetInfo:
                frozen_mass_mask = None
        return self._build(
            dataset_info.gridded_operations,
            dataset_info.ocean_vertical_coordinate,
            dataset_info.timestep,
            lat_1d=lat_1d,
            frozen_mass_mask=frozen_mass_mask,
        )

    def _build(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasOceanDepthIntegral | None,
        timestep: datetime.timedelta,
        lat_1d: torch.Tensor | None = None,
        frozen_mass_mask: torch.Tensor | None = None,
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
                SurfaceEnergyFluxCorrection(
                    self.surface_energy_flux_correction.method,
                    self.surface_energy_flux_correction.runoff_and_calving,
                )
            )
        if self.frozen_mass_budget_correction is not None:
            if lat_1d is None:
                raise ValueError(
                    "frozen_mass_budget_correction needs latitudes (lat_1d), "
                    "which this grid does not provide"
                )
            corrections.append(
                FrozenMassBudgetCorrection(
                    self.frozen_mass_budget_correction,
                    gridded_operations.area_weighted_sum,
                    _hemisphere_masks(lat_1d),
                    timestep_seconds,
                    mask=frozen_mass_mask,
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


_compute_ocean_net_surface_energy_flux = ocean_net_surface_energy_flux


def _correct_hfds(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    method: Literal["residual_prediction", "prescribed", "prescribed_open_ocean"],
    runoff_and_calving: bool = False,
) -> TensorDict:
    """Apply surface energy flux correction to the generated hfds.

    If ``runoff_and_calving``, net_flux gains ``+ hfrunoffds - calving_residue``
    from ``gen_data`` (per ocean area) before any ``* sea_surface_fraction``.

    The ocean_fraction naturally zeroes the correction on land and reduces
    it under sea ice. The arithmetic is ``fme.core.frozen_mass_budget.correct_hfds``,
    shared with the data loader's ``frozen_mass_energy_budget_residual``.

    Methods:
        residual_prediction: gen_hfds + ocean_fraction * net_flux
        prescribed: net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
        prescribed_open_ocean: net_flux where ocean_fraction == 1, else gen_hfds
    """
    input = OceanData(input_data)
    forcing = OceanData(forcing_data)
    net_flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, input.sea_surface_temperature
    )
    if "hfds" in gen_data:
        hfds_name = "hfds"
        sea_surface_fraction = None
    else:
        hfds_name = "hfds_total_area"
        sea_surface_fraction = forcing.sea_surface_fraction
    hfrunoffds: torch.Tensor | None = None
    calving_residue: torch.Tensor | None = None
    if runoff_and_calving:
        missing = [n for n in ("hfrunoffds", "calving_residue") if n not in gen_data]
        if missing:
            raise ValueError(
                f"runoff_and_calving is set, but {missing} are not in the "
                "generated data"
            )
        hfrunoffds = gen_data["hfrunoffds"]
        calving_residue = gen_data["calving_residue"]
    return {
        hfds_name: correct_hfds(
            net_flux,
            gen_data[hfds_name],
            input.ocean_fraction,
            method,
            sea_surface_fraction=sea_surface_fraction,
            hfrunoffds=hfrunoffds,
            calving_residue=calving_residue,
        )
    }


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
