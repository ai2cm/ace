import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, Literal, Protocol

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
from fme.core.corrector.utils import ForcePositive, replace_value_keep_gradient
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import GriddedOperations
from fme.core.ocean_data import HasOceanDepthIntegral, OceanData
from fme.core.ocean_eos import (
    DELTA_RHO_THRESHOLD,
    MLD_REF_LAYER,
    _mixed_layer_depth,
    _sea_floor_depth,
)
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
class OceanHeatContentWeightsConfig:
    """Vertical-structure weights ``w`` for the ``weighted_temperature`` OHC
    correction, ``thetao_k,corr = thetao_k,gen + c * w_k``.

    Parameters:
        type: ``"mld"`` gives the mixed-layer weight
            ``w_ik = clamp((min(MLD_i, deptho_i) - z_top_k) / dz_ik, 0, 1)``
            (dimensionless), with ``MLD_i`` [m] the density-threshold mixed
            layer depth of the previous-step state ``(thetao_in, so_in)``,
            using the Wright (1997) density at zero pressure. ``"theta"`` gives
            ``w = thetao_gen`` [degC], the ``scaled_temperature`` shape.
        delta_rho_threshold: Density increase [kg/m**3] relative to the
            reference layer that defines the mixed layer base (``"mld"`` only).
        mld_ref_layer: Index of the reference layer for the density threshold
            (``"mld"`` only), dimensionless.
    """

    type: Literal["mld", "theta"] = "mld"
    delta_rho_threshold: float = DELTA_RHO_THRESHOLD
    mld_ref_layer: int = MLD_REF_LAYER

    def __post_init__(self):
        if self.mld_ref_layer < 0:
            raise ValueError(
                f"mld_ref_layer must be non-negative, got {self.mld_ref_layer}"
            )


@dataclasses.dataclass
class OceanHeatContentBudgetConfig:
    """Configuration for ocean heat content budget correction.

    Parameters:
        method: Method to use for OHC budget correction.
            "scaled_temperature" enforces conservation of heat content by
            scaling the predicted potential temperature by a vertically and
            horizontally uniform correction factor (dimensionless).
            "weighted_temperature" adds ``c * w`` to the predicted potential
            temperature, with ``w`` set by ``weights`` and the scalar ``c``
            [K per unit of ``w``] chosen so the global mean heat content
            closes the budget.
        constant_unaccounted_heating: Area-weighted global mean
            column-integrated heating in W/m**2 to be added to the energy flux
            into the ocean when conserving the heat content. This can be useful
            for correcting errors in heat budget in target data. The same
            additional heating is imposed at all time steps and grid cells.
        weights: Weights for "weighted_temperature"; unused otherwise.
        detach_weights: If True, ``w`` carries no gradient in
            "weighted_temperature"; ``c`` keeps its gradient.
    """

    method: Literal["scaled_temperature", "weighted_temperature"]
    constant_unaccounted_heating: float = 0.0
    weights: OceanHeatContentWeightsConfig = dataclasses.field(
        default_factory=OceanHeatContentWeightsConfig
    )
    detach_weights: bool = True


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
class ZosGlobalMeanCorrectionConfig:
    """Configuration for setting the global mean of generated sea surface
    height (``zos``) to a reference value each step.

    The global mean is weighted by cell area times ``sea_surface_fraction``
    (taken from forcing data) over the ``zos`` mask, so fractional coastal
    cells count by their ocean fraction.

    Parameters:
        reference_global_mean: Target global mean of ``zos`` in m.
    """

    reference_global_mean: float = 0.0


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
    method: Literal["scaled_temperature", "weighted_temperature"]
    unaccounted_heating: float
    weights: OceanHeatContentWeightsConfig = dataclasses.field(
        default_factory=OceanHeatContentWeightsConfig
    )
    detach_weights: bool = True

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
            weights=self.weights,
            detach_weights=self.detach_weights,
        )
        return corrected, corrector_state


@dataclasses.dataclass
class ZosGlobalMeanCorrection:
    """Correction that shifts ``zos`` uniformly so its
    sea-surface-fraction-weighted global mean equals ``reference_global_mean``.

    A no-op when ``zos`` is not in ``gen_data``.
    """

    area_weighted_mean: AreaWeightedMean
    reference_global_mean: float

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only ``zos``, or is empty when
            ``zos`` is absent from ``gen_data``.
        """
        if "zos" not in gen_data:
            return {}, corrector_state
        zos = gen_data["zos"]
        s = OceanData(forcing_data).sea_surface_fraction
        global_mean = self.area_weighted_mean(
            s * zos, keepdim=True, name="zos"
        ) / self.area_weighted_mean(s, keepdim=True, name="zos")
        return {"zos": zos - global_mean + self.reference_global_mean}, corrector_state


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
        zos_global_mean_correction: Optional configuration for setting the
            sea-surface-fraction-weighted global mean of the generated ``zos``
            to a reference value.
    """

    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: SeaIceFractionConfig | None = None
    surface_energy_flux_correction: SurfaceEnergyFluxCorrectionConfig | None = None
    ocean_heat_content_correction: OceanHeatContentBudgetConfig | None = None
    keep_gradient_through_clamps: bool = False
    zos_global_mean_correction: ZosGlobalMeanCorrectionConfig | None = None

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
                    weights=self.ocean_heat_content_correction.weights,
                    detach_weights=self.ocean_heat_content_correction.detach_weights,
                )
            )
        if self.zos_global_mean_correction is not None:
            corrections.append(
                ZosGlobalMeanCorrection(
                    area_weighted_mean,
                    self.zos_global_mean_correction.reference_global_mean,
                )
            )
        return OceanCorrector(corrections)


class OceanCorrector(CorrectionSequence):
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ):
        if "hfds_total_area" in forcing_data:
            # Prescribing hfds_total_area only changes the trajectory through
            # the SurfaceEnergyFluxCorrection early return feeding
            # OceanHeatContentCorrection; with either correction missing the
            # prescription would silently no-op on the dynamics.
            correction_types = {type(c) for c in self._corrections}
            required = {SurfaceEnergyFluxCorrection, OceanHeatContentCorrection}
            if not required <= correction_types:
                raise ValueError(
                    "hfds_total_area is prescribed via forcing_data, but the "
                    "ocean corrector is missing "
                    f"{[t.__name__ for t in required - correction_types]}; "
                    "the prescription requires both SurfaceEnergyFluxCorrection "
                    "and OceanHeatContentCorrection to be active."
                )
        return super().__call__(input_data, gen_data, forcing_data, corrector_state)


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
    # Hack: keys on "hfds_total_area" in forcing_data (supplied only when it is
    # in prescribed_prognostic_names) and is only valid for checkpoints emitting
    # hfds_total_area — a checkpoint emitting hfds would take a wrong path.
    if "hfds_total_area" in forcing_data:
        return {"hfds_total_area": forcing_data["hfds_total_area"]}
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


class _HasDepthGeometry(Protocol):
    idepth: torch.Tensor
    mask: torch.Tensor
    deptho: torch.Tensor | None

    @property
    def dz(self) -> torch.Tensor: ...

    def depth_integral(self, integrand: torch.Tensor) -> torch.Tensor: ...


@dataclasses.dataclass
class OceanHeatContentCorrectionDiagnostics:
    """Per-step diagnostics of the ``weighted_temperature`` OHC correction,
    detached, each of shape ``(n_samples, 1, 1)`` (global means keep dims).

    Parameters:
        raw_adv: ``-dE / dt`` [W/m**2], the heat-content tendency the
            prediction misses relative to the budget.
        c: Correction amplitude [K per unit of ``w``].
        denominator: ``< depth_integral(RHO_0 c_p w) >`` [J/m**2 per K].
        closure_residual: ``<OHC_corr> - <OHC_in> - (<F> + unaccounted) dt``
            [J/m**2].
        max_abs_dT: ``max abs(c w)`` over the grid [K].
        mean_mld: Area-weighted mean mixed layer depth [m]; None for
            ``theta`` weights.
    """

    raw_adv: torch.Tensor
    c: torch.Tensor
    denominator: torch.Tensor
    closure_residual: torch.Tensor
    max_abs_dT: torch.Tensor
    mean_mld: torch.Tensor | None


def _check_depth_geometry(vertical_coordinate: HasOceanDepthIntegral) -> None:
    if not all(
        hasattr(vertical_coordinate, a) for a in ("idepth", "mask", "dz", "deptho")
    ):
        raise ValueError(
            "weighted_temperature OHC correction requires a depth coordinate "
            "with idepth, mask, dz and deptho, got "
            f"{type(vertical_coordinate).__name__}."
        )


def _weighted_temperature_correction(
    input: OceanData,
    gen: OceanData,
    expected_change_ocean_heat_content: torch.Tensor,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
    weights: OceanHeatContentWeightsConfig,
    detach_weights: bool,
) -> tuple[TensorDict, OceanHeatContentCorrectionDiagnostics]:
    """``thetao_corr = thetao_gen + c w`` with
    ``c = dE / < depth_integral(RHO_0 c_p w) >``,
    ``dE = <OHC_in> + (<F> + unaccounted) dt - <OHC_gen>``.

    ``w`` is zero off ``mask_k`` and where ``dz`` is NaN, the support of
    ``DepthCoordinate.depth_integral``; ``dz`` is the coordinate's own, the
    same one ``OceanData.ocean_heat_content`` uses.
    """
    _check_depth_geometry(vertical_coordinate)
    coord: _HasDepthGeometry = vertical_coordinate  # type: ignore[assignment]
    thetao_gen = gen.sea_water_potential_temperature
    dz = coord.dz.to(thetao_gen.dtype)
    mask = coord.mask
    support = (mask > 0) & torch.isfinite(dz)
    mld: torch.Tensor | None = None
    if weights.type == "mld":
        idepth = coord.idepth.to(thetao_gen.dtype)
        deptho = _sea_floor_depth(idepth, mask, coord.deptho)
        mld = _mixed_layer_depth(
            input.sea_water_potential_temperature,
            input.sea_water_salinity,
            idepth,
            mask,
            deptho,
            weights.delta_rho_threshold,
            weights.mld_ref_layer,
        )
        support = support & (dz > 0)
        dz_safe = torch.where(support, dz, torch.ones_like(dz))
        m = torch.minimum(mld, deptho.expand(mld.shape)).unsqueeze(-1)
        w = torch.clamp((m - idepth[:-1]) / dz_safe, 0.0, 1.0)
        w = torch.where(support, w, torch.zeros_like(w))
        w_sst = w[..., 0]
    elif weights.type == "theta":
        support = support.expand(thetao_gen.shape)
        w = torch.where(support, thetao_gen, torch.zeros_like(thetao_gen))
        if "sst" in gen.data:
            w_sst = torch.where(
                support[..., 0],
                gen.data["sst"] - FREEZING_TEMPERATURE_KELVIN,
                torch.zeros_like(gen.data["sst"]),
            )
        else:
            w_sst = w[..., 0]
    else:
        raise NotImplementedError(f"OHC correction weights {weights.type!r}")
    if detach_weights:
        w = w.detach()
        w_sst = w_sst.detach()
    rho_cp = SPECIFIC_HEAT_OF_SEA_WATER_CM4 * DENSITY_OF_SEA_WATER_CM4
    denominator = area_weighted_mean(
        coord.depth_integral(rho_cp * w), keepdim=True, name="ocean_heat_content"
    )
    if not bool((denominator > 0).all()):
        raise ValueError(
            "weighted_temperature OHC correction denominator "
            "<depth_integral(RHO_0 c_p w)> must be positive, got "
            f"{denominator.detach().flatten().tolist()}."
        )
    global_input_ohc = area_weighted_mean(
        input.ocean_heat_content, keepdim=True, name="ocean_heat_content"
    )
    global_gen_ohc = area_weighted_mean(
        gen.ocean_heat_content, keepdim=True, name="ocean_heat_content"
    )
    dE = global_input_ohc + expected_change_ocean_heat_content - global_gen_ohc
    c = dE / denominator
    dT = c.unsqueeze(-1) * w
    out: TensorDict = {}
    for k in range(thetao_gen.shape[-1]):
        name = f"thetao_{k}"
        out[name] = gen.data[name] + dT[..., k]
    if "sst" in gen.data:
        out["sst"] = gen.data["sst"] + c * w_sst
    with torch.no_grad():
        corrected = OceanData(dict(out), vertical_coordinate)
        global_corr_ohc = area_weighted_mean(
            corrected.ocean_heat_content, keepdim=True, name="ocean_heat_content"
        )
        diagnostics = OceanHeatContentCorrectionDiagnostics(
            raw_adv=(-dE / timestep_seconds).detach(),
            c=c.detach(),
            denominator=denominator.detach(),
            closure_residual=(
                global_corr_ohc - global_input_ohc - expected_change_ocean_heat_content
            ).detach(),
            max_abs_dT=dT.detach().abs().amax(dim=(-3, -2, -1)).reshape(c.shape),
            mean_mld=(
                None
                if mld is None
                else area_weighted_mean(
                    mld.detach(), keepdim=True, name="ocean_heat_content"
                )
            ),
        )
    return out, diagnostics


def _net_energy_flux_into_ocean(
    input: OceanData, gen: OceanData, forcing: OceanData
) -> torch.Tensor:
    try:
        # First priority: pre-weighted heat flux in gen_data
        return (
            gen.net_downward_surface_heat_flux_total_area
            + forcing.geothermal_heat_flux * forcing.sea_surface_fraction
        )
    except KeyError:
        try:
            # Second priority: standard heat flux in gen_data
            return (
                gen.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction
        except KeyError:
            # Third priority: standard heat flux in input_data
            return (
                input.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction


def _force_conserve_ocean_heat_content(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
    method: Literal[
        "scaled_temperature", "weighted_temperature"
    ] = "scaled_temperature",
    unaccounted_heating: float = 0.0,
    weights: OceanHeatContentWeightsConfig | None = None,
    detach_weights: bool = True,
) -> TensorDict:
    if method not in ("scaled_temperature", "weighted_temperature"):
        raise NotImplementedError(
            f"Method {method!r} not implemented for ocean heat content conservation"
        )
    if method == "weighted_temperature":
        _check_depth_geometry(vertical_coordinate)
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
    net_energy_flux_into_ocean = _net_energy_flux_into_ocean(input, gen, forcing)
    energy_flux_global_mean = area_weighted_mean(
        net_energy_flux_into_ocean,
        keepdim=True,
        name="ocean_heat_content",
    )
    expected_change_ocean_heat_content = (
        energy_flux_global_mean + unaccounted_heating
    ) * timestep_seconds
    if method == "weighted_temperature":
        weighted_out, _ = _weighted_temperature_correction(
            input,
            gen,
            expected_change_ocean_heat_content,
            area_weighted_mean,
            vertical_coordinate,
            timestep_seconds,
            weights if weights is not None else OceanHeatContentWeightsConfig(),
            detach_weights,
        )
        return weighted_out
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
