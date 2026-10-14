import dataclasses
import datetime
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Protocol

import torch

from fme.core.atmosphere_data import AtmosphereData
from fme.core.constants import (
    FREEZING_TEMPERATURE_KELVIN,
    LATENT_HEAT_OF_VAPORIZATION,
    REFERENCE_SALINITY,
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
from fme.core.gridded_ops import GriddedOperations
from fme.core.ocean_data import HasOceanDepthIntegral, OceanData
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


def _require_sea_surface_fraction_weighting(
    budget_type: str, weight_by_sea_surface_fraction: bool
) -> None:
    if not weight_by_sea_surface_fraction:
        raise ValueError(
            f"The {budget_type!r} salt budget requires "
            "weight_by_sea_surface_fraction=True."
        )


@dataclasses.dataclass
class SeaSurfaceHeightSaltBudgetConfig:
    """Salt budget from the change of the sea surface height,
    -S_ref * sum((SSH_gen - SSH_input) * ssf * A). Requires ``SSH``, the height
    including its global mean, in the inputs and outputs.

    Parameters:
        reference_salinity_psu: Salinity at which the added water dilutes the
            salt, in psu.
        type: Selects this budget.
    """

    reference_salinity_psu: float = REFERENCE_SALINITY
    type: Literal["sea_surface_height"] = "sea_surface_height"

    def validate(self, weight_by_sea_surface_fraction: bool) -> None:
        _require_sea_surface_fraction_weighting(
            self.type, weight_by_sea_surface_fraction
        )

    def build(self, spatial_mask_provider: SpatialMaskProviderABC) -> SaltBudget:
        return SeaSurfaceHeightSaltBudget(self.reference_salinity_psu)


SaltBudgetConfig = SeaSurfaceHeightSaltBudgetConfig


@dataclasses.dataclass
class OceanSaltContentBudgetConfig:
    """Configuration for ocean salt content budget correction.

    Assumes area weights normalized over the whole sphere.

    Parameters:
        method: Method to use when applying the salt budget correction
            computed from the budget. "scaled_salinity". scales the predicted
            salinity by a vertically and horizontally uniform factor.
        budget_config: Budget for the expected change of the salt content
            over a step. None holds the salt content fixed, up to the constant
            term.
        constant_unaccounted_salting: Ocean area-weighted global mean rate of column
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

    With ``flux_source`` "sis2", net_flux (per total cell area) is the SIS2/OM4
    open-ocean flux instead of the AM4 one::

        Q        = SW + LW - LH - SH - L_f SNOWFL + P_h
        net_flux = Q - calving_residue + s hfrunoffds
        P_h      = s (hfrainds + hfevapds), else the AM4 precipitation heat

    with terms from ``surface_flux_term`` and s = sea_surface_fraction; it
    needs ``hfds_total_area`` in gen_data.

    Parameters:
        method: Method to use for the correction.
        flux_source: "am4" (AM4 forcing) or "sis2" (SIS2/OM4 fluxes); also the
            anchor flux F of ``open_ocean``.
        sea_ice: Optional extensive correction of the generated
            ``hfds_total_area`` over the sea-ice support; ``method`` then
            applies only off that support.
        open_ocean: Optional AM4-anchored correction of the generated
            ``hfds_total_area`` on open, non-coastal cells off the sea-ice
            support, in place of ``method`` there.
    """

    method: Literal["residual_prediction", "prescribed", "prescribed_open_ocean"]
    flux_source: "FluxSource" = "am4"
    sea_ice: "SeaIceHfdsCorrectionConfig | None" = None
    open_ocean: "OpenOceanAnchorConfig | None" = None


FluxSource = Literal["am4", "sis2"]

SIS2_LATENT_HEAT_OF_FUSION = 3.34e5  # J/kg, ecand3_transform.LF
SIS2_CP_ICE = 2100.0  # J/kg/K
SIS2_CP_WATER = 4200.0  # J/kg/K
SIS2_DTFREEZE_DS = -0.054  # degC/(g/kg)
SIS2_ICE_LAYER_SALINITY = (0.65, 2.35, 3.03, 3.19)  # g/kg

SeaIceFluxTerm = Literal[
    "lf_snowfl", "minus_f_top", "calving_residue", "minus_hfrunoffds"
]
SEA_ICE_FLUX_TERMS: tuple[SeaIceFluxTerm, ...] = (
    "lf_snowfl",
    "minus_f_top",
    "calving_residue",
    "minus_hfrunoffds",
)

AreaBasis = Literal["total_area", "ocean_area"]


def area_basis(names: Iterable[str]) -> AreaBasis:
    """Area basis of fields: "total_area" if every name ends "_total_area"
    (per total cell area), "ocean_area" if none does (per ocean area).

    Raises:
        ValueError: if the names mix the two bases.
    """
    names = list(names)
    total = [n.endswith("_total_area") for n in names]
    if all(total):
        return "total_area"
    if not any(total):
        return "ocean_area"
    raise ValueError(f"fields mix per-total-area and per-ocean-area bases: {names}")


@dataclasses.dataclass
class SeaIceHfdsCorrectionConfig:
    """Extensive ``form_A_sis2`` correction of ``hfds_total_area`` over sea ice.

    Per hemisphere h, on the support S_h (frozen mass > 0 at both step
    endpoints, ocean cells)::

        r_c     = S_c - (m(k) D(k) - m(k-1) D(k-1)) / dt
        S       = lf_snowfl + minus_f_top + hfds + calving_residue + minus_hfrunoffds
        X_h     = sum_{S_h} a A r
        delta_c = -w_c X_h / sum_{S_h} a A w

    with a the grid area weight and A set by ``area_basis``: A = 1 for fields
    per total cell area (names ending "_total_area"), A = sea_surface_fraction
    for fields per ocean area. Then sum_{S_h} a A (r + delta) = 0: the corrected
    hfds closes the sea-ice energy budget integrated over S_h. Flux terms are
    read by ``surface_flux_term`` (gen_data, then forcing, then the AM4
    equivalent where one exists), always per total cell area.

    Parameters:
        method: Budget form; only "form_A_sis2".
        weight: Spatial weight w: "uniform" (w = 1) or "sea_ice_fraction".
        frozen_mass_name: Frozen (ice + snow) mass per total cell area, kg/m**2;
            gen_data at k, input_data at k-1.
        ice_layer_temperature_names: SIS2 ice layer temperatures, degC.
        omit_terms: Flux terms to treat as zero when no source is found.
    """

    method: Literal["form_A_sis2"] = "form_A_sis2"
    weight: Literal["uniform", "sea_ice_fraction"] = "uniform"
    frozen_mass_name: str = "frozen_mass_total_area"
    ice_layer_temperature_names: list[str] = dataclasses.field(
        default_factory=lambda: ["T1", "T2", "T3", "T4"]
    )
    omit_terms: list[SeaIceFluxTerm] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        n = len(SIS2_ICE_LAYER_SALINITY)
        if len(self.ice_layer_temperature_names) != n:
            raise ValueError(f"ice_layer_temperature_names must have {n} entries")
        _ = self.area_basis  # raises on mixed basis

    @property
    def area_basis(self) -> AreaBasis:
        return area_basis(["hfds_total_area", self.frozen_mass_name])


OpenOceanQTerm = Literal[
    "hfds",
    "calving_residue",
    "minus_hfrunoffds",
    "f_top",
    "minus_lf_snowfl",
    "precipitation_heat",
]


@dataclasses.dataclass
class OpenOceanAnchorConfig:
    """AM4-anchored generated flux on open cells (issue 25 ``gen_shift_block{n}``).

    On M = (input ocean_fraction == 1 and ssf > 0) minus the sea-ice support::

        Q_hat  = sum of ``q_terms``                   generated SIS2-side flux
        F      = net surface flux x ssf     (AM4, or Q of ``flux_source`` "sis2")
        Q      = Q_hat + <F>_B - <Q_hat>_B            <x>_B: a A-weighted mean on B ∩ M
        hfds   = Q - calving_residue - minus_hfrunoffds

    B are ``block_size`` x ``block_size`` lat-lon blocks of the local grid, so
    sum_{B ∩ M} a A Q = sum_{B ∩ M} a A F per block, with A from ``area_basis``
    (1 per total cell area, sea_surface_fraction per ocean area). Terms come
    from ``surface_flux_term`` (gen_data, forcing, AM4; per total cell area);
    "hfds" is the generated ``hfds_total_area``.

    Parameters:
        q_terms: Terms summed into Q_hat; e.g. ["hfds", "calving_residue",
            "minus_hfrunoffds"] (Q_o) or ["f_top", "minus_lf_snowfl",
            "precipitation_heat"] (Q_a plus precipitation heat).
        block_size: Block edge n in grid cells; must divide the local grid.
        coastal: Cells off M and off the sea-ice support: "method" keeps the
            ``method`` result, "generated" keeps the generated hfds.
        omit_terms: Terms treated as zero when no source is found.
    """

    q_terms: list[OpenOceanQTerm]
    block_size: int = 5
    coastal: Literal["method", "generated"] = "method"
    omit_terms: list[OpenOceanQTerm] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if len(self.q_terms) == 0:
            raise ValueError("q_terms must not be empty")

    @property
    def area_basis(self) -> AreaBasis:
        """Every term is ``hfds_total_area`` or a ``surface_flux_term`` output."""
        return area_basis(["hfds_total_area"])


AnchoredFlux = Literal[
    "SW_total_area", "LW_total_area", "LH_total_area", "SH_total_area"
]
ANCHORED_FLUXES: tuple[AnchoredFlux, ...] = (
    "SW_total_area",
    "LW_total_area",
    "LH_total_area",
    "SH_total_area",
)
_AM4_FLUX_COUNTERPART: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # flux: (positive AM4 names, negative AM4 names); x^A = (sum pos - sum neg) ssf
    "SW_total_area": (("DSWRFsfc",), ("USWRFsfc",)),
    "LW_total_area": (("DLWRFsfc",), ("ULWRFsfc",)),
    "LH_total_area": (("LHTFLsfc",), ()),
    "SH_total_area": (("SHTFLsfc",), ()),
}


@dataclasses.dataclass
class HemispherePair:
    """A constant per hemisphere (north: lat > 0)."""

    north: float
    south: float


@dataclasses.dataclass
class CorrectedAM4FluxConfig:
    """AM4 surface fluxes mapped to SIS2-equivalent fluxes, per total cell area.

    With x^A the AM4 counterpart of x (see ``SurfaceFluxAnchorConfig``), h the
    hemisphere, and fields at the step endpoints k-1 (input) and k (gen)::

        c    = (osif(k-1) + osif(k)) / 2                       ice fraction of ssf
        T    = (TS(k-1) + TS(k)) / 2      degC, finite endpoints only;
               min(T_w, 0) if neither is finite
        T_w  = (sst(k-1) + sst(k)) / 2 - 273.15   degC; 0 if an endpoint is missing
        f_x  = x^A ssf + ssf (a_{x,h} c + b_{x,h} c^2)
               + [x = LH] k_h (L_f / L_v) LH^A ssf P
        P    = c e_i(T) / (c e_i(T) + (1 - c) e_w(T_w))        0 where c = 0
        e_i  = 6.112 exp(22.46 T / (272.62 + T))              hPa, over ice
        e_w  = 6.112 exp(17.62 T_w / (243.12 + T_w))          hPa, over water

    T is clipped to [-80, 0] and T_w to [-80, 40] degC before e_i, e_w (keeps
    them finite). Defaults: per-hemisphere least-squares fit of the pentad
    hemisphere integrals on interior sea ice, CM4 piControl and 1pctCO2
    (analysis frozen-deficit-sensible-brine 07, ``theta_interior_S``).

    Parameters:
        a: Coefficient of c ssf per flux and hemisphere, W/m**2.
        b: Coefficient of c**2 ssf per flux and hemisphere, W/m**2.
        k_latent: Weight k of the sublimation term on LH per hemisphere.
        surface_temperature_name: SIS2 ice surface temperature, degC.
        sea_ice_fraction_name: Sea-ice fraction of the sea surface.
    """

    a: dict[str, HemispherePair] = dataclasses.field(
        default_factory=lambda: {
            "SW_total_area": HemispherePair(-1.6304945127633705, -0.4022235549041929),
            "LW_total_area": HemispherePair(-0.3281091888012141, -0.007214586865773239),
            "LH_total_area": HemispherePair(0.9234031941869304, 3.1743228690416596),
            "SH_total_area": HemispherePair(-1.4440363502988236, -0.14606114629674224),
        }
    )
    b: dict[str, HemispherePair] = dataclasses.field(
        default_factory=lambda: {
            "SW_total_area": HemispherePair(1.7212766373926998, 0.4282568517898835),
            "LW_total_area": HemispherePair(0.37650403607943506, 0.006496533866703292),
            "LH_total_area": HemispherePair(-0.994746896743058, -3.7104199131610978),
            "SH_total_area": HemispherePair(1.65059116441646, 0.16054086385235647),
        }
    )
    k_latent: HemispherePair = dataclasses.field(
        default_factory=lambda: HemispherePair(0.8797973784044766, 0.6723897652236426)
    )
    surface_temperature_name: str = "TS"
    sea_ice_fraction_name: str = "ocean_sea_ice_fraction"

    def __post_init__(self):
        for label, coeffs in (("a", self.a), ("b", self.b)):
            missing = [f for f in ANCHORED_FLUXES if f not in coeffs]
            unknown = [f for f in coeffs if f not in ANCHORED_FLUXES]
            if missing or unknown:
                raise ValueError(
                    f"{label} needs exactly the keys {list(ANCHORED_FLUXES)}; "
                    f"missing {missing}, unknown {unknown}"
                )

    @property
    def prognostic_names(self) -> list[str]:
        """Read at both step endpoints, so in in_names and out_names."""
        return [self.surface_temperature_name, self.sea_ice_fraction_name, "sst"]


AnchorRegion = Literal["open_ocean", "open_ocean_and_ice"]


@dataclasses.dataclass
class SurfaceFluxAnchorConfig:
    """Block-anchors each generated SIS2 surface flux to an AM4-derived target.

    For x in ``fluxes``, on the anchor set K::

        x^A  = AM4 counterpart x ssf     (SW: DSWRFsfc - USWRFsfc,
                                          LW: DLWRFsfc - ULWRFsfc,
                                          LH: LHTFLsfc, SH: SHTFLsfc)
        f_x  = x^A                       ``target`` None
             = CorrectedAM4FluxConfig f_x  otherwise
        x    = x_hat + <f_x>_B - <x_hat>_B    on K;  x_hat off K

    <.>_B is the a A-weighted mean over B ∩ K, B the ``block_size`` x
    ``block_size`` lat-lon blocks of the local grid, so
    sum_{B ∩ K} a A x = sum_{B ∩ K} a A f_x per block and flux, A from
    ``area_basis`` (1 per total cell area).

    Regions, with S = (m(k-1) > 0 and m(k) > 0 and ssf > 0) the sea-ice support
    and M = (input ocean_fraction == 1 and ssf > 0) minus S::

        coast     = ssf < interior_min_sea_surface_fraction
                    or land_fraction > interior_max_land_fraction
        H         = ssf == 1 and land_fraction > 0      (land hidden in ssf)
        "open_ocean"          K = M
        "open_ocean_and_ice"  K = (M ∪ S) minus (coast and not H)

    Parameters:
        fluxes: Generated fluxes to anchor, per total cell area.
        block_size: Block edge n in grid cells; must divide the local grid.
        region: Anchor set K.
        frozen_mass_name: Frozen mass defining S, as in
            ``SeaIceHfdsCorrectionConfig``.
        target: AM4-to-SIS2 flux map f_x; None anchors to x^A.
        interior_min_sea_surface_fraction: ssf below this is coastal.
        interior_max_land_fraction: land_fraction above this is coastal.
    """

    fluxes: list[AnchoredFlux] = dataclasses.field(
        default_factory=lambda: list(ANCHORED_FLUXES)
    )
    block_size: int = 5
    region: AnchorRegion = "open_ocean"
    frozen_mass_name: str = "frozen_mass_total_area"
    target: CorrectedAM4FluxConfig | None = None
    interior_min_sea_surface_fraction: float = 0.999
    interior_max_land_fraction: float = 1e-6

    def __post_init__(self):
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if len(self.fluxes) == 0:
            raise ValueError("fluxes must not be empty")
        if len(set(self.fluxes)) != len(self.fluxes):
            raise ValueError(f"fluxes must be unique: {self.fluxes}")
        _ = self.area_basis  # raises on mixed basis

    @property
    def area_basis(self) -> AreaBasis:
        return area_basis(self.fluxes)

    @property
    def am4_names(self) -> list[str]:
        names: list[str] = []
        for flux in self.fluxes:
            positive, negative = _AM4_FLUX_COUNTERPART[flux]
            names.extend(positive + negative)
        return names

    def validate_names(
        self,
        out_names: Iterable[str],
        forcing_names: Iterable[str],
        in_names: Iterable[str],
    ):
        out_names, forcing_names = set(out_names), set(forcing_names)
        in_names = set(in_names)
        missing_out = [f for f in self.fluxes if f not in out_names]
        if missing_out:
            raise ValueError(
                f"surface_flux_anchor fluxes must be in out_names: {missing_out}"
            )
        am4_names = list(self.am4_names)
        if self.target is not None and "LH_total_area" in self.fluxes:
            am4_names += ["LHTFLsfc"]
        missing_forcing = [n for n in am4_names if n not in forcing_names]
        if missing_forcing:
            raise ValueError(
                "surface_flux_anchor needs AM4 fluxes in the corrector's forcing "
                f"(in_names or corrector_forcing_names): {missing_forcing}"
            )
        if self.target is not None:
            not_prognostic = [
                n
                for n in self.target.prognostic_names
                if n not in in_names or n not in out_names
            ]
            if not_prognostic:
                raise ValueError(
                    "surface_flux_anchor target needs these in in_names and "
                    f"out_names: {not_prognostic}"
                )
        if self.region == "open_ocean_and_ice" and "land_fraction" not in (
            in_names | forcing_names
        ):
            raise ValueError(
                "surface_flux_anchor region open_ocean_and_ice needs land_fraction "
                "in in_names or forcing"
            )


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
    flux_source: FluxSource = "am4"
    sea_ice: "SeaIceHfdsCorrection | None" = None
    open_ocean: "OpenOceanAnchor | None" = None

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
            flux_source=self.flux_source,
        )
        if self.sea_ice is None and self.open_ocean is None:
            return corrected, corrector_state
        hfds = corrected["hfds_total_area"]
        on_ice = torch.zeros_like(hfds, dtype=torch.bool)
        if self.sea_ice is not None:
            on_ice, hfds_on_ice = self.sea_ice(input_data, gen_data, forcing_data)
            hfds = torch.where(on_ice, hfds_on_ice, hfds)
        if self.open_ocean is not None:
            anchored, hfds_anchored = self.open_ocean(
                input_data, gen_data, forcing_data, on_ice
            )
            hfds = torch.where(anchored, hfds_anchored, hfds)
            if self.open_ocean.config.coastal == "generated":
                hfds = torch.where(
                    ~anchored & ~on_ice, gen_data["hfds_total_area"], hfds
                )
        corrected["hfds_total_area"] = hfds
        return corrected, corrector_state


@dataclasses.dataclass
class AreaWeight:
    """<X> = area_weighted_mean(A X) / area_weighted_mean(A), A = 1 for
    ``basis`` "total_area", sea_surface_fraction for "ocean_area"; ``local``
    gives the per-cell weight a A for block means (a = cos(lat), proportional
    to the lat-lon grid's area weight).
    """

    area_weighted_mean: AreaWeightedMean
    cos_lat: torch.Tensor  # (lat, 1), local latitude rows
    basis: AreaBasis

    def field(self, input_data: TensorMapping, forcing_data: TensorMapping):
        ssf = OceanData({**input_data, **forcing_data}).sea_surface_fraction
        return torch.ones_like(ssf) if self.basis == "total_area" else ssf

    def total(self, A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Proportional to sum a A x; same constant for every x."""
        return self.area_weighted_mean(A * x, keepdim=True)

    def local(self, A: torch.Tensor) -> torch.Tensor:
        return self.cos_lat.to(A.device, A.dtype) * A


@dataclasses.dataclass
class SeaIceHfdsCorrection:
    """Applies ``SeaIceHfdsCorrectionConfig``; see its docstring for the math."""

    config: SeaIceHfdsCorrectionConfig
    area_weight: AreaWeight
    north: torch.Tensor  # (lat, 1) bool, local latitude rows
    timestep_seconds: float

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            The support S (bool) and the corrected ``hfds_total_area`` on it.
        """
        cfg = self.config
        if "hfds_total_area" not in gen_data:
            raise KeyError("sea_ice hfds correction needs hfds_total_area in gen_data")
        hfds = gen_data["hfds_total_area"]
        ssf = OceanData({**input_data, **forcing_data}).sea_surface_fraction
        m0 = input_data[cfg.frozen_mass_name]
        m1 = gen_data[cfg.frozen_mass_name]
        source = hfds
        for term in SEA_ICE_FLUX_TERMS:
            try:
                source = source + surface_flux_term(term, gen_data, forcing_data, ssf)
            except KeyError:
                if term not in cfg.omit_terms:
                    raise
        storage = (
            m1 * _frozen_deficit(gen_data, m1, cfg.ice_layer_temperature_names)
            - m0 * _frozen_deficit(input_data, m0, cfg.ice_layer_temperature_names)
        ) / self.timestep_seconds
        residual = source - storage
        support = _sea_ice_support(m0, m1, ssf)
        if cfg.weight == "uniform":
            weight = torch.ones_like(hfds)
        else:
            weight = OceanData(
                {**input_data, **forcing_data, **gen_data}
            ).sea_ice_fraction
        north = self.north.to(hfds.device)
        A = self.area_weight.field(input_data, forcing_data)
        delta = torch.zeros_like(hfds)
        for hemi in (north, ~north):
            on = support & hemi
            X = self.area_weight.total(A, torch.where(on, residual, 0.0))
            aw = self.area_weight.total(A, torch.where(on, weight, 0.0))
            scale = torch.where(aw > 0, -X / torch.where(aw > 0, aw, 1.0), 0.0)
            delta = torch.where(on, weight * scale, delta)
        return support, hfds + delta


@dataclasses.dataclass
class OpenOceanAnchor:
    """Applies ``OpenOceanAnchorConfig``; see its docstring for the math."""

    config: OpenOceanAnchorConfig
    area_weight: AreaWeight
    flux_source: FluxSource = "am4"

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        on_ice: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            The anchored cells M (bool) and the corrected ``hfds_total_area``.
        """
        cfg = self.config
        if "hfds_total_area" not in gen_data:
            raise KeyError("open_ocean hfds correction needs hfds_total_area")
        hfds = gen_data["hfds_total_area"]
        inp = OceanData({**input_data, **forcing_data})
        ssf = inp.sea_surface_fraction
        sst = inp.sea_surface_temperature

        def term(name: str) -> torch.Tensor:
            if name == "hfds":
                return hfds
            try:
                return OPEN_OCEAN_Q_SIGN[name] * surface_flux_term(
                    OPEN_OCEAN_Q_SOURCE[name], gen_data, forcing_data, ssf, sst=sst
                )
            except KeyError:
                if name in cfg.omit_terms:
                    return torch.zeros_like(hfds)
                raise

        q_hat = sum((term(n) for n in cfg.q_terms), torch.zeros_like(hfds))
        F = _open_ocean_q(self.flux_source, gen_data, forcing_data, ssf, sst)
        anchored = _open_ocean_cells(inp, on_ice)
        w = torch.where(
            anchored,
            self.area_weight.local(self.area_weight.field(input_data, forcing_data)),
            0.0,
        )
        q = (
            q_hat
            + block_mean(F, w, cfg.block_size)
            - block_mean(q_hat, w, cfg.block_size)
        )
        return anchored, q - term("calving_residue") - term("minus_hfrunoffds")


def _sea_ice_support(
    m0: torch.Tensor, m1: torch.Tensor, ssf: torch.Tensor
) -> torch.Tensor:
    """S: frozen mass > 0 at both step endpoints, on ocean cells."""
    return (m0 > 0) & (m1 > 0) & (ssf > 0)


def _open_ocean_cells(inp: OceanData, on_ice: torch.Tensor) -> torch.Tensor:
    """M: input ocean_fraction == 1 and ssf > 0, off ``on_ice``."""
    return (inp.ocean_fraction == 1) & (inp.sea_surface_fraction > 0) & ~on_ice


@dataclasses.dataclass
class SurfaceFluxAnchor:
    """Applies ``SurfaceFluxAnchorConfig``; see its docstring for the math."""

    config: SurfaceFluxAnchorConfig
    area_weight: AreaWeight
    north: torch.Tensor  # (lat, 1) bool, local latitude rows

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        """
        Returns:
            A tuple whose ``TensorDict`` contains only the anchored fluxes.
        """
        cfg = self.config
        inp = OceanData({**input_data, **forcing_data})
        ssf = inp.sea_surface_fraction
        on_ice = _sea_ice_support(
            input_data[cfg.frozen_mass_name], gen_data[cfg.frozen_mass_name], ssf
        )
        K = _open_ocean_cells(inp, on_ice)
        if cfg.region == "open_ocean_and_ice":
            K = (K | on_ice) & ~_coast_without_hidden_land(
                ssf,
                inp.land_fraction,
                cfg.interior_min_sea_surface_fraction,
                cfg.interior_max_land_fraction,
            )
        w = torch.where(
            K,
            self.area_weight.local(self.area_weight.field(input_data, forcing_data)),
            0.0,
        )
        x_am4: dict[str, torch.Tensor] = {
            flux: _am4_flux(flux, forcing_data, ssf, gen_data[flux])
            for flux in cfg.fluxes
        }
        if cfg.target is None:
            target = x_am4
        else:
            target = corrected_am4_fluxes(
                cfg.target,
                x_am4,
                forcing_data["LHTFLsfc"] * ssf,
                input_data,
                gen_data,
                ssf,
                self.north.to(ssf.device),
            )
        out: TensorDict = {}
        for flux in cfg.fluxes:
            x_hat = gen_data[flux]
            anchored = (
                x_hat
                + block_mean(target[flux], w, cfg.block_size)
                - block_mean(x_hat, w, cfg.block_size)
            )
            out[flux] = torch.where(K, anchored, x_hat)
        return out, corrector_state


def _am4_flux(
    flux: str, forcing_data: TensorMapping, ssf: torch.Tensor, like: torch.Tensor
) -> torch.Tensor:
    """x^A per total cell area."""
    positive, negative = _AM4_FLUX_COUNTERPART[flux]
    return (
        sum((forcing_data[n] for n in positive), torch.zeros_like(like))
        - sum((forcing_data[n] for n in negative), torch.zeros_like(like))
    ) * ssf


def _coast_without_hidden_land(
    ssf: torch.Tensor,
    land_fraction: torch.Tensor,
    min_ssf: float,
    max_land_fraction: float,
) -> torch.Tensor:
    """Coast and not H; see ``SurfaceFluxAnchorConfig``."""
    coast = (ssf < min_ssf) | (land_fraction > max_land_fraction)
    hidden_land = (ssf == 1) & (land_fraction > 0)
    return coast & ~hidden_land


_T_MIN_CELSIUS, _T_MAX_CELSIUS = -80.0, 40.0


def _saturation_vapor_pressure_ice(T: torch.Tensor) -> torch.Tensor:
    return 6.112 * torch.exp(22.46 * T / (272.62 + T))


def _saturation_vapor_pressure_water(T: torch.Tensor) -> torch.Tensor:
    return 6.112 * torch.exp(17.62 * T / (243.12 + T))


def _endpoint_mean_finite(t0: torch.Tensor, t1: torch.Tensor):
    """Mean of the finite endpoints and whether any endpoint was finite."""
    f0, f1 = torch.isfinite(t0), torch.isfinite(t1)
    n = f0.to(t0.dtype) + f1.to(t0.dtype)
    total = torch.where(f0, t0, 0.0) + torch.where(f1, t1, 0.0)
    return total / torch.clamp(n, min=1.0), n > 0


def corrected_am4_fluxes(
    config: CorrectedAM4FluxConfig,
    x_am4: Mapping[str, torch.Tensor],
    lh_am4: torch.Tensor,
    input_data: TensorMapping,
    gen_data: TensorMapping,
    ssf: torch.Tensor,
    north: torch.Tensor,
) -> TensorDict:
    """f_x of ``CorrectedAM4FluxConfig`` for each flux in ``x_am4``.

    Args:
        config: Constants and field names.
        x_am4: x^A per total cell area, by flux name.
        lh_am4: LH^A per total cell area (LHTFLsfc ssf).
        input_data: Fields at k-1.
        gen_data: Fields at k.
        ssf: Sea surface fraction.
        north: Northern-hemisphere mask, broadcastable to the fields.
    """
    c = 0.5 * (
        input_data[config.sea_ice_fraction_name]
        + gen_data[config.sea_ice_fraction_name]
    )
    sst0, sst1 = input_data["sst"], gen_data["sst"]
    sst_ok = torch.isfinite(sst0) & torch.isfinite(sst1) & (sst0 > 0) & (sst1 > 0)
    T_w = torch.where(
        sst_ok,
        0.5 * (torch.where(sst_ok, sst0, 0.0) + torch.where(sst_ok, sst1, 0.0))
        - FREEZING_TEMPERATURE_KELVIN,
        0.0,
    )
    T, any_finite = _endpoint_mean_finite(
        input_data[config.surface_temperature_name],
        gen_data[config.surface_temperature_name],
    )
    T = torch.where(any_finite, T, torch.clamp(T_w, max=0.0))
    T = torch.clamp(T, _T_MIN_CELSIUS, 0.0)
    T_w = torch.clamp(T_w, _T_MIN_CELSIUS, _T_MAX_CELSIUS)
    num = c * _saturation_vapor_pressure_ice(T)
    den = num + (1 - c) * _saturation_vapor_pressure_water(T_w)
    sublimation_share = torch.where(den > 0, num / torch.where(den > 0, den, 1.0), 0.0)

    def per_hemisphere(pair: HemispherePair) -> torch.Tensor:
        return torch.where(north, c.new_tensor(pair.north), c.new_tensor(pair.south))

    out: TensorDict = {}
    for flux, xa in x_am4.items():
        a, b = per_hemisphere(config.a[flux]), per_hemisphere(config.b[flux])
        f = xa + ssf * (a * c + b * c**2)
        if flux == "LH_total_area":
            f = (
                f
                + per_hemisphere(config.k_latent)
                * (SIS2_LATENT_HEAT_OF_FUSION / LATENT_HEAT_OF_VAPORIZATION)
                * lh_am4
                * sublimation_share
            )
        out[flux] = f
    return out


def block_mean(x: torch.Tensor, w: torch.Tensor, n: int) -> torch.Tensor:
    """Per-cell mean of x weighted by w over its n x n lat-lon block; 0 where
    the block weight is 0.
    """
    ny, nx = x.shape[-2:]
    if ny % n or nx % n:
        raise ValueError(f"block_size {n} does not divide the local grid {(ny, nx)}")
    lead = x.shape[:-2]
    shape = (*lead, ny // n, n, nx // n, n)
    w = w.expand_as(x)
    num = (x * w).reshape(shape).sum(dim=(-3, -1))
    den = w.reshape(shape).sum(dim=(-3, -1))
    mean = torch.where(den > 0, num / torch.where(den > 0, den, 1.0), 0.0)
    return mean.repeat_interleave(n, dim=-2).repeat_interleave(n, dim=-1)


def sis2_ice_deficit(layer_temperatures: list[torch.Tensor]) -> torch.Tensor:
    """SIS2 ice enthalpy deficit D_ice, J/kg: the layer mean of
    h_liq_fr(S) - enth_from_TS(T, S) with the hard-coded layer salinities
    (``ecand3_transform.ice_deficit``). Temperatures in degC.
    """
    total = torch.zeros_like(layer_temperatures[0])
    for T, salinity in zip(layer_temperatures, SIS2_ICE_LAYER_SALINITY):
        t_fr = SIS2_DTFREEZE_DS * salinity
        a = -t_fr
        t = torch.clamp(-T, min=a)  # brine branch only where T < t_fr; keeps log finite
        brine = (
            SIS2_LATENT_HEAT_OF_FUSION * (1.0 - a / t)
            + SIS2_CP_ICE * (t - a)
            + (SIS2_CP_WATER - SIS2_CP_ICE) * a * torch.log(t / a)
        )
        liquid = SIS2_CP_WATER * (t_fr - T)
        total = total + torch.where(T >= t_fr, liquid, brine)
    return total / len(SIS2_ICE_LAYER_SALINITY)


def _frozen_deficit(
    data: TensorMapping, frozen_mass: torch.Tensor, temperature_names: list[str]
) -> torch.Tensor:
    """D: D_ice where frozen mass > 0, L_f elsewhere."""
    d_ice = sis2_ice_deficit([data[n] for n in temperature_names])
    return torch.where(
        frozen_mass > 0, d_ice, torch.full_like(d_ice, SIS2_LATENT_HEAT_OF_FUSION)
    )


_SIS2_FLUX_SOURCES: dict[str, list[tuple[tuple[str, ...], Any]]] = {
    "lf_snowfl": [
        (
            ("SNOWFL_total_area",),
            lambda d, ssf: SIS2_LATENT_HEAT_OF_FUSION * d["SNOWFL_total_area"],
        ),
    ],
    "minus_f_top": [
        (
            ("SW_total_area", "LW_total_area", "LH_total_area", "SH_total_area"),
            lambda d, ssf: -(
                d["SW_total_area"]
                + d["LW_total_area"]
                - d["LH_total_area"]
                - d["SH_total_area"]
            ),
        ),
    ],
    "calving_residue": [
        (
            ("calving_residue_total_area",),
            lambda d, ssf: d["calving_residue_total_area"],
        ),
        (
            ("hflso", "evs", "prsn"),
            lambda d, ssf: (
                -d["hflso"]
                + LATENT_HEAT_OF_VAPORIZATION * d["evs"]
                - SIS2_LATENT_HEAT_OF_FUSION * d["prsn"]
            )
            * ssf,
        ),
    ],
    "minus_hfrunoffds": [
        (("hfrunoffds",), lambda d, ssf: -d["hfrunoffds"] * ssf),
    ],
}

_AM4_FLUX_SOURCES: dict[str, list[tuple[tuple[str, ...], Any]]] = {
    "lf_snowfl": [
        (
            ("total_frozen_precipitation_rate",),
            lambda d, ssf: SIS2_LATENT_HEAT_OF_FUSION
            * d["total_frozen_precipitation_rate"]
            * ssf,
        ),
    ],
    "minus_f_top": [
        (
            (
                "DSWRFsfc",
                "USWRFsfc",
                "DLWRFsfc",
                "ULWRFsfc",
                "LHTFLsfc",
                "SHTFLsfc",
            ),
            lambda d, ssf: -(
                d["DSWRFsfc"]
                - d["USWRFsfc"]
                + d["DLWRFsfc"]
                - d["ULWRFsfc"]
                - d["LHTFLsfc"]
                - d["SHTFLsfc"]
            )
            * ssf,
        ),
    ],
}


_AM4_FLUX_SOURCES["precipitation_heat"] = [
    (
        ("sst",),
        lambda d, ssf: _precipitation_heat_flux(d, d["sst"]) * ssf,
    ),
]
_SIS2_FLUX_SOURCES["precipitation_heat"] = [
    (
        ("precipitation_heat_total_area",),
        lambda d, ssf: d["precipitation_heat_total_area"],
    ),
    (
        ("hfrainds", "hfevapds"),
        lambda d, ssf: (d["hfrainds"] + d["hfevapds"]) * ssf,
    ),
]

OPEN_OCEAN_Q_SOURCE = {
    "calving_residue": "calving_residue",
    "minus_hfrunoffds": "minus_hfrunoffds",
    "f_top": "minus_f_top",
    "minus_lf_snowfl": "lf_snowfl",
    "precipitation_heat": "precipitation_heat",
}
OPEN_OCEAN_Q_SIGN = {
    "calving_residue": 1.0,
    "minus_hfrunoffds": 1.0,
    "f_top": -1.0,
    "minus_lf_snowfl": -1.0,
    "precipitation_heat": 1.0,
}


def surface_flux_term(
    term: str,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    sea_surface_fraction: torch.Tensor,
    sst: torch.Tensor | None = None,
) -> torch.Tensor:
    """One sea-ice budget flux term, W/m**2 per total cell area.

    Lookup order, first hit wins: SIS2 fields in gen_data, SIS2 fields in
    forcing_data, AM4 equivalent in forcing_data (with ``sst``, K, where the
    AM4 form needs it). Per-ocean-area fields are scaled by
    ``sea_surface_fraction``.

    Raises:
        KeyError: if no source provides the term.
    """
    am4_data = forcing_data if sst is None else {**forcing_data, "sst": sst}
    for data, sources in (
        (gen_data, _SIS2_FLUX_SOURCES),
        (forcing_data, _SIS2_FLUX_SOURCES),
        (am4_data, _AM4_FLUX_SOURCES),
    ):
        for names, combine in sources.get(term, []):
            if all(n in data for n in names):
                return combine(data, sea_surface_fraction)
    raise KeyError(f"no source for sea-ice flux term {term!r}")


def _open_ocean_q(
    flux_source: FluxSource,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    sea_surface_fraction: torch.Tensor,
    sst: torch.Tensor,
) -> torch.Tensor:
    """Open-ocean net surface energy flux Q before calving and runoff heat,
    W/m**2 per total cell area.
    """
    if flux_source == "am4":
        return (
            _compute_ocean_net_surface_energy_flux(forcing_data, sst)
            * sea_surface_fraction
        )
    args = (gen_data, forcing_data, sea_surface_fraction)
    return (
        -surface_flux_term("minus_f_top", *args)
        - surface_flux_term("lf_snowfl", *args)
        + surface_flux_term("precipitation_heat", *args, sst=sst)
    )


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
        ocean_salt_content_correction: Optional configuration for an ocean salt
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
        surface_flux_anchor: Optional block anchor of generated SIS2 surface
            fluxes to AM4 or corrected AM4; runs before
            ``surface_energy_flux_correction`` so it reads the anchored fluxes.
    """

    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: SeaIceFractionConfig | None = None
    surface_energy_flux_correction: SurfaceEnergyFluxCorrectionConfig | None = None
    ocean_heat_content_correction: OceanHeatContentBudgetConfig | None = None
    ocean_salt_content_correction: OceanSaltContentBudgetConfig | None = None
    keep_gradient_through_clamps: bool = False
    zos_global_mean_correction: ZosGlobalMeanCorrectionConfig | None = None
    surface_flux_anchor: SurfaceFluxAnchorConfig | None = None

    def validate_names(
        self,
        out_names: Iterable[str],
        forcing_names: Iterable[str],
        in_names: Iterable[str],
    ) -> None:
        if self.surface_flux_anchor is not None:
            self.surface_flux_anchor.validate_names(out_names, forcing_names, in_names)

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
        lat = None
        sefc = self.surface_energy_flux_correction
        if self.surface_flux_anchor is not None or (
            sefc is not None
            and (sefc.sea_ice is not None or sefc.open_ocean is not None)
        ):
            lat = dataset_info.horizontal_coordinates.localize().lat_1d
        spatial_mask_provider: SpatialMaskProviderABC
        try:
            spatial_mask_provider = dataset_info.spatial_mask_provider
        except MissingDatasetInfo:
            spatial_mask_provider = NullSpatialMaskProvider
        return self._build(
            dataset_info.gridded_operations,
            dataset_info.ocean_vertical_coordinate,
            dataset_info.timestep,
            lat=lat,
            spatial_mask_provider=spatial_mask_provider,
        )

    def _build(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasOceanDepthIntegral | None,
        timestep: datetime.timedelta,
        lat: torch.Tensor | None = None,
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
        if self.surface_flux_anchor is not None:
            if lat is None:
                raise ValueError("surface_flux_anchor needs latitudes")
            corrections.append(
                SurfaceFluxAnchor(
                    self.surface_flux_anchor,
                    AreaWeight(
                        area_weighted_mean,
                        torch.cos(torch.deg2rad(lat)).unsqueeze(-1),
                        self.surface_flux_anchor.area_basis,
                    ),
                    (lat > 0).unsqueeze(-1),
                )
            )
        sefc = self.surface_energy_flux_correction
        if sefc is not None:
            sea_ice, open_ocean = None, None
            if sefc.sea_ice is not None or sefc.open_ocean is not None:
                if lat is None:
                    raise ValueError("sea_ice/open_ocean corrections need latitudes")
                cos_lat = torch.cos(torch.deg2rad(lat)).unsqueeze(-1)
                if sefc.sea_ice is not None:
                    sea_ice = SeaIceHfdsCorrection(
                        sefc.sea_ice,
                        AreaWeight(
                            area_weighted_mean, cos_lat, sefc.sea_ice.area_basis
                        ),
                        (lat > 0).unsqueeze(-1),
                        timestep_seconds,
                    )
                if sefc.open_ocean is not None:
                    open_ocean = OpenOceanAnchor(
                        sefc.open_ocean,
                        AreaWeight(
                            area_weighted_mean, cos_lat, sefc.open_ocean.area_basis
                        ),
                        sefc.flux_source,
                    )
            corrections.append(
                SurfaceEnergyFluxCorrection(
                    sefc.method, sefc.flux_source, sea_ice, open_ocean
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
        if self.zos_global_mean_correction is not None:
            corrections.append(
                ZosGlobalMeanCorrection(
                    area_weighted_mean,
                    self.zos_global_mean_correction.reference_global_mean,
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
    return base_flux + _precipitation_heat_flux(forcing_data, sst)


def _precipitation_heat_flux(
    forcing_data: TensorMapping, sst: torch.Tensor
) -> torch.Tensor:
    """Heat carried by precipitation and evaporation at the SST, W/m**2.

    precipitation_rate is total (liquid + frozen) precipitation.
    """
    atmos = AtmosphereData(forcing_data)
    return (
        SPECIFIC_HEAT_OF_SEA_WATER_CM4
        * (
            atmos.precipitation_rate
            - (atmos.latent_heat_flux / LATENT_HEAT_OF_VAPORIZATION)
        )  # missing: + river runoff + calving
        * (sst - FREEZING_TEMPERATURE_KELVIN)
    )


def _correct_hfds(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    method: Literal["residual_prediction", "prescribed", "prescribed_open_ocean"],
    flux_source: FluxSource = "am4",
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
    sst = input.sea_surface_temperature
    out: TensorDict = {}
    hfds_name = "hfds" if "hfds" in gen_data else "hfds_total_area"
    if flux_source == "sis2":
        if hfds_name == "hfds":
            raise ValueError("flux_source 'sis2' needs hfds_total_area in gen_data")
        ssf = forcing.sea_surface_fraction
        net_flux = (
            _open_ocean_q("sis2", gen_data, forcing_data, ssf, sst)
            - surface_flux_term("calving_residue", gen_data, forcing_data, ssf)
            - surface_flux_term("minus_hfrunoffds", gen_data, forcing_data, ssf)
        )
    else:
        net_flux = _compute_ocean_net_surface_energy_flux(forcing_data, sst)
        if hfds_name == "hfds_total_area":
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
    """Scale the generated temperature to conserve global ocean heat content.

    The scaling closes

        sum(a s H_corrected) = sum(a s H_in) + sum(a F) dt,

    with a the cell area, s the sea surface fraction, H the column heat
    content per unit ocean area and F the net energy flux into the ocean
    (including unaccounted heating) per unit total cell area.
    """
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
        gen.ocean_heat_content * forcing.sea_surface_fraction,
        keepdim=True,
        name="ocean_heat_content",
    )
    global_input_ocean_heat_content = area_weighted_mean(
        input.ocean_heat_content * forcing.sea_surface_fraction,
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
