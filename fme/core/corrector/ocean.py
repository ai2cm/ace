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


@dataclasses.dataclass
class SeaIceHfdsCorrectionConfig:
    """Extensive ``form_A_sis2`` correction of ``hfds_total_area`` over sea ice.

    Per hemisphere h, on the support S_h (frozen mass > 0 at both step
    endpoints, ocean cells)::

        r_c     = S_c - (m(k) D(k) - m(k-1) D(k-1)) / dt
        S       = lf_snowfl + minus_f_top + hfds + calving_residue + minus_hfrunoffds
        delta_c = -w_c sum_{S_h} a A r / sum_{S_h} a A w

    with a the grid area weight and A = sea_surface_fraction.

    so the corrected hfds closes the sea-ice energy budget integrated over S_h.
    Flux terms are read by ``surface_flux_term`` (gen_data, then forcing, then
    the AM4 equivalent where one exists).

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
        Q      = Q_hat + <F>_B - <Q_hat>_B            <x>_B: A-weighted mean on B ∩ M
        hfds   = Q - calving_residue - minus_hfrunoffds

    B are ``block_size`` x ``block_size`` lat-lon blocks of the local grid, so
    sum_{B ∩ M} a A Q = sum_{B ∩ M} a A F per block. Terms come from
    ``surface_flux_term`` (gen_data, forcing, AM4); "hfds" is the generated
    ``hfds_total_area``.

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
    """<X> = area_weighted_mean(A X) / area_weighted_mean(A), A =
    sea_surface_fraction; ``local`` gives the per-cell weight a A for block
    means (a = cos(lat), proportional to the lat-lon grid's area weight).
    """

    area_weighted_mean: AreaWeightedMean
    cos_lat: torch.Tensor  # (lat, 1), local latitude rows

    def field(self, input_data: TensorMapping, forcing_data: TensorMapping):
        return OceanData({**input_data, **forcing_data}).sea_surface_fraction

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
        support = (m0 > 0) & (m1 > 0) & (ssf > 0)
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
        anchored = (inp.ocean_fraction == 1) & (ssf > 0) & ~on_ice
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
        lat = None
        sefc = self.surface_energy_flux_correction
        if sefc is not None and (
            sefc.sea_ice is not None or sefc.open_ocean is not None
        ):
            lat = dataset_info.horizontal_coordinates.localize().lat_1d
        return self._build(
            dataset_info.gridded_operations,
            dataset_info.ocean_vertical_coordinate,
            dataset_info.timestep,
            lat=lat,
        )

    def _build(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasOceanDepthIntegral | None,
        timestep: datetime.timedelta,
        lat: torch.Tensor | None = None,
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
        sefc = self.surface_energy_flux_correction
        if sefc is not None:
            sea_ice, open_ocean = None, None
            if sefc.sea_ice is not None or sefc.open_ocean is not None:
                if lat is None:
                    raise ValueError("sea_ice/open_ocean corrections need latitudes")
                area_weight = AreaWeight(
                    area_weighted_mean,
                    torch.cos(torch.deg2rad(lat)).unsqueeze(-1),
                )
                if sefc.sea_ice is not None:
                    sea_ice = SeaIceHfdsCorrection(
                        sefc.sea_ice,
                        area_weight,
                        (lat > 0).unsqueeze(-1),
                        timestep_seconds,
                    )
                if sefc.open_ocean is not None:
                    open_ocean = OpenOceanAnchor(
                        sefc.open_ocean, area_weight, sefc.flux_source
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
