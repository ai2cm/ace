from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from fme.core.area_weights_provider import (
    AreaWeightsProviderABC,
    NullAreaWeightsProvider,
)
from fme.core.constants import (
    DENSITY_OF_SEA_WATER_CM4,
    FREEZING_TEMPERATURE_KELVIN,
    REFERENCE_SALINITY,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.distributed import Distributed
from fme.core.ocean_eos import (
    DELTA_RHO_THRESHOLD,
    MLD_REF_LAYER,
    RHO_0,
    column_density_integral,
    density_anomaly,
    mixed_layer_depth,
    sea_floor_depth,
)
from fme.core.stacker import Stacker
from fme.core.typing_ import TensorDict, TensorMapping

OCEAN_FIELD_NAME_PREFIXES = MappingProxyType(
    {
        "sea_water_potential_temperature": ["thetao_"],
        "sea_water_salinity": ["so_"],
        "sea_water_x_velocity": ["uo_"],
        "sea_water_y_velocity": ["vo_"],
        "sea_surface_height_above_geoid": ["zos"],
        "sea_surface_height": ["SSH"],
        "sea_surface_temperature": ["sst", "SST_degK"],
        "sea_ice_fraction": ["sea_ice_fraction"],
        "sea_ice_thickness": ["HI"],
        "sea_ice_volume": ["sea_ice_volume"],
        "ocean_sea_ice_thickness_mean": ["ocean_sea_ice_thickness_mean"],
        "ocean_sea_ice_fraction": ["ocean_sea_ice_fraction"],
        "land_fraction": ["land_fraction", "LANDFRAC"],
        "net_downward_surface_heat_flux": ["hfds"],
        "net_downward_surface_heat_flux_total_area": ["hfds_total_area"],
        "geothermal_heat_flux": ["hfgeou"],
        "water_flux_into_sea_water": ["wfo"],
        "downward_sea_ice_basal_salt_flux": ["sfdsi"],
        "sea_surface_fraction": ["sea_surface_fraction"],
    }
)


@runtime_checkable
class HasOceanDepthIntegral(Protocol):
    def depth_integral(
        self,
        integrand: torch.Tensor,
    ) -> torch.Tensor: ...


@runtime_checkable
class HasOceanLayerGeometry(Protocol):
    """Layer geometry of a depth coordinate, as needed by the wright97
    variables.
    """

    @property
    def idepth(self) -> torch.Tensor: ...

    @property
    def mask(self) -> torch.Tensor: ...

    @property
    def dz(self) -> torch.Tensor: ...

    @property
    def deptho(self) -> torch.Tensor | None: ...


class HasCellAreaInMetersSquared(Protocol):
    """Protocol for objects that can provide cell areas in square meters."""

    @property
    def area_weights_m2(self) -> torch.Tensor: ...


class OceanData:
    """Container for ocean data for accessing variables and providing
    torch.Tensor views on data with multiple depth levels.
    """

    def __init__(
        self,
        ocean_data: TensorMapping,
        depth_coordinate: HasOceanDepthIntegral | None = None,
        ocean_field_name_prefixes: Mapping[str, list[str]] = OCEAN_FIELD_NAME_PREFIXES,
        cell_area_provider: HasCellAreaInMetersSquared | None = None,
        area_weights_provider: AreaWeightsProviderABC = NullAreaWeightsProvider,
    ):
        """
        Initializes the instance based on the provided data and prefixes.

        Args:
            ocean_data: Mapping from field names to tensors.
            depth_coordinate: The depth coordinate of the model.
            ocean_field_name_prefixes: Mapping which defines the correspondence
                between an arbitrary set of "standard" names (e.g.,
                "potential_temperature" or "salinity") and lists of possible
                names or prefix variants (e.g., ["thetao_"] or
                ["zos"]) found in the data.
            cell_area_provider: An object providing cell areas in square meters
                via the ``area_weights_m2`` property. Used by derived variables
                that need cell area information (e.g. sea ice thickness).
            area_weights_provider: Ocean-area weights (areacello *
                sea_surface_fraction) of the dataset, if it has them.
        """
        self._data = dict(ocean_data)
        self._prefix_map = ocean_field_name_prefixes
        self._depth_coordinate = depth_coordinate
        self._stacker = Stacker(ocean_field_name_prefixes)
        self._cell_area_provider = cell_area_provider
        self._area_weights_provider = area_weights_provider

    @property
    def data(self) -> TensorDict:
        """Mapping from field names to tensors."""
        return self._data

    def __getitem__(self, name: str):
        return getattr(self, name)

    def _get_prefix(self, prefix):
        return self.data[prefix]

    def _set(self, name, value):
        for prefix in self._prefix_map[name]:
            if prefix in self.data.keys():
                self._set_prefix(prefix, value)
                return
        raise KeyError(name)

    def _set_prefix(self, prefix, value):
        self.data[prefix] = value

    def _get(self, name):
        for prefix in self._prefix_map[name]:
            if prefix in self.data.keys():
                return self._get_prefix(prefix)
        raise KeyError(name)

    @property
    def sea_water_potential_temperature(self) -> torch.Tensor:
        """Returns all depth levels of potential temperature."""
        return self._stacker("sea_water_potential_temperature", self.data)

    @property
    def sea_water_salinity(self) -> torch.Tensor:
        """Returns all depth levels of salinity."""
        return self._stacker("sea_water_salinity", self.data)

    @property
    def sea_water_x_velocity(self) -> torch.Tensor:
        """Returns all depth levels of x-velocity."""
        return self._stacker("sea_water_x_velocity", self.data)

    @property
    def sea_water_y_velocity(self) -> torch.Tensor:
        """Returns all depth levels of y-velocity."""
        return self._stacker("sea_water_y_velocity", self.data)

    @property
    def sea_surface_temperature(self) -> torch.Tensor:
        """Returns the sea surface temperature in K.

        Falls back to the top potential-temperature layer (degC) + 273.15 when
        no "sst" field is present, e.g. when sst is an output-only diagnostic
        of the ocean model and the step input carries only the prognostic
        layers. The training stores define sst this way (exactly on UFS and
        GLORYS, to 0.03 K rms on CM4).
        """
        try:
            return self._get("sea_surface_temperature")
        except KeyError:
            top_layer = self._prefix_map["sea_water_potential_temperature"][0] + "0"
            if top_layer in self.data:
                return self.data[top_layer] + FREEZING_TEMPERATURE_KELVIN
            raise

    @property
    def sea_surface_height_above_geoid(self) -> torch.Tensor:
        """Returns sea surface height above geoid."""
        return self._get("sea_surface_height_above_geoid")

    @property
    def sea_surface_height(self) -> torch.Tensor:
        """Returns the sea surface height (SSH), including its global mean."""
        return self._get("sea_surface_height")

    @property
    def ocean_heat_content(self) -> torch.Tensor:
        """Returns column-integrated ocean heat content."""
        if self._depth_coordinate is None:
            raise ValueError(
                "Depth coordinate must be provided to compute column-integrated "
                "ocean heat content."
            )
        return self._depth_coordinate.depth_integral(
            self.sea_water_potential_temperature
            * SPECIFIC_HEAT_OF_SEA_WATER_CM4
            * DENSITY_OF_SEA_WATER_CM4
        )

    @property
    def ocean_salt_content(self) -> torch.Tensor:
        """Returns column-integrated ocean salt content in g/m2, per unit
        total cell area, weighted by the sea surface fraction.
        """
        if self._depth_coordinate is None:
            raise ValueError(
                "Depth coordinate must be provided to compute column-integrated "
                "ocean salt content."
            )
        return (
            self._depth_coordinate.depth_integral(
                self.sea_water_salinity * DENSITY_OF_SEA_WATER_CM4
            )
            * self.sea_surface_fraction
        )

    def _layer_geometry(self, label: str) -> HasOceanLayerGeometry:
        coord = self._depth_coordinate
        # the depth coordinate is typed by depth_integral alone; the wright97
        # variables also need its layer geometry
        if not isinstance(coord, HasOceanLayerGeometry):
            raise ValueError(
                "A depth coordinate with idepth, mask, dz and deptho must be "
                f"provided to compute {label}, got {type(coord).__name__}."
            )
        return coord

    def _wright97_profiles(
        self, label: str
    ) -> tuple[HasOceanLayerGeometry, torch.Tensor, torch.Tensor]:
        """``(geometry, thetao, so)``, the profiles with the depth coordinate's
        ``nz`` levels. Missing profiles raise before the depth coordinate is
        checked, so ``compute_ocean_derived_quantities`` skips ``label``.

        Raises:
            KeyError: If either is missing, or a level ``k < nz`` of either.
            ValueError: If the depth coordinate is not a
                ``HasOceanLayerGeometry``, or either has more than ``nz`` levels.
        """
        thetao = self.sea_water_potential_temperature
        so = self.sea_water_salinity
        geometry = self._layer_geometry(label)
        nz = geometry.mask.shape[-1]
        for name, x in (("thetao", thetao), ("so", so)):
            if x.shape[-1] < nz:
                raise KeyError(f"{name}_{x.shape[-1]}")
            if x.shape[-1] > nz:
                raise ValueError(
                    f"{name} has {x.shape[-1]} levels, more than the "
                    f"{nz} of the depth coordinate."
                )
        return geometry, thetao, so

    @property
    def mld_wright97(self) -> torch.Tensor:
        """Density-threshold mixed layer depth [m], positive down.

        ``mixed_layer_depth`` of ``fme.core.ocean_eos`` with
        ``DELTA_RHO_THRESHOLD`` and ``MLD_REF_LAYER``: the depth where the
        Wright (1997) zero-pressure density first exceeds that of the reference
        layer by the threshold, else the sea floor depth (the depth
        coordinate's ``deptho``, or the deepest unmasked interface). The
        reference is level 0, so the MLD lies below that level's centre. NaN
        where ``mask_0 == 0``, as ``DepthCoordinate.depth_integral``, and where
        an input is non-finite at a level with ``mask_k > 0``.

        Raises:
            ValueError: If the depth coordinate is not a
                ``HasOceanLayerGeometry``, or potential temperature or salinity
                has more levels than it.
            KeyError: If potential temperature or salinity is missing from the
                data, or it has fewer than ``MLD_REF_LAYER + 2`` levels (no
                level below the reference layer), so
                ``compute_ocean_derived_quantities`` skips it.
        """
        geometry, thetao, so = self._wright97_profiles("mld_wright97")
        if thetao.shape[-1] < MLD_REF_LAYER + 2:
            # a level below the reference layer is missing, as a Stacker miss
            raise KeyError(
                f"mld_wright97 needs at least {MLD_REF_LAYER + 2} levels, "
                f"got {thetao.shape[-1]}."
            )
        idepth = geometry.idepth.to(thetao.dtype)
        mask = geometry.mask
        deptho = sea_floor_depth(idepth, mask, geometry.deptho)
        mld = mixed_layer_depth(
            thetao, so, idepth, mask, deptho, DELTA_RHO_THRESHOLD, MLD_REF_LAYER
        )
        mask_0 = mask.select(dim=-1, index=0).expand(mld.shape)
        return mld.where(mask_0 > 0, float("nan"))

    @property
    def rho_wright97(self) -> torch.Tensor:
        """Wright (1997) in-situ density anomaly ``rho_k - RHO_0`` [kg m-3] at
        the Boussinesq pressure of each level centre, ``(..., nz)`` with level
        ``k`` on the last dim. NaN where ``mask_k == 0`` or an input is NaN.

        A partial bottom cell takes the pressure of the full cell's centre, not
        of its wet part's. Inputs are not clamped to the fit range
        (``fme.core.ocean_eos``); outside it the fit extrapolates.

        Raises:
            ValueError: If the depth coordinate is not a
                ``HasOceanLayerGeometry``, or potential temperature or salinity
                has more levels than it.
            KeyError: If a level of potential temperature or salinity is
                missing.
        """
        geometry, thetao, so = self._wright97_profiles("rho_wright97")
        return density_anomaly(thetao, so, geometry.idepth, geometry.mask)

    @property
    def steric_sea_level_wright97(self) -> torch.Tensor:
        """Steric sea level [m] of the static column, Wright (1997).

            C          = sum_k rho_wright97_k * dz_k             [kg m-2]
            eta_steric = -C / RHO_0                              [m]

        k: depth level; dz_k: the depth coordinate's ``dz``. NaN where
        ``mask_0 == 0``; a NaN ``rho_wright97_k`` contributes 0 to C.

        Relative to a column of density ``RHO_0`` over the static levels (sea
        surface at 0), as ``manometric_sea_level_wright97``, with which
        ``zos - <zos> = eta_steric - <eta_steric> + eta_mano`` (<x> as there).
        Not demeaned, and needs neither ``zos`` nor a cell area provider.

        The static compressibility part of C scales with ``deptho`` and
        dominates the field; gen - target differences and time anomalies are
        the meaningful metrics, not the field itself.

        Raises:
            KeyError: If potential temperature or salinity is missing, or a
                level of either; raised before the depth coordinate is checked,
                so ``compute_ocean_derived_quantities`` skips it.
            ValueError: If the depth coordinate is not a
                ``HasOceanLayerGeometry``, or potential temperature or salinity
                has more levels than it.
        """
        geometry, thetao, so = self._wright97_profiles("steric_sea_level_wright97")
        rho = density_anomaly(thetao, so, geometry.idepth, geometry.mask)
        C = column_density_integral(rho, geometry.dz)
        mask_0 = geometry.mask.select(dim=-1, index=0).to(C.device)
        return torch.where(mask_0 > 0, -C / RHO_0, torch.nan)

    @property
    def manometric_sea_level_wright97(self) -> torch.Tensor:
        """Globally demeaned manometric sea level [m], Wright (1997).

            C        = sum_k rho_wright97_k * dz_k                [kg m-2]
            eta_mano = zos + C / RHO_0 - <zos + C / RHO_0>        [m]

        k: depth level; dz_k: the depth coordinate's ``dz``; <x>: mean weighted
        by ``area_weights_m2`` over the cells where ``mask_0 > 0`` and ``zos``
        is finite; NaN elsewhere. A NaN ``rho_wright97_k`` at a level with
        ``mask_k > 0`` (NaN T or S) counts as 0 in C, as NaN integrands in
        ``DepthCoordinate.depth_integral``; ``mld_wright97`` is NaN there.

        With the steric sea level ``eta_steric = -C / RHO_0``,
        ``zos = eta_steric + eta_mano`` up to <x> (Gregory et al. 2019, Surv.
        Geophys.). Both integrate over the static column (sea surface at 0),
        so ``eta_mano`` differs from bottom pressure / (RHO_0 * G_EARTH) minus
        ``deptho`` by ``rho_wright97_0 * zos / RHO_0`` before the mean is
        removed. A spatially uniform offset in ``zos`` cancels in the mean, so
        ``zos`` and ``SSH`` give the same field.

        The static compressibility part of C scales with ``deptho`` and
        dominates the field; gen - target differences and time anomalies are
        the meaningful metrics, not the field itself.

        Raises:
            ValueError: If the depth coordinate is not a
                ``HasOceanLayerGeometry``, or potential temperature or salinity
                has more levels than it.
            KeyError: If a level of potential temperature or salinity, ``zos``,
                the cell area provider or its ``area_weights_m2`` (``None`` on
                HEALPix) is missing.
        """
        area = (
            None
            if self._cell_area_provider is None
            else self._cell_area_provider.area_weights_m2
        )
        if area is None:
            raise KeyError(
                "area_weights_m2, needed for the mean of manometric_sea_level_wright97"
            )
        zos = self.sea_surface_height_above_geoid
        geometry, thetao, so = self._wright97_profiles("manometric_sea_level_wright97")
        rho = density_anomaly(thetao, so, geometry.idepth, geometry.mask)
        C = column_density_integral(rho, geometry.dz)
        mask_0 = geometry.mask.select(dim=-1, index=0).to(C.device)
        wet = (mask_0 > 0) & zos.isfinite()
        eta = torch.where(wet, zos, 0.0) + C / RHO_0
        wet = wet.expand(eta.shape)
        area = area.to(device=eta.device, dtype=eta.dtype)
        mean = Distributed.get_instance().weighted_mean(
            eta, wet.to(eta.dtype) * area, dim=(-2, -1), keepdim=True
        )
        return torch.where(wet, eta - mean, torch.nan)

    @property
    def water_flux_into_sea_water(self) -> torch.Tensor:
        """Returns water flux into sea water in kg/m2/s."""
        return self._get("water_flux_into_sea_water")

    @property
    def sea_surface_fraction(self) -> torch.Tensor:
        """Returns the sea surface fraction."""
        try:
            return self._get("sea_surface_fraction")
        except KeyError:
            return 1 - self.land_fraction

    @property
    def net_downward_surface_heat_flux(self) -> torch.Tensor:
        """Net heat flux downward across the ocean surface (below the sea-ice)."""
        try:
            return self._get("net_downward_surface_heat_flux")
        except KeyError:
            # derive from the sea-surface-fraction-weighted version
            return (
                self.net_downward_surface_heat_flux_total_area
                / self.sea_surface_fraction
            )

    @property
    def net_downward_surface_heat_flux_total_area(self) -> torch.Tensor:
        """Net heat flux downward across the ocean surface (below the sea-ice),
        normalized by total grid cell area.
        """
        return self._get("net_downward_surface_heat_flux_total_area")

    @property
    def geothermal_heat_flux(self) -> torch.Tensor:
        """Geothermal heat flux."""
        try:
            return self._get("geothermal_heat_flux")
        except KeyError:
            return torch.zeros_like(self.sea_surface_fraction)

    @property
    def net_energy_flux_into_ocean(self) -> torch.Tensor:
        return (
            self.net_downward_surface_heat_flux + self.geothermal_heat_flux
        ) * self.sea_surface_fraction

    @property
    def downward_sea_ice_basal_salt_flux(self) -> torch.Tensor:
        """Returns the salt flux from sea ice into the ocean in kg/m2/s, with
        NaN as zero.
        """
        return torch.nan_to_num(self._get("downward_sea_ice_basal_salt_flux"), nan=0.0)

    @property
    def net_virtual_salt_flux_into_ocean(self) -> torch.Tensor:
        """Virtual salt flux into the ocean column in g/m2/s, per unit total
        cell area: the water flux into sea water times a fixed reference
        salinity, weighted by the sea surface fraction.
        """
        return (
            -REFERENCE_SALINITY
            * self.water_flux_into_sea_water
            * self.sea_surface_fraction
        )

    @property
    def net_salt_flux_into_ocean(self) -> torch.Tensor:
        """Net salt flux into the ocean column in g/m2/s, per unit total cell
        area: the virtual salt flux plus the salt flux from sea ice, weighted
        by the sea surface fraction.
        """
        sea_ice_salt_flux = 1000 * self.downward_sea_ice_basal_salt_flux  # kg -> g
        return (
            self.net_virtual_salt_flux_into_ocean
            + sea_ice_salt_flux * self.sea_surface_fraction
        )

    @property
    def sea_ice_fraction(self) -> torch.Tensor:
        """Returns the sea ice fraction."""
        try:
            return self._get("sea_ice_fraction")
        except KeyError:
            land_fraction = self.land_fraction
            ocean_sea_ice_fraction = self.ocean_sea_ice_fraction
            return ocean_sea_ice_fraction * (1 - land_fraction)

    @property
    def land_fraction(self) -> torch.Tensor:
        """Returns the land fraction."""
        return self._get("land_fraction")

    @property
    def ocean_sea_ice_fraction(self) -> torch.Tensor:
        """Returns the sea ice fraction as a proportion of the sea surface."""
        return self._get("ocean_sea_ice_fraction")

    @property
    def ocean_fraction(self) -> torch.Tensor:
        """Returns the dynamic ocean fraction, computed from the sea ice
        fraction and land fraction.
        """
        return 1 - self.land_fraction - self.sea_ice_fraction

    @property
    def area_weights_m2(self) -> torch.Tensor:
        """Returns cell areas in square meters.

        Raises:
            ValueError: If a cell area provider was not provided.
        """
        if self._cell_area_provider is None:
            raise ValueError(
                "A cell area provider must be provided to access cell area information."
            )
        return self._cell_area_provider.area_weights_m2

    @property
    def ocean_area_weights(self) -> torch.Tensor | None:
        """2-D (lat, lon) ocean-area weights (areacello * sea_surface_fraction,
        normalized), or None if the dataset has no area weights provider.
        """
        return self._area_weights_provider.area_weights

    @property
    def sea_ice_thickness(self) -> torch.Tensor:
        """Returns the sea ice thickness in m, per unit sea ice area.

        Lookup order: stored ``HI``; else ``ocean_sea_ice_thickness_mean``
        divided by ``ocean_sea_ice_fraction`` (both per unit ocean area), 0
        where that fraction is not positive; else derived from
        ``sea_ice_volume``. NaN where an input is NaN (land).
        """
        try:
            return self._get("sea_ice_thickness")
        except KeyError:
            pass
        try:
            thickness_mean = self.ocean_sea_ice_thickness_mean
            ocean_sea_ice_frac = self.ocean_sea_ice_fraction
        except KeyError:
            return self._sea_ice_thickness_from_volume()
        is_nan = torch.isnan(thickness_mean) | torch.isnan(ocean_sea_ice_frac)
        has_ice = (ocean_sea_ice_frac > 0) & ~is_nan
        # safe operands keep the unselected branch, and its gradient, finite
        safe_numerator = torch.where(has_ice, thickness_mean, 0.0)
        safe_denominator = torch.where(has_ice, ocean_sea_ice_frac, 1.0)
        thickness = torch.where(has_ice, safe_numerator / safe_denominator, 0.0)
        return torch.where(is_nan, float("nan"), thickness)

    def _sea_ice_thickness_from_volume(self) -> torch.Tensor:
        sfrac = self.sea_surface_fraction
        sea_ice_vol = self.sea_ice_volume
        try:
            sea_ice_frac = self.ocean_sea_ice_fraction * sfrac
        except KeyError:
            # assumes that sea_ice_fraction comes from compute_coupled_sea_ice
            # in scripts/data_process/coupled_dataset_utils.py
            lfrac = self.land_fraction
            sea_ice_frac = self.sea_ice_fraction * sfrac / (1 - lfrac)
        cell_area = self.area_weights_m2
        return torch.where(
            torch.isnan(sea_ice_vol),
            float("nan"),
            torch.nan_to_num(
                torch.exp(
                    torch.log(sea_ice_vol)
                    - torch.log(cell_area)
                    - torch.log(sea_ice_frac)
                )
            ),
        )

    @property
    def sea_ice_volume(self) -> torch.Tensor:
        """Returns the sea ice volume."""
        return self._get("sea_ice_volume")

    @property
    def ocean_sea_ice_thickness_mean(self) -> torch.Tensor:
        """Returns sea ice thickness times sea ice fraction, averaged over the
        ocean area of the cell (ice volume per unit ocean area), in m.
        """
        return self._get("ocean_sea_ice_thickness_mean")
