import dataclasses
import logging
from typing import Literal

import xarray as xr
from create_window_avg_dataset import WindowAvgDatasetConfig


@dataclasses.dataclass
class ExtraFieldsConfig:
    """Additional fields to include in output."""

    names_and_prefixes: list[str] | None = None

    def copy_extra_data_vars(
        self, input_ds: xr.Dataset, output_ds: xr.Dataset
    ) -> xr.Dataset:
        """Copy additional data variables from input to output dataset. Mutates
        output_ds.

        Matches variables by exact name or prefix (if ending with '_').

        Args:
            input_ds: Source dataset containing variables to copy.
            output_ds: Target dataset to copy variables into.

        Returns:
            The output dataset with additional variables copied in.

        """
        if self.names_and_prefixes is None:
            return output_ds

        def match(prefix_or_name: str, name: str):
            if prefix_or_name.endswith("_"):
                return name.startswith(prefix_or_name)
            return name == prefix_or_name

        for prefix_or_name in self.names_and_prefixes:
            for name in input_ds.data_vars:
                if match(prefix_or_name, name):
                    output_ds[name] = input_ds[name]
        return output_ds

    def drop_extra_data_vars(self, ds: xr.Dataset) -> xr.Dataset:
        """Drop the additional data variables from a dataset where they were
        previously added.

        Returns:
            A copy of the dataset, with any additional variables dropped.
        """
        if self.names_and_prefixes is not None:
            names = [
                var
                for var in ds.data_vars
                if any(var.startswith(prefix) for prefix in self.names_and_prefixes)
            ]
        else:
            names = []
        return ds.drop_vars(names, errors="ignore")


def _clip_coastal_grid_cells(
    thresh: float,
    ts: xr.DataArray,
    ofrac: xr.DataArray,
    min: xr.DataArray,
    max: xr.DataArray,
) -> xr.DataArray:
    """Clip temperature values in coastal grid cells to physical bounds.

    Args:
        thresh: Ocean fraction threshold defining coastal cells.
        ts: Surface temperature field.
        ofrac: Ocean fraction field.
        min: Minimum allowable temperature.
        max: Maximum allowable temperature.

    Returns:
        Clipped temperature field for coastal regions.
    """
    return ts.where(ofrac > 0).where(ofrac < thresh).clip(min=min, max=max)


def _minmax_coastal_solid_temp(
    ts: xr.DataArray, sst: xr.DataArray, ofrac: xr.DataArray, cutoff: float = 0.4
) -> tuple[xr.DataArray, xr.DataArray]:
    """Returns the time-dependent min/max map for solid temperature.

    Calculates physically reasonable bounds for land/ice temperature in coastal
    grid cells based on the observed relationship between total surface temperature
    and sea surface temperature.

    Args:
        ts: Skin surface temperature field (surface-type weighted).
        sst: Sea surface temperature field.
        ofrac: Ocean fraction.
        cutoff: Ocean fraction threshold below which solid fraction is computed
            directly from ofrac.

    Returns:
        Tuple of (min_solid_ts, max_solid_ts): Time-dependent minimum and
        maximum solid surface temperature fields.

    """
    # compute min and max difference ts - sst on coastal grid cells
    coastal_ts_sst_diff = (ts - sst).where(ofrac > 0.0).where(ofrac < 1)
    alpha = coastal_ts_sst_diff.min(dim="time").load()
    beta = coastal_ts_sst_diff.max(dim="time").load()
    solid_frac = (1 - ofrac).where(ofrac < cutoff, 1 - cutoff)
    min_solid_ts = sst + alpha / solid_frac
    max_solid_ts = sst + beta / solid_frac
    return min_solid_ts, max_solid_ts


def _interpolate_sst(
    ts: xr.DataArray,
    sst: xr.DataArray,
    ofrac: xr.DataArray,
    thresh: float = 1.0,
) -> xr.DataArray:
    ofrac = ofrac.where(ofrac < thresh, 1.0)
    return (1 - ofrac) * ts + ofrac * sst


def _make_serializable_time_coord(
    ds: xr.Dataset,
    timedelta: str = "6h",
    tdim: str = "time",
) -> xr.CFTimeIndex:
    # this is needed for proper time coord serialization
    tcoord = xr.date_range(
        str(ds[tdim].isel({tdim: 0}).values.item()),
        str(ds[tdim].isel({tdim: -1}).values.item()),
        freq=timedelta,
        calendar=ds[tdim].isel({tdim: 0}).values.item().calendar,
        use_cftime=True,
    )
    assert len(tcoord) == len(ds[tdim])
    tcoord.attrs = ds[tdim].attrs
    return tcoord


@dataclasses.dataclass
class CoupledSurfaceTemperatureConfig:
    """Determines how to apply the input ocean dataset's SST to the input
    atmosphere dataset's surface temperature field in order produce a target
    that is similar to the generated surface temperature field in coupled
    emulation.

    There are three options for 'how':

        "solid_ts": Where ocean fraction > the threshold, the output surface
            temperature is equal to sst. Where ofrac < the threshold, output
            surface temperature becomes (ts - ofrac*sst) / (lfrac + sic).
        "interpolate_sst": Where ocean fraction > the threshold, the output
            surface temperature is equal to sst. Otherwise, it is equal to
            ofrac * sst + (1 - ofrac) * ts.
        "threshold": Where ocean fraction > the threshold, the output surface
            temperature is equal to sst. Otherwise, it is equal to ts.

    Parameters:
        how: How to compute the coupled surface temperature.
        ocean_fraction_threshold: Threshold above which ocean fractions values are
            treated as 1.
        timedelta: Time resolution of the atmosphere data (default: "6h").

    """

    how: Literal["solid_ts", "interpolate_sst", "threshold"]
    ocean_fraction_threshold: float = 1.0
    timedelta: str = "6h"

    def apply_sst_to_ts(
        self,
        ts: xr.DataArray,
        sst: xr.DataArray,
        ofrac: xr.DataArray,
    ) -> xr.DataArray:
        """Apply ocean SST to atmosphere surface temperature field.

        Creates a coupled surface temperature field appropriate for training
        coupled emulators by blending SST into the atmosphere's surface temperature.

        Args:
            ts: Skin surface temperature field (surface-type weighted).
            sst: Sea surface temperature.
            ofrac: Ocean fraction.

        Returns:
            Modified surface temperature field with SST applied according to the
            configured method ('solid_ts', 'interpolate_sst', or 'threshold').
        """
        thresh = self.ocean_fraction_threshold
        if self.how == "solid_ts":
            solid_frac = (1 - ofrac).where(ofrac < thresh)
            solid_ts = (ts - ofrac * sst.fillna(0)) / solid_frac
            solid_ts = solid_ts.fillna(sst)
            # get an estimate of the min/max of ts - sst
            min_series, max_series = _minmax_coastal_solid_temp(solid_ts, sst, ofrac)
            # compute coastal combined temp over land & sea ice
            ts_mod = _clip_coastal_grid_cells(
                thresh, solid_ts, ofrac, min_series, max_series
            )
        elif self.how == "interpolate_sst":
            ts_mod = _interpolate_sst(ts, sst, ofrac, thresh)
        else:
            # simple threshold
            ts_mod = ts.where(ofrac < thresh, sst)
        return ts_mod.fillna(ts)


@dataclasses.dataclass
class PrecomputedSeaIceMaskConfig:
    zarr_path: str
    name: str = "mask_sea_ice_fraction"

    def get_sea_ice_mask(self) -> xr.DataArray:
        logging.info(f"Loading {self.name} from {self.zarr_path}.")
        mask = xr.open_zarr(self.zarr_path)[self.name]
        if len(mask.dims) != 2:
            raise ValueError(
                f"Expected a 2D precomputed sea ice mask, but got {str(mask)}."
            )
        return mask.load()


@dataclasses.dataclass
class CoupledSeaSurfaceConfig:
    """Configuration of surface fields for the coupled ocean dataset, including
    sea ice masking and time-averaged fluxes.

    Parameters:
        surface_flux_window_avg: Configuration for computing windowed time-averaged
            surface flux variables from the atmosphere dataset.
        sst_threshold: Upper threshold for the SST time mean in degrees Kelvin.
            The resulting sea ice mask is 1 where the time mean SST < the
            threshold and 0 otherwise.
        ocean_extra_masked_names: Additional sea-ice-related variable names for which
            copies of the mask will be added as "mask_{extra_name}". NOTE: The
            sea-ice fraction and "ocean" sea-ice fraction masks don't need to be
            specified here.
        sea_ice_window_avg: Optional configuration for windowed time-averaging of the
            coupled sea ice dataset. If None, the coupled sea ice is simply subsampled
            to the ocean dataset temporal frequency.
        timedelta: Time resolution of the ocean data (default: "120h").

    """

    surface_flux_window_avg: WindowAvgDatasetConfig
    sst_threshold: float | None = None
    ocean_extra_masked_names: list[str] = dataclasses.field(default_factory=list)
    precomputed_sea_ice_mask: PrecomputedSeaIceMaskConfig | None = None
    sea_ice_window_avg: WindowAvgDatasetConfig | None = None
    timedelta: str = "120h"
    _mask: xr.DataArray | None = None

    def __post_init__(self):
        if self.sst_threshold is None and self.precomputed_sea_ice_mask is None:
            raise ValueError(
                "Either sst_threshold or precomputed_sea_ice_mask must be configured."
            )

    def compute_sea_ice_mask(self, sst: xr.DataArray) -> xr.DataArray:
        """Compute sea ice mask based on time-mean SST threshold.

        Args:
            sst: Sea surface temperature field.

        Returns:
            Binary mask (1 where sea ice is present, 0 elsewhere).
        """
        if self.precomputed_sea_ice_mask is not None:
            if self.sst_threshold is not None:
                logging.warning(
                    f"sst_threshold={self.sst_threshold} configured ",
                    "but precomputed_sea_ice_mask is also non-null. Ignoring "
                    "sst_threshold and loading the precomputed mask.",
                )
            self._mask = self.precomputed_sea_ice_mask.get_sea_ice_mask()
        else:
            logging.info(
                "Computing sea ice mask from the time-mean SST map using "
                f"sst_threshold={self.sst_threshold}."
            )
            sst_tm = sst.mean(dim="time")
            self._mask = (sst_tm < self.sst_threshold).fillna(0.0)
        return self._mask

    def apply_mask(self, da: xr.DataArray) -> xr.DataArray:
        """Apply the computed sea ice mask to a data array.

        Args:
            da: Data array to mask.

        Returns:
            Masked data array (NaN where mask is 0).

        Raises:
            RuntimeError: If compute_sea_ice_mask has not been called first.
        """
        if self._mask is None:
            raise RuntimeError("Call compute_sea_ice_mask before apply_mask.")
        return da.where(self._mask > 0)

    def apply_sea_ice_window_avg(self, ds: xr.Dataset) -> xr.Dataset:
        if self.sea_ice_window_avg is not None:
            logging.info("Computing window-averaged sea ice fields")
            return self.sea_ice_window_avg.get_window_avg(ds)
        return ds

    def apply_surface_flux_window_avg(self, ds: xr.Dataset) -> xr.Dataset:
        logging.info("Computing window-averaged surface fluxes")
        return self.surface_flux_window_avg.get_window_avg(ds)


@dataclasses.dataclass
class AtmosphereInputFieldsConfig:
    """Names of fields from atmosphere input dataset.

    Attributes:
        surface_temperature_name: Name of surface temperature variable.
        sea_ice_fraction_name: Name of sea ice fraction variable.
        land_fraction_name: Name of land fraction variable.
        ocean_fraction_name: Name of ocean fraction variable.
        sea_surface_fraction_name: Name of sea surface fraction variable.
    """

    surface_temperature_name: str = "surface_temperature"
    sea_ice_fraction_name: str = "sea_ice_fraction"
    land_fraction_name: str = "land_fraction"
    ocean_fraction_name: str = "ocean_fraction"
    sea_surface_fraction_name: str = "sea_surface_fraction"


@dataclasses.dataclass
class OceanInputFieldsConfig:
    """Names of fields from ocean input dataset.

    Attributes:
        sea_surface_fraction_name: Name of sea surface fraction variable.
        sea_surface_temperature_name: Name of sea surface temperature variable.
    """

    sea_surface_fraction_name: str = "sea_surface_fraction"
    sea_surface_temperature_name: str = "sst"


@dataclasses.dataclass
class DerivedFieldsConfig:
    """Names of coupled fields that will be derived from existing inputs.

    Attributes:
        ocean_sea_ice_fraction_name: Name of the variable for sea ice concentration
            as a fraction the ocean area in a given grid cell.
    """

    ocean_sea_ice_fraction_name: str = "ocean_sea_ice_fraction"


@dataclasses.dataclass
class CoupledFieldNamesConfig:
    """Comprehensive configuration for all field and dimension names.

    Attributes:
        time_dim: Name of time dimension.
        latitude_dim: Name of latitude dimension.
        longitude_dim: Name of longitude dimension.
        atmosphere: Configuration for atmosphere field names.
        ocean: Configuration for ocean field names.
        derived: Configuration for derived field names.
    """

    time_dim: str = "time"
    latitude_dim: str = "lat"
    longitude_dim: str = "lon"
    atmosphere: AtmosphereInputFieldsConfig = dataclasses.field(
        default_factory=AtmosphereInputFieldsConfig
    )
    ocean: OceanInputFieldsConfig = dataclasses.field(
        default_factory=OceanInputFieldsConfig
    )
    derived: DerivedFieldsConfig = dataclasses.field(
        default_factory=DerivedFieldsConfig
    )


@dataclasses.dataclass
class CoupledSeaIceConfig:
    """Configuration for computing the coupled sea ice dataset.

    Parameters:
        window_avg: Optional configuration for windowed time-averaging of the
            coupled sea ice dataset.
        include_ts: If true, then the atmosphere surface temperature field is added
            to the output dataset. If window_avg is also configured, then the window-
            average is applied to ocean grid cells only based on the forumula
            (1 - ofrac_window_avg) * ts + ofrac_window_avg * ts_window_avg.
        timedelta: Time resolution of the sea ice data (default: "6h").

    """

    window_avg: WindowAvgDatasetConfig | None = None
    include_ts: bool = False
    timedelta: str = "6h"

    def apply_window_avg_and_reindex(
        self,
        ds: xr.Dataset,
        atmos_names: AtmosphereInputFieldsConfig,
        time_dim="time",
    ) -> xr.Dataset:
        ds_avg = ds
        ts_name = atmos_names.surface_temperature_name
        if ts_name in ds_avg and not self.include_ts:
            ds_avg = ds_avg.drop(ts_name)
        if self.window_avg is not None:
            logging.info("Computing window-averaged coupled sea ice")
            ds_avg = self.window_avg.get_window_avg(ds)
            ds_avg = ds_avg.reindex(time=ds["time"], method="ffill")
            logging.info(
                f"After reindex window-averaged time coord:\n {str(ds_avg[time_dim])}"
            )
            if ts_name in ds_avg.data_vars:
                ds_avg[ts_name] = _interpolate_sst(
                    ts=ds[ts_name],
                    sst=ds_avg[ts_name],
                    ofrac=ds_avg[atmos_names.ocean_fraction_name],
                )
        return ds_avg


def compute_coupled_sea_ice(
    atmos: xr.Dataset,
    config: CoupledSeaIceConfig,
    sea_ice: xr.Dataset | None = None,
    ocean: xr.Dataset | None = None,
    input_field_names: CoupledFieldNamesConfig | None = None,
    atmos_extras: ExtraFieldsConfig | None = None,
    sea_ice_extras: ExtraFieldsConfig | None = None,
) -> xr.Dataset:
    """Compute coupled sea ice fields from ocean and atmosphere datasets.

    Derives consistent sea ice concentration, fractions, and masks by combining
    information from the processed atmosphere and ocean datasets.

    Args:
        atmos: Atmosphere dataset containing land fraction, sea ice fraction, and
            ocean fraction.
        sea_ice: Optional separate sea ice dataset. If provided, sea ice fraction
            is taken from here instead of from atmosphere dataset. Assumed to have
            the same temporal resolution as the atmos dataset.
        ocean: Optional separate dataset containing sea surface fraction. Unused if
            sea_ice is non-null and contains sea surface fraction.
        config: Configuration for window averaging and including surface temperature.
        input_field_names: Names of input fields. If None, uses defaults.
        atmos_extras: Optional configuration for copying extra variables from the
            atmosphere dataset.
        sea_ice_extras: Optional configuration for copying extra variables from the
            sea ice dataset. If given, sea_ice must be non-None.

    Returns:
        Dataset with the same temporal resolution as the atmos input dataset,
        containing coupled sea ice fields including:
            - Modified land, ocean, and sea ice fractions
            - Sea ice concentration
            - Sea surface fraction
            - Sea ice mask(s) for masking variables
            - Additional masked sea ice variables as configured
    """
    if sea_ice is None and sea_ice_extras is not None:
        raise ValueError("Got non-None sea_ice_extras but sea_ice is None.")

    if input_field_names is None:
        input_field_names = CoupledFieldNamesConfig()

    tdim = input_field_names.time_dim
    latdim = input_field_names.latitude_dim
    londim = input_field_names.longitude_dim

    # atmosphere inputs
    lfrac_name = input_field_names.atmosphere.land_fraction_name
    ifrac_name = input_field_names.atmosphere.sea_ice_fraction_name
    ofrac_name = input_field_names.atmosphere.ocean_fraction_name
    ts_name = input_field_names.atmosphere.surface_temperature_name

    # ocean inputs
    sfrac_name = input_field_names.ocean.sea_surface_fraction_name

    # new field
    sic_name = input_field_names.derived.ocean_sea_ice_fraction_name

    lfrac = atmos[lfrac_name].clip(min=0.0, max=1.0)

    sfrac = 1 - lfrac
    if sea_ice is not None and sfrac_name in sea_ice:
        sfrac = sea_ice[sfrac_name]
    elif ocean is not None and sfrac_name in ocean:
        sfrac = ocean[sfrac_name]
    else:
        logging.warning(
            f"{sfrac_name} not found. "
            "Assuming sea surface fraction is 1 - land fraction."
        )
    sfrac = (
        sfrac.fillna(0.0)
        .clip(min=0.0, max=1.0)
        .assign_coords({latdim: atmos.lat, londim: atmos.lon})
    )
    sfrac.attrs = {
        "long_name": "sea surface fraction",
        "units": "unitless",
    }

    ifrac = atmos[ifrac_name].clip(min=0.0, max=1.0)
    if sea_ice is not None:
        ifrac = (
            sea_ice[ifrac_name]
            .fillna(0.0)
            .clip(min=0.0, max=1.0)
            .assign_coords({latdim: atmos[latdim], londim: atmos[londim]})
        )

    sfrac_mod = (1 - lfrac).where(sfrac > 0, 0.0)
    lfrac_mod = 1 - sfrac_mod

    # compute sea ice concentratiion
    sic_mod = (ifrac / sfrac).clip(0, 1).fillna(0.0)
    sic_mod.attrs = {
        "long_name": "sea ice concentration",
        "units": "unitless",
    }

    # compute sea ice fraction and ocean fraction from sic
    ifrac_mod = sic_mod * sfrac_mod
    ofrac_mod = (1 - sic_mod) * sfrac_mod

    lfrac_mod.attrs = lfrac.attrs
    ifrac_mod.attrs = ifrac.attrs
    ofrac_mod.attrs = atmos[ofrac_name].attrs

    ds = xr.Dataset(
        {
            lfrac_name: lfrac_mod,
            sfrac_name: sfrac,
            ofrac_name: ofrac_mod,
            sic_name: sic_mod,
            ifrac_name: ifrac_mod,
            ts_name: atmos[ts_name],
        }
    )
    ds = config.apply_window_avg_and_reindex(
        ds,
        input_field_names.atmosphere,
        input_field_names.time_dim,
    )
    ds[tdim] = _make_serializable_time_coord(
        ds=atmos, tdim=tdim, timedelta=config.timedelta
    )

    # copy over additional variables from inputs
    for extras, input_ds in zip([atmos_extras, sea_ice_extras], [atmos, sea_ice]):
        if extras is not None:
            ds = extras.copy_extra_data_vars(input_ds, ds)

    logging.info(f"Coupled sea ice has variables:\n{ds.data_vars}")
    logging.info(f"Coupled sea ice dataset size is {ds.nbytes / 1e9} GB")
    return ds


def compute_coupled_ocean(
    ocean: xr.Dataset,
    atmos: xr.Dataset,
    coupled_sea_ice: xr.Dataset,
    config: CoupledSeaSurfaceConfig,
    input_field_names: CoupledFieldNamesConfig | None = None,
    extras: ExtraFieldsConfig | None = None,
) -> xr.Dataset:
    """Create coupled ocean dataset at ocean timesteps.

    Selects coupled sea ice fields at ocean time coordinates and optionally
    adds extra variables from the ocean dataset.

    Args:
        ocean: Ocean input dataset.
        atmos: Atmosphere input dataset.
        coupled_sea_ice: Computed coupled sea ice fields from compute_coupled_sea_ice.
        input_field_names: Names of input fields. If None, uses defaults.
        extras: Optional configuration for copying extra variables from ocean dataset.

    Returns:
        Dataset containing coupled fields at ocean timesteps.
    """
    if input_field_names is None:
        input_field_names = CoupledFieldNamesConfig()

    tdim = input_field_names.time_dim
    latdim = input_field_names.latitude_dim
    londim = input_field_names.longitude_dim

    ocean = ocean.assign_coords({latdim: atmos[latdim], londim: atmos[londim]})
    coupled_sea_ice = coupled_sea_ice.assign_coords(
        {latdim: atmos[latdim], londim: atmos[londim]}
    )

    sst_name = input_field_names.ocean.sea_surface_temperature_name
    ifrac_name = input_field_names.atmosphere.sea_ice_fraction_name
    sic_name = input_field_names.derived.ocean_sea_ice_fraction_name
    ts_name = input_field_names.atmosphere.surface_temperature_name

    ds = coupled_sea_ice
    if ts_name in ds.data_vars:
        ds = ds.drop(ts_name)

    ds = config.apply_sea_ice_window_avg(coupled_sea_ice)
    ds = ds.sel({tdim: ocean[tdim]})
    ds = xr.merge([ds, config.apply_surface_flux_window_avg(atmos)])
    ds[tdim] = _make_serializable_time_coord(
        ds=ocean, tdim=tdim, timedelta=config.timedelta
    )

    sea_ice_mask = config.compute_sea_ice_mask(ocean[sst_name])
    sea_ice_mask.attrs = {
        "long_name": "sea ice mask",
        "units": "unitless",
    }

    # apply masking to sea ice fractions
    for name in [ifrac_name, sic_name]:
        ds[name] = config.apply_mask(ds[name])
        ds[f"mask_{name}"] = sea_ice_mask

    # apply masking to other sea ice variables in the ocean dataset
    for name in config.ocean_extra_masked_names:
        ds[name] = config.apply_mask(ocean[name])
        ds[name].attrs = ocean[name].attrs
        ds[f"mask_{name}"] = sea_ice_mask

    if extras is not None:
        # copy over additional variables from the ocean dataset
        return extras.copy_extra_data_vars(ocean, ds)

    logging.info(f"Coupled ocean has variables:\n{ds.data_vars}")
    logging.info(f"Coupled ocean dataset size is {ds.nbytes / 1e9} GB")
    return ds


def compute_coupled_atmosphere(
    atmos: xr.Dataset,
    ocean: xr.Dataset,
    coupled_ocean: xr.Dataset,
    config: CoupledSurfaceTemperatureConfig,
    input_field_names: CoupledFieldNamesConfig | None = None,
    extras: ExtraFieldsConfig | None = None,
) -> xr.Dataset:
    """Create coupled atmosphere dataset at atmosphere timesteps.

    Applies ocean SST to atmosphere surface temperature and combines with
    coupled sea ice fractions to create a consistent coupled atmosphere dataset.

    Args:
        atmos: Atmosphere input dataset.
        ocean: Ocean input dataset containing SST.
        coupled_ocean: Computed coupled sea ice fields from compute_coupled_ocean.
        config: Configuration for surface temperature coupling.
        input_field_names: Names of input fields. If None, uses defaults.
        extras: Optional configuration for copying extra variables from atmosphere.

    Returns:
        Dataset containing coupled atmosphere fields at atmosphere timesteps,
            including modified surface temperature with SST applied.
    """
    if input_field_names is None:
        input_field_names = CoupledFieldNamesConfig()

    tdim = input_field_names.time_dim
    latdim = input_field_names.latitude_dim
    londim = input_field_names.longitude_dim

    ocean = ocean.assign_coords({latdim: atmos[latdim], londim: atmos[londim]})
    coupled_ocean = coupled_ocean.assign_coords(
        {latdim: atmos[latdim], londim: atmos[londim]}
    )

    # atmos names
    ts_name = input_field_names.atmosphere.surface_temperature_name
    lfrac_name = input_field_names.atmosphere.land_fraction_name
    ifrac_name = input_field_names.atmosphere.sea_ice_fraction_name
    ofrac_name = input_field_names.atmosphere.ocean_fraction_name

    # ocean names
    sfrac_name = input_field_names.ocean.sea_surface_fraction_name
    sst_name = input_field_names.ocean.sea_surface_temperature_name

    ts = atmos[ts_name]
    sst = ocean[sst_name]
    lfrac = coupled_ocean[lfrac_name]
    ifrac = coupled_ocean[ifrac_name].sel({tdim: ocean[tdim]})
    ofrac = coupled_ocean[ofrac_name].sel({tdim: ocean[tdim]})
    sfrac = coupled_ocean[sfrac_name]

    ifrac_reindex = ifrac.reindex({tdim: atmos[tdim]}, method="ffill")
    ofrac_reindex = ofrac.reindex({tdim: atmos[tdim]}, method="ffill")

    sst_reindex = sst.reindex({tdim: ts[tdim]}, method="ffill")

    ts_mod = config.apply_sst_to_ts(
        ts,
        sst_reindex,
        ofrac_reindex,
    )
    ts_mod.attrs = ts.attrs

    ds = xr.Dataset(
        {
            lfrac_name: lfrac,
            ofrac_name: ofrac_reindex,
            ifrac_name: ifrac_reindex,
            sfrac_name: sfrac,
            ts_name: ts_mod,
        }
    )
    ds[tdim] = _make_serializable_time_coord(
        ds=atmos, tdim=tdim, timedelta=config.timedelta
    )
    if extras is not None:
        # copy over additional variables from the atmosphere dataset
        return extras.copy_extra_data_vars(atmos, ds)
    logging.info(f"Coupled atmosphere has variables:\n{ds.data_vars}")
    logging.info(f"Coupled atmosphere dataset size is {ds.nbytes / 1e9} GB")
    return ds
