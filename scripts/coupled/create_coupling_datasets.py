import dataclasses
import os
import pdb
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import click
import dacite
import xarray as xr
import xpartition  # noqa
import yaml
from dask.distributed import Client

INNER_CHUNKS = {"time": 1, "lon": -1, "lat": -1}
OUTER_CHUNKS = {"time": 360, "lon": -1, "lat": -1}

EPS = 1e-6


@dataclasses.dataclass
class InputDatasetConfig:
    zarr_path: str
    time_chunk_size: int
    timedelta: str
    extra_names_and_prefixes: list[str] | None

    def get_dataset(self) -> xr.Dataset:
        return xr.open_zarr(self.zarr_path, chunks={"time": self.time_chunk_size})

    def copy_extra_data_vars(
        self, input_ds: xr.Dataset, output_ds: xr.Dataset
    ) -> xr.Dataset:
        if self.extra_names_and_prefixes is None:
            return

        def match(prefix_or_name: str, name: str):
            if prefix_or_name.endswith("_"):
                return name.startswith(prefix_or_name)
            return name == prefix_or_name

        for prefix_or_name in self.extra_names_and_prefixes:
            for name in input_ds.data_vars:
                if match(prefix_or_name, name):
                    output_ds[name] = input_ds[name]
        return output_ds


@dataclasses.dataclass
class OutputWriterConfig:
    n_split: int = 50
    starting_split: int = 0
    n_dask_workers: int | None = None
    _client: Client | None = None

    def start_dask_client(self, debug: bool = False):
        if debug or self.n_dask_workers is None:
            return None
        print(f"Using dask Client(n_workers={self.n_dask_workers})...")
        client = Client(n_workers=self.n_dask_workers)
        print(client.dashboard_link)
        self._client = client

    def close_dask_client(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def write(self, ds: xr.Dataset, output_store: str):
        ds = ds.chunk(OUTER_CHUNKS)
        if self.starting_split == 0:
            if os.path.isdir(output_store):
                raise ValueError(
                    f"Output store {output_store} already exists. "
                    "Use starting_split > 0 to continue writing or "
                    "manually delete the directory to start from 0."
                )
            ds.partition.initialize_store(output_store, inner_chunks=INNER_CHUNKS)
        else:
            if not os.path.isdir(output_store):
                raise ValueError(
                    f"starting_split > 0 but output store {output_store} "
                    "hasn't yet been initialized. Use starting_split = 0?"
                )
        for i in range(self.starting_split, self.n_split):
            print(f"Writing segment {i + 1} / {self.n_split}")
            ds.partition.write(
                output_store,
                self.n_split,
                ["time"],
                i,
                collect_variable_writes=True,
            )


@dataclasses.dataclass
class AtmosphereInputFieldsConfig:
    surface_temperature_name: str = "surface_temperature"
    sea_ice_fraction_name: str = "sea_ice_fraction"
    land_fraction_name: str = "land_fraction"
    ocean_fraction_name: str = "ocean_fraction"
    sea_surface_fraction_name: str = "sea_surface_fraction"


@dataclasses.dataclass
class OceanInputFieldsConfig:
    sea_surface_fraction_name: str = "sea_surface_fraction"
    sea_surface_temperature_name: str = "sst"


@dataclasses.dataclass
class DerivedFieldsConfig:
    ocean_sea_ice_fraction_name: str = "ocean_sea_ice_fraction"


def _clip_coastal_grid_cells(
    thresh: float,
    ts: xr.DataArray,
    ofrac: xr.DataArray,
    min: xr.DataArray,
    max: xr.DataArray,
) -> xr.DataArray:
    return ts.where(ofrac > 0).where(ofrac < thresh).clip(min=min, max=max)


def _minmax_coastal_solid_temp(
    ts: xr.DataArray, sst: xr.DataArray, ofrac: xr.DataArray, cutoff: float = 0.4
) -> tuple[xr.DataArray, xr.DataArray]:
    """Returns the time-dependent min/max map for solid temperature."""
    # compute min and max difference ts - sst on coastal grid cells
    coastal_ts_sst_diff = (ts - sst).where(ofrac > 0.0).where(ofrac < 1)
    alpha = coastal_ts_sst_diff.min(dim="time").load()
    beta = coastal_ts_sst_diff.max(dim="time").load()
    solid_frac = (1 - ofrac).where(ofrac < cutoff, 1 - cutoff)
    min_solid_ts = sst + alpha / solid_frac
    max_solid_ts = sst + beta / solid_frac
    return min_solid_ts, max_solid_ts


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
        sst_from_ts: If true, then SSTs come from the ts field after computing the
            mean on windows of size equal to the ocean's timedelta.

    """

    how: Literal["solid_ts", "interpolate_sst", "threshold"]
    ocean_fraction_threshold: float = 1.0
    sst_from_ts: bool = False

    def get_window_mean_sst(
        self,
        sst: xr.DataArray,
        ts: xr.DataArray,
        sst_timedelta: str,
    ) -> xr.DataArray:
        if self.sst_from_ts:
            sst0 = sst.isel(time=0).drop_vars("time")  # for ocean mask
            sst = (
                ts.where(sst0.notnull())
                .resample(time=sst_timedelta, closed="right", label="right")
                .mean()
            )
        return sst.reindex(time=ts["time"], method="bfill")

    def apply_sst_to_ts(
        self,
        ts: xr.DataArray,
        sst: xr.DataArray,
        ofrac: xr.DataArray,
    ) -> xr.DataArray:
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
            ofrac = ofrac.where(ofrac < thresh, 1.0)
            ts_mod = (1 - ofrac) * ts + ofrac * sst
        else:
            # simple threshold
            ts_mod = ts.where(ofrac < thresh, sst)
        return ts_mod.fillna(ts)


@dataclasses.dataclass
class CoupledSeaIceConfig:
    """Configuration for creating the sea ice mask.

    Parameters:
        sst_threshold: Upper threshold for the SST time mean in degrees Kelvin.
            The resulting sea ice mask is 1 where the time mean SST < the
            threshold and 0 otherwise.
        extra_masked_names: Additional sea-ice-related variable names for which
            copies of the mask will be added as "mask_{extra_name}". NOTE: The
            sea-ice fraction and "ocean" sea-ice fraction masks don't need to be
            specified here.

    """

    sst_threshold: float
    extra_masked_names: list[str] = dataclasses.field(default_factory=list)
    _mask: xr.DataArray | None = None

    def compute_sea_ice_mask(self, sst: xr.DataArray) -> xr.DataArray:
        sst_tm = sst.mean(dim="time")
        self._mask = (sst_tm < self.sst_threshold).fillna(0.0)
        return self._mask

    def apply_mask(self, da: xr.DataArray) -> xr.DataArray:
        if self._mask is None:
            raise RuntimeError("Call compute_sea_ice_mask before apply_mask.")
        return da.where(self._mask > 0)


@dataclasses.dataclass
class CoupledBoundaryConditionsConfig:
    """Configuration for creating a coupled surface boundary conditions dataset
    from separately processed ocean and atmosphere datasets.

    The output dataset zarr directory will be named like
      f'{todays_date}-{output_name_prefix}-ocean.zarr' and
      f'{todays_date}-{output_name_prefix}-{how}-atmosphere.zarr'
      where 'how' is f'{coupled_ts.how}' or f'{coupled_ts.how}-sst_from_ts'
      depending on the value of coupled_ts.sst_from_ts.

    Parameters:
        coupled_ts: Configuration for surface temperature.
        coupled_sea_ice: Configuration for sea ice.
        output_name_prefix: Prefix for output dataset naming.
        output_directory: Directory where the output datasets will be created.
        ocean: Configuration of preprocessed ocean dataset.
        atmosphere: Configuration of preprocessed atmosphere dataset.
        ocean_fields: Configuration of ocean variable names.
        atmosphere_fields: Configuration of atmosphere variable names.
        derived_fields: Configuration of names for derived variables.
        output_writer: Configuration for dask and xpartition.

    """

    coupled_ts: CoupledSurfaceTemperatureConfig
    coupled_sea_ice: CoupledSeaIceConfig
    output_name_prefix: str
    output_directory: str
    ocean: InputDatasetConfig
    atmosphere: InputDatasetConfig
    ocean_fields: OceanInputFieldsConfig = dataclasses.field(
        default_factory=OceanInputFieldsConfig
    )
    atmosphere_fields: AtmosphereInputFieldsConfig = dataclasses.field(
        default_factory=AtmosphereInputFieldsConfig
    )
    derived_fields: DerivedFieldsConfig = dataclasses.field(
        default_factory=DerivedFieldsConfig
    )
    output_writer: OutputWriterConfig = dataclasses.field(
        default_factory=OutputWriterConfig
    )
    version: str | None = None

    @classmethod
    def from_file(cls, path: str) -> "CoupledBoundaryConditionsConfig":
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )


def _add_history(
    ds: xr.Dataset,
    urls: list[str],
    yaml_path: str,
    subsample: bool,
):
    script_dir = Path(__file__).parent
    git_sha = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=script_dir, stderr=subprocess.DEVNULL
        )
        .strip()
        .decode()
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    history_entry = f"""
  - timestamp: {timestamp}
  - script: full-model/scripts/coupled/create_surface_fractions_datasets.py
  - git_sha: {git_sha}
  - arguments:
    - yaml: {yaml_path}
    - subsample: {subsample}
  - input_sources: {urls}
"""
    ds.attrs = {"history": history_entry}


@click.command()
@click.option("--yaml", help="Path to dataset configuration YAML file.")
@click.option(
    "--debug",
    is_flag=True,
    help="Print metadata instead of writing output.",
)
@click.option(
    "--subsample", is_flag=True, help="Subsample one year of the data before writing."
)
@click.option(
    "--dataset",
    default="both",
    help="Which dataset to write.",
)
def main(
    yaml: str,
    debug: bool,
    subsample: bool,
    dataset: Literal["both", "ocean", "atmosphere"],
):
    config = CoupledBoundaryConditionsConfig.from_file(yaml)
    print(f"Input ocean dataset: {config.ocean.zarr_path}")
    print(f"Input atmosphere dataset: {config.atmosphere.zarr_path}")
    if config.version is None:
        version = datetime.today().strftime("%Y-%m-%d")
    else:
        version = config.version
    ocean_output_store = os.path.join(
        config.output_directory,
        f"{version}-{config.output_name_prefix}-ocean.zarr",
    )
    atmos_prefix = f"{version}-{config.output_name_prefix}-{config.coupled_ts.how}"
    if config.coupled_ts.sst_from_ts:
        atmos_prefix += "-sst_from_ts"
    atmos_output_store = os.path.join(
        config.output_directory,
        f"{atmos_prefix}-atmosphere.zarr",
    )
    if subsample:
        ocean_output_store = ocean_output_store.replace(".zarr", "-subsample.zarr")
        atmos_output_store = atmos_output_store.replace(".zarr", "-subsample.zarr")
    print(f"Ocean output store: {ocean_output_store}")
    print(f"Atmosphere output store: {atmos_output_store}")

    config.output_writer.start_dask_client(debug)

    urls = [config.ocean.zarr_path, config.atmosphere.zarr_path]

    ocean = config.ocean.get_dataset()
    atmos = config.atmosphere.get_dataset()

    # ensure matching horizontal coordinates
    # FIXME: assumption about horizontal dim names
    ocean = ocean.assign_coords({"lat": atmos.lat, "lon": atmos.lon})

    # atmosphere inputs
    ts_name = config.atmosphere_fields.surface_temperature_name
    sic_name = config.atmosphere_fields.sea_ice_fraction_name
    lfrac_name = config.atmosphere_fields.land_fraction_name
    ofrac_name = config.atmosphere_fields.ocean_fraction_name

    # ocean inputs
    sfrac_name = config.ocean_fields.sea_surface_fraction_name
    sst_name = config.ocean_fields.sea_surface_temperature_name

    # derived outputs
    osic_name = config.derived_fields.ocean_sea_ice_fraction_name

    ts = atmos[ts_name]
    # FIXME: clipping and nan filling should be already be done in the input ds
    sic = atmos[sic_name].clip(min=0.0, max=1.0)
    lfrac = atmos[lfrac_name].clip(min=0.0, max=1.0)
    ofrac = atmos[ofrac_name].clip(min=0.0, max=1.0)
    try:
        sfrac = ocean[sfrac_name].fillna(0.0).clip(min=0.0, max=1.0)
    except KeyError:
        print(
            f"Warning: {sfrac_name} not found in ocean dataset. "
            "Assuming sea surface fraction is 1 - land fraction."
        )
        sfrac = 1 - lfrac
        sfrac = sfrac.fillna(0.0).clip(min=0.0, max=1.0)
        sfrac.attrs = {
            "long_name": "sea surface fraction",
            "units": "unitless",
        }
    sst = ocean[sst_name]

    # ensure agreement between atmosphere fractions and sea surface fraction

    lfrac_mod = lfrac.where(lfrac > EPS, 0.0).where(sfrac > 0, 1.0)
    ofrac_mod = ofrac.where(lfrac_mod < 1.0, 0.0)
    sic_mod = sic.where(lfrac_mod < 1.0, 0.0).where(ofrac_mod < 1.0, 0.0)

    # get resampled mean of ocean and sea-ice fractions to match ocean step size

    ofrac_mod = ofrac_mod.resample(
        time=config.ocean.timedelta, closed="right", label="right"
    ).mean()
    ofrac_mod = ofrac_mod.where(ofrac_mod > EPS, 0.0)
    sic_mod = sic_mod.resample(
        time=config.ocean.timedelta, closed="right", label="right"
    ).mean()
    sic_mod = sic_mod.where(sic_mod > EPS, 0.0)

    # compute ocean_sea_ice_fraction
    osic = sic_mod / (1.0 - lfrac_mod)
    osic = osic.where(lfrac_mod != 1.0, float("nan")).where(osic > EPS, 0.0)
    osic.attrs = {
        "long_name": "sea ice fraction / (1 - land_fraction)",
        "units": "unitless",
    }

    sea_ice_mask = config.coupled_sea_ice.compute_sea_ice_mask(sst)
    sea_ice_mask.attrs = {
        "long_name": "sea ice mask",
        "units": "unitless",
    }

    # create the ocean output dataset

    lfrac_mod.attrs = lfrac.attrs
    ofrac_mod.attrs = ofrac.attrs
    sic_mod.attrs = sic.attrs

    extra_sea_ice_masked_vars = {
        name: config.coupled_sea_ice.apply_mask(ocean[name])
        for name in config.coupled_sea_ice.extra_masked_names
    }
    for name in config.coupled_sea_ice.extra_masked_names:
        extra_sea_ice_masked_vars[name].attrs = ocean[name].attrs
        extra_sea_ice_masked_vars[f"mask_{name}"] = sea_ice_mask

    ocean_output = xr.Dataset(
        {
            lfrac_name: lfrac_mod,
            ofrac_name: ofrac_mod,
            osic_name: config.coupled_sea_ice.apply_mask(osic),
            sic_name: config.coupled_sea_ice.apply_mask(sic_mod),
            sfrac_name: sfrac,
            sst_name: sst,
            f"mask_{sic_name}": sea_ice_mask,
            f"mask_{osic_name}": sea_ice_mask,
            **extra_sea_ice_masked_vars,
        }
    )
    # add additional variables
    ocean_output = config.ocean.copy_extra_data_vars(ocean, ocean_output)
    _add_history(ocean_output, urls, yaml, subsample)
    print(f"Ocean output dataset size is {ocean_output.nbytes / 1e9} GB")

    if subsample:
        ocean_output = ocean_output.isel(time=slice(None, 73))

    if debug:
        with xr.set_options(display_max_rows=500):
            print(ocean_output)
    elif dataset in ["both", "ocean"]:
        config.output_writer.write(ocean_output, ocean_output_store)

    if dataset == "ocean":
        config.output_writer.close_dask_client()
        return

    # apply SST to atmosphere surface temperature field

    ofrac_reindex = ofrac_mod.reindex(time=ts["time"], method="bfill")
    sic_reindex = sic_mod.reindex(time=ts["time"], method="bfill")
    sst_reindex = config.coupled_ts.get_window_mean_sst(
        sst,
        ts,
        config.ocean.timedelta,
    )

    ts_mod = config.coupled_ts.apply_sst_to_ts(
        ts,
        sst_reindex,
        ofrac_reindex,
    )

    time = xr.date_range(
        str(atmos["time"].isel(time=0).values.item()),
        str(atmos["time"].isel(time=-1).values.item()),
        freq=config.atmosphere.timedelta,
        calendar=atmos["time"].isel(time=0).values.item().calendar,
        use_cftime=True,
    )  # this is needed for proper time coord serialization
    assert len(time) == len(atmos.time)
    time.attrs = atmos["time"].attrs

    atmos_output = xr.Dataset(
        {
            lfrac_name: lfrac_mod,
            ofrac_name: ofrac_reindex,
            sic_name: sic_reindex,
            sfrac_name: sfrac,
            ts_name: ts_mod,
        }
    )
    atmos_output["time"] = time
    # add additional variables
    atmos_output = config.atmosphere.copy_extra_data_vars(atmos, atmos_output)
    _add_history(atmos_output, urls, yaml, subsample)
    print(f"Atmosphere output dataset size is {atmos_output.nbytes / 1e9} GB")

    if subsample:
        atmos_output = atmos_output.isel(time=slice(None, 365 * 4))

    if debug:
        with xr.set_options(display_max_rows=500):
            print(atmos_output)
    elif dataset in ["both", "atmosphere"]:
        config.output_writer.write(atmos_output, atmos_output_store)

    config.output_writer.close_dask_client()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        pdb.post_mortem()  # Start the debugger
        raise  # Re-raise the exception to preserve the traceback
