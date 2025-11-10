import dataclasses
import datetime
import pathlib
import re
from typing import Literal, cast

import cftime
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.inference import ExplicitIndices, ForcingDataLoaderConfig
from fme.ace.data_loading.test_data_loader import _get_coords
from fme.ace.requirements import DataRequirements
from fme.ace.testing import save_scalar_netcdf
from fme.core.coordinates import (
    HorizontalCoordinates,
    OptionalDepthCoordinate,
    OptionalHybridSigmaPressureCoordinate,
    VerticalCoordinate,
)
from fme.core.dataset.merged import MergeNoConcatDatasetConfig
from fme.core.dataset.xarray import (
    XarrayDataConfig,
    _get_mask_provider,
    _get_vertical_coordinate,
    get_horizontal_coordinates,
)
from fme.core.mask_provider import MaskProvider
from fme.core.typing_ import Slice
from fme.coupled.data_loading.batch_data import CoupledBatchData, CoupledPrognosticState
from fme.coupled.data_loading.data_typing import (
    CoupledHorizontalCoordinates,
    CoupledVerticalCoordinate,
)
from fme.coupled.requirements import CoupledDataRequirements

from .config import CoupledDataLoaderConfig, CoupledDatasetConfig
from .getters import get_forcing_data, get_gridded_data
from .inference import (
    CoupledForcingDataLoaderConfig,
    InferenceDataLoaderConfig,
    InferenceDataset,
)

N_LAT = 16
N_LON = 32


def _save_netcdf(
    filename,
    dim_sizes,
    variable_names,
    calendar,
    realm: Literal["ocean", "atmosphere"],
    timestep_size=1,
    timestep_start=0,
    nz=3,
):
    data_vars = {}
    dim_sizes_without_time = {k: v for k, v in dim_sizes.items() if k != "time"}
    for name in variable_names:
        if name == "constant_mask" or name.startswith("mask_"):
            dim_sizes_to_use = dim_sizes_without_time
            rng = np.random.default_rng()
            data = rng.integers(
                low=0, high=2, size=list(dim_sizes_to_use.values())
            ).astype(np.float32)
        elif name == "land_fraction":
            dim_sizes_to_use = dim_sizes_without_time
            data = np.ones(tuple(dim_sizes_to_use.values()))
            data[0, 0] = 0.0
        else:
            dim_sizes_to_use = dim_sizes
            data = np.random.uniform(size=list(dim_sizes_to_use.values()))
        data_vars[name] = xr.DataArray(
            data, dims=list(dim_sizes_to_use), attrs={"units": "m", "long_name": name}
        )
    coords = _get_coords(dim_sizes, calendar, timestep_size, timestep_start)
    if realm == "atmosphere":
        for i in range(nz):
            data_vars[f"ak_{i}"] = float(i)
            data_vars[f"bk_{i}"] = float(i + 1)
    elif realm == "ocean":
        if "mask_2d" in data_vars and "mask_0" in data_vars:
            data_vars["mask_2d"] = data_vars["mask_0"]
        if "mask_2d" in data_vars or "mask_0" in data_vars:
            mask = data_vars.get("mask_2d", data_vars.get("mask_0"))
            assert mask is not None
            # add nans to 2D vars
            names = [
                name
                for name in data_vars
                if re.search(r"_\d+$", name) is None and name != "mask_2d"
            ]
            for name in names:
                data_vars[name] = data_vars[name].where(mask == 1, float("nan"))
        for i in range(nz):
            if f"mask_{i}" in data_vars:
                # add nans to 3D vars
                mask = data_vars[f"mask_{i}"]
                names = [
                    name
                    for name in data_vars
                    if name != f"mask_{i}" and name.endswith(f"_{i}")
                ]
                for name in names:
                    data_vars[name] = data_vars[name].where(mask == 1, float("nan"))
            data_vars[f"idepth_{i}"] = float(i)
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")
    return ds


@dataclasses.dataclass
class MockComponentData:
    ds: xr.Dataset
    data_dir: str
    means_path: str
    stds_path: str
    timedelta: str

    def __post_init__(self):
        self.ds["time"] = cftime.num2date(
            self.ds["time"].values,
            units=self.ds["time"].units,
            calendar=self.ds["time"].calendar,
        )

    @property
    def timestep(self) -> datetime.timedelta:
        return pd.Timedelta(self.timedelta).to_pytimedelta()

    @property
    def mask_provider(self) -> MaskProvider:
        return _get_mask_provider(self.ds, dtype=None)

    @property
    def vcoord(self) -> VerticalCoordinate:
        return _get_vertical_coordinate(self.ds, dtype=None)

    @property
    def hcoord(self) -> HorizontalCoordinates:
        return get_horizontal_coordinates(
            self.ds,
            spatial_dimensions="latlon",
            dtype=None,
        )[0]


@dataclasses.dataclass
class MockCoupledData:
    ocean: MockComponentData
    atmosphere: MockComponentData

    @property
    def n_times_ocean(self):
        return len(self.ocean.ds["time"])

    @property
    def n_times_atmosphere(self):
        return len(self.atmosphere.ds["time"])

    @property
    def img_shape(self) -> tuple[int, int]:
        # NOTE: assumes atmosphere has same img_shape
        return self.ocean.ds[next(iter(self.ocean.ds.data_vars))].shape[-2:]

    @property
    def hcoord(self) -> CoupledHorizontalCoordinates:
        return CoupledHorizontalCoordinates(
            ocean=self.ocean.hcoord,
            atmosphere=self.atmosphere.hcoord,
        )

    @property
    def vcoord(self) -> CoupledVerticalCoordinate:
        return CoupledVerticalCoordinate(
            ocean=cast(OptionalDepthCoordinate, self.ocean.vcoord),
            atmosphere=cast(
                OptionalHybridSigmaPressureCoordinate, self.atmosphere.vcoord
            ),
        )

    @property
    def dataset_config(self) -> CoupledDatasetConfig:
        return CoupledDatasetConfig(
            ocean=XarrayDataConfig(str(self.ocean.data_dir)),
            atmosphere=XarrayDataConfig(str(self.atmosphere.data_dir)),
        )


def create_coupled_data_on_disk(
    data_dir: pathlib.Path,
    n_forward_times_ocean: int,
    n_forward_times_atmosphere: int,
    ocean_names: list[str],
    atmosphere_names: list[str],
    atmosphere_start_time_offset_from_ocean: bool,
    n_levels_ocean: int = 2,
    n_levels_atmosphere: int = 2,
    timestep_start: int = 0,
) -> MockCoupledData:
    np.random.seed(0)

    ocean_dir = data_dir / "ocean"
    ocean_dir.mkdir()
    ocean_dim_sizes = {"time": n_forward_times_ocean + 1, "lat": N_LAT, "lon": N_LON}
    ocean_timestep_size = n_forward_times_atmosphere / n_forward_times_ocean
    if ocean_timestep_size != int(ocean_timestep_size):
        raise ValueError(
            "n_forward_times_atmosphere should be a multiple of n_forward_times_ocean."
        )
    ocean_timestep_size = int(ocean_timestep_size)
    ocean_ds = _save_netcdf(
        filename=ocean_dir / "data.nc",
        dim_sizes=ocean_dim_sizes,
        variable_names=ocean_names,
        calendar="proleptic_gregorian",
        realm="ocean",
        # _save_netcdf has a default timestep of 1 day which we interpret as the
        # atmosphere timestep, so the ocean data needs
        timestep_size=ocean_timestep_size,
        nz=n_levels_ocean + 1,
        timestep_start=timestep_start,
    )

    atmos_dir = data_dir / "atmos"
    atmos_dir.mkdir()
    n_times_atmos = n_forward_times_atmosphere + 1
    timestep_start_atmosphere = 0
    if atmosphere_start_time_offset_from_ocean:
        n_times_atmos += 1
        timestep_start_atmosphere = -1
    if timestep_start != 0:
        timestep_start_atmosphere = timestep_start
    atmos_dim_sizes = {
        "time": n_times_atmos,
        "lat": N_LAT,
        "lon": N_LON,
    }
    atmos_ds = _save_netcdf(
        filename=atmos_dir / "data.nc",
        dim_sizes=atmos_dim_sizes,
        variable_names=atmosphere_names,
        calendar="proleptic_gregorian",
        realm="atmosphere",
        timestep_size=1,
        timestep_start=timestep_start_atmosphere,
        nz=n_levels_atmosphere + 1,
    )
    # _save_netcdf creates integer times in units of "days since 1970-01-01"
    timedelta_atmos = "1D"
    timedelta_ocean = f"{ocean_timestep_size}D"

    stats_dir = data_dir / "stats"
    stats_dir.mkdir()
    all_names = list(set(ocean_names + atmosphere_names))
    save_scalar_netcdf(stats_dir / "means.nc", variable_names=all_names)
    save_scalar_netcdf(stats_dir / "stds.nc", variable_names=all_names)
    return MockCoupledData(
        ocean=MockComponentData(
            ds=ocean_ds,
            data_dir=str(ocean_dir),
            means_path=str(stats_dir / "means.nc"),
            stds_path=str(stats_dir / "stds.nc"),
            timedelta=timedelta_ocean,
        ),
        atmosphere=MockComponentData(
            ds=atmos_ds,
            data_dir=str(atmos_dir),
            means_path=str(stats_dir / "means.nc"),
            stds_path=str(stats_dir / "stds.nc"),
            timedelta=timedelta_atmos,
        ),
    )


@pytest.mark.parametrize("atmosphere_times_offset", [False, True])
def test_coupled_data_loader(tmp_path, atmosphere_times_offset: bool):
    """Tests that the coupled loader returns the correct number of timesteps."""

    # Create datasets with fast and slow timesteps.
    ocean_names = ["bar"]
    atmos_names = ["foo"]

    n_ics = 2
    ics = []
    for i in range(n_ics):
        # create dataset with 2 samples
        ic_path = tmp_path / f"ic{i}"
        ic_path.mkdir()
        ic = create_coupled_data_on_disk(
            ic_path,
            n_forward_times_ocean=2,
            n_forward_times_atmosphere=4,
            ocean_names=ocean_names,
            atmosphere_names=atmos_names,
            atmosphere_start_time_offset_from_ocean=atmosphere_times_offset,
        )
        ics.append(ic)

    # subset atmosphere data to align with beginning of ocean data
    if atmosphere_times_offset:
        atmos_data_subset = Slice(start=1)
    else:
        atmos_data_subset = Slice()

    config = CoupledDataLoaderConfig(
        dataset=[
            CoupledDatasetConfig(
                ocean=XarrayDataConfig(
                    data_path=ics[i].ocean.data_dir,
                ),
                atmosphere=XarrayDataConfig(
                    data_path=ics[i].atmosphere.data_dir,
                    subset=atmos_data_subset,
                ),
            )
            for i in range(n_ics)
        ],
        batch_size=1,
        num_data_workers=0,
        strict_ensemble=True,
    )
    coupled_requirements = CoupledDataRequirements(
        ocean_timestep=datetime.timedelta(days=2),
        ocean_requirements=DataRequirements(ocean_names, n_timesteps=2),
        atmosphere_timestep=datetime.timedelta(days=1),
        atmosphere_requirements=DataRequirements(atmos_names, n_timesteps=3),
    )
    # unshuffled data loader
    data = get_gridded_data(config, False, coupled_requirements)

    assert data.n_batches == 2 * n_ics  # 2 samples per IC
    for batch in data.loader:
        ocean_data = batch.ocean_data
        atmosphere_data = batch.atmosphere_data
        assert isinstance(ocean_data, BatchData)
        assert isinstance(atmosphere_data, BatchData)
        assert len(ocean_data.time.isel(sample=0).values) == 2  # IC + 1 forward step
        assert (
            len(atmosphere_data.time.isel(sample=0).values) == 3
        )  # IC + 2 forward steps
        assert set(ocean_data.data.keys()) == set(ocean_names)
        assert set(atmosphere_data.data.keys()) == set(atmos_names)
        # initial condition times should match:
        assert ocean_data.time.isel(time=0) == atmosphere_data.time.isel(time=0)
        # final step times should match:
        assert ocean_data.time.isel(time=-1) == atmosphere_data.time.isel(time=-1)
        # check data matches expectations
        assert ocean_data.data["bar"].shape[1] == 2
        assert atmosphere_data.data["foo"].shape[1] == 3

    # check that the sample data matches
    ic_idx = 0
    sample_idx = 1
    ocean_ds = ics[ic_idx].ocean.ds
    atmos_ds = ics[ic_idx].atmosphere.ds
    sample = data._loader._dataset[sample_idx]
    ocean_sample_init_time = sample.ocean[1].isel(time=0).item()
    atmos_sample_init_time = sample.atmosphere[1].isel(time=0).item()

    # we already checked for matching ocean/atmos sample init times above, now
    # checking that they match times at the right positions in each dataset
    offset = 1 if atmosphere_times_offset else 0  # for atmosphere data times on disk
    assert ocean_sample_init_time == ocean_ds.isel(time=1)["time"].item()
    assert atmos_sample_init_time == atmos_ds.isel(time=2 + offset)["time"].item()

    expected_ocean = ocean_ds["bar"].isel(time=slice(1, 3)).values
    expected_atmos = atmos_ds["foo"].isel(time=slice(2 + offset, 5 + offset)).values
    # check that
    assert np.allclose(sample.ocean[0]["bar"].cpu().numpy(), expected_ocean)
    assert np.allclose(sample.atmosphere[0]["foo"].cpu().numpy(), expected_atmos)


def test_zarr_engine_used_true():
    config = CoupledDataLoaderConfig(
        dataset=[
            CoupledDatasetConfig(
                ocean=XarrayDataConfig(data_path="ocean", engine="netcdf4"),
                atmosphere=XarrayDataConfig(
                    data_path="atmos", file_pattern="data.zarr", engine="zarr"
                ),
            ),
            CoupledDatasetConfig(
                ocean=XarrayDataConfig(data_path="ocean", engine="netcdf4"),
                atmosphere=XarrayDataConfig(data_path="atmos", engine="netcdf4"),
            ),
        ],
        batch_size=1,
    )
    assert config.zarr_engine_used


def test_zarr_engine_used_false():
    config = CoupledDataLoaderConfig(
        dataset=[
            CoupledDatasetConfig(
                ocean=XarrayDataConfig(data_path="ocean", engine="netcdf4"),
                atmosphere=XarrayDataConfig(data_path="atmos", engine="netcdf4"),
            )
        ],
        batch_size=1,
    )
    assert not config.zarr_engine_used


def test_zarr_engine_used_true_inference():
    config = InferenceDataLoaderConfig(
        dataset=CoupledDatasetConfig(
            ocean=XarrayDataConfig(data_path="ocean", engine="netcdf4"),
            atmosphere=XarrayDataConfig(
                data_path="atmos", file_pattern="data.zarr", engine="zarr"
            ),
        ),
        start_indices=ExplicitIndices([0]),
    )
    assert config.zarr_engine_used


def test_zarr_engine_used_false_inference():
    config = InferenceDataLoaderConfig(
        dataset=CoupledDatasetConfig(
            ocean=XarrayDataConfig(data_path="ocean", engine="netcdf4"),
            atmosphere=XarrayDataConfig(data_path="atmos", engine="netcdf4"),
        ),
        start_indices=ExplicitIndices([0]),
    )
    assert not config.zarr_engine_used


def test_coupled_data_loader_merge_no_concat(tmp_path):
    ocean_names = ["var_ocean_1", "var_ocean_2"]
    atmos_names = ["var_atmos_1", "var_atmos_2"]
    n_forward_times_ocean = 2
    n_forward_times_atmosphere = 4

    data_path = tmp_path / "data"
    data_path.mkdir()

    ocean_dir = data_path / "ocean"
    ocean_dir_part1 = data_path / "ocean_part1"
    ocean_dir_part2 = data_path / "ocean_part2"
    ocean_dir.mkdir()
    ocean_dir_part1.mkdir()
    ocean_dir_part2.mkdir()

    atmos_dir = data_path / "atmos"
    atmos_dir.mkdir()

    ocean_dim_sizes = {"time": n_forward_times_ocean + 1, "lat": N_LAT, "lon": N_LON}
    atmos_dim_sizes = {
        "time": n_forward_times_atmosphere + 1,
        "lat": N_LAT,
        "lon": N_LON,
    }

    ocean_timestep_size = n_forward_times_atmosphere // n_forward_times_ocean

    ocean_config: MergeNoConcatDatasetConfig | XarrayDataConfig
    atmos_config: MergeNoConcatDatasetConfig | XarrayDataConfig
    _save_netcdf(
        filename=ocean_dir_part1 / "data.nc",
        dim_sizes=ocean_dim_sizes,
        variable_names=[ocean_names[0]],
        calendar="proleptic_gregorian",
        realm="ocean",
        timestep_size=ocean_timestep_size,
    )
    _save_netcdf(
        filename=ocean_dir_part2 / "data.nc",
        dim_sizes=ocean_dim_sizes,
        variable_names=[ocean_names[1]],
        calendar="proleptic_gregorian",
        realm="ocean",
        timestep_size=ocean_timestep_size,
    )
    _save_netcdf(
        filename=atmos_dir / "data.nc",
        dim_sizes=atmos_dim_sizes,
        variable_names=atmos_names,
        calendar="proleptic_gregorian",
        realm="atmosphere",
    )
    ocean_config = MergeNoConcatDatasetConfig(
        merge=[
            XarrayDataConfig(data_path=str(ocean_dir_part1)),
            XarrayDataConfig(data_path=str(ocean_dir_part2)),
        ]
    )
    atmos_config = XarrayDataConfig(data_path=str(atmos_dir))

    # test CoupledDataLoaderConfig
    config = CoupledDataLoaderConfig(
        dataset=[
            CoupledDatasetConfig(
                ocean=ocean_config,
                atmosphere=atmos_config,
            )
        ],
        batch_size=1,
        num_data_workers=0,
    )
    coupled_requirements = CoupledDataRequirements(
        ocean_timestep=datetime.timedelta(days=ocean_timestep_size),
        ocean_requirements=DataRequirements(ocean_names, n_timesteps=2),
        atmosphere_timestep=datetime.timedelta(days=1),
        atmosphere_requirements=DataRequirements(atmos_names, n_timesteps=3),
    )

    data = get_gridded_data(config, False, coupled_requirements)
    batch = next(iter(data.loader))
    assert set(batch.ocean_data.data.keys()) == set(ocean_names)
    assert set(batch.atmosphere_data.data.keys()) == set(atmos_names)

    # test InferenceDataLoaderConfig
    inference_config = InferenceDataLoaderConfig(
        dataset=CoupledDatasetConfig(
            ocean=ocean_config,
            atmosphere=atmos_config,
        ),
        start_indices=ExplicitIndices([0]),
    )
    dataset = InferenceDataset(
        config=inference_config,
        total_coupled_steps=1,
        requirements=coupled_requirements,
        dataset_info=data.dataset_info,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # batching is handled by the dataset
        num_workers=inference_config.num_data_workers,
    )

    inference_batch = next(iter(loader))
    assert set(inference_batch.ocean_data.data.keys()) == set(ocean_names)
    assert set(inference_batch.atmosphere_data.data.keys()) == set(atmos_names)


@pytest.mark.parametrize("n_initial_conditions", [1, 2])
def test_get_forcing_data(tmp_path, n_initial_conditions):
    calendar = "proleptic_gregorian"
    ocean_names = ["bar"]
    atmos_names = ["foo"]
    n_forward_times_ocean = 4
    n_forward_times_atmosphere = 8
    inner_steps = 2
    total_coupled_steps = 2
    coupled_steps_in_memory = 2
    create_coupled_data_on_disk(
        tmp_path,
        n_forward_times_ocean=n_forward_times_ocean,
        n_forward_times_atmosphere=n_forward_times_atmosphere,
        ocean_names=ocean_names,
        atmosphere_names=atmos_names,
        atmosphere_start_time_offset_from_ocean=False,
        timestep_start=-1,
    )
    config = CoupledForcingDataLoaderConfig(
        atmosphere=ForcingDataLoaderConfig(
            XarrayDataConfig(data_path=tmp_path / "atmos")
        ),
        ocean=ForcingDataLoaderConfig(XarrayDataConfig(data_path=tmp_path / "ocean")),
    )
    ocean_timestep_size = n_forward_times_atmosphere / n_forward_times_ocean
    window_requirements = CoupledDataRequirements(
        ocean_timestep=datetime.timedelta(days=ocean_timestep_size),
        ocean_requirements=DataRequirements(
            ocean_names, n_timesteps=coupled_steps_in_memory + 1
        ),
        atmosphere_timestep=datetime.timedelta(days=1),
        atmosphere_requirements=DataRequirements(
            atmos_names, n_timesteps=coupled_steps_in_memory * inner_steps + 1
        ),
    )
    atmos_initial_condition = BatchData.new_for_testing(
        names=atmos_names,
        n_samples=n_initial_conditions,
        n_timesteps=1,
        t_initial=cftime.datetime(1970, 1, 2),
        calendar=calendar,
    )
    ocean_initial_condition = BatchData.new_for_testing(
        names=ocean_names,
        n_samples=n_initial_conditions,
        n_timesteps=1,
        t_initial=cftime.datetime(1970, 1, 2),
        calendar=calendar,
    )
    data = get_forcing_data(
        config,
        total_coupled_steps,
        window_requirements=window_requirements,
        initial_condition=CoupledPrognosticState(
            ocean_data=PrognosticState(ocean_initial_condition),
            atmosphere_data=PrognosticState(atmos_initial_condition),
        ),  # type: ignore
    )
    batch_data = next(iter(data.loader))
    assert isinstance(batch_data, CoupledBatchData)
    assert isinstance(batch_data.ocean_data.data["bar"], torch.Tensor)
    assert isinstance(batch_data.atmosphere_data.data["foo"], torch.Tensor)
    assert batch_data.ocean_data.data["bar"].shape[0] == n_initial_conditions
    assert batch_data.ocean_data.data["bar"].shape[1] == total_coupled_steps + 1
    assert batch_data.atmosphere_data.data["foo"].shape[0] == n_initial_conditions
    assert (
        batch_data.atmosphere_data.data["foo"].shape[1]
        == total_coupled_steps * inner_steps + 1
    )
    assert list(batch_data.ocean_data.time.dims) == ["sample", "time"]
    assert list(batch_data.atmosphere_data.time.dims) == ["sample", "time"]
    xr.testing.assert_allclose(
        batch_data.ocean_data.time[:, 0], ocean_initial_condition.time[:, 0]
    )
    xr.testing.assert_allclose(
        batch_data.atmosphere_data.time[:, 0], atmos_initial_condition.time[:, 0]
    )
    assert batch_data.ocean_data.time.dt.calendar == calendar
    assert batch_data.atmosphere_data.time.dt.calendar == calendar
    xr.testing.assert_equal(
        data.initial_condition.ocean_data.as_batch_data().time,
        ocean_initial_condition.time,
    )
    xr.testing.assert_equal(
        data.initial_condition.atmosphere_data.as_batch_data().time,
        atmos_initial_condition.time,
    )
    np.testing.assert_allclose(
        data.initial_condition.ocean_data.as_batch_data().data["bar"].cpu().numpy(),
        ocean_initial_condition.data["bar"].cpu().numpy(),
    )
    np.testing.assert_allclose(
        data.initial_condition.atmosphere_data.as_batch_data()
        .data["foo"]
        .cpu()
        .numpy(),
        atmos_initial_condition.data["foo"].cpu().numpy(),
    )
