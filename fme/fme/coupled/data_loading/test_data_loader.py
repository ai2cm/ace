import dataclasses
import datetime
import pathlib
from typing import List

import cftime
import numpy as np
import xarray as xr

from fme.core.data_loading.batch_data import BatchData
from fme.core.data_loading.config import XarrayDataConfig
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.test_data_loader import _save_netcdf

from .config import CoupledDataConfig, CoupledDataLoaderConfig
from .getters import get_coupled_data_loader
from .requirements import CoupledDataRequirements


@dataclasses.dataclass
class MockCoupledData:
    ocean: xr.Dataset
    atmosphere: xr.Dataset
    ocean_dir: str
    atmosphere_dir: str

    def __post_init__(self):
        self.ocean["time"] = cftime.num2date(
            self.ocean["time"].values,
            units=self.ocean["time"].units,
            calendar=self.ocean["time"].calendar,
        )
        self.atmosphere["time"] = cftime.num2date(
            self.atmosphere["time"].values,
            units=self.atmosphere["time"].units,
            calendar=self.atmosphere["time"].calendar,
        )


def create_coupled_data_on_disk(
    data_dir: pathlib.Path,
    n_forward_times_ocean: int,
    n_forward_times_atmosphere: int,
    ocean_names: List[str],
    atmosphere_names: List[str],
) -> MockCoupledData:
    np.random.seed(0)

    ocean_dir = data_dir / "ocean"
    ocean_dir.mkdir()
    ocean_dim_sizes = {"time": n_forward_times_ocean + 1, "lat": 4, "lon": 8}
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
        # _save_netcdf has a default timestep of 1 day which we interpret as the
        # atmosphere timestep, so the ocean data needs
        timestep_size=ocean_timestep_size,
    )

    atmos_dir = data_dir / "atmos"
    atmos_dir.mkdir()
    atmos_dim_sizes = {"time": n_forward_times_atmosphere + 1, "lat": 4, "lon": 8}
    atmos_ds = _save_netcdf(
        filename=atmos_dir / "data.nc",
        dim_sizes=atmos_dim_sizes,
        variable_names=atmosphere_names,
        calendar="proleptic_gregorian",
        timestep_size=1,
    )

    return MockCoupledData(
        ocean=ocean_ds,
        atmosphere=atmos_ds,
        ocean_dir=str(ocean_dir),
        atmosphere_dir=str(atmos_dir),
    )


def test_coupled_data_loader(tmp_path):
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
        )
        ics.append(ic)

    config = CoupledDataLoaderConfig(
        dataset=[
            CoupledDataConfig(
                ocean=XarrayDataConfig(data_path=ics[i].ocean_dir),
                atmosphere=XarrayDataConfig(data_path=ics[i].atmosphere_dir),
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
    data = get_coupled_data_loader(config, False, coupled_requirements)

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
    ocean_ds = ics[ic_idx].ocean
    atmos_ds = ics[ic_idx].atmosphere
    sample = data._loader.dataset[sample_idx]  # type: ignore
    ocean_sample_init_time = sample.ocean[1].isel(time=0).item()
    atmos_sample_init_time = sample.atmosphere[1].isel(time=0).item()

    # we already checked for matching ocean/atmos sample init times above, now
    # checking that they match times at the right positions in each dataset
    assert ocean_sample_init_time == ocean_ds.isel(time=1)["time"].item()
    assert atmos_sample_init_time == atmos_ds.isel(time=2)["time"].item()

    expected_ocean = ocean_ds["bar"].isel(time=slice(1, 3)).values
    expected_atmos = atmos_ds["foo"].isel(time=slice(2, 5)).values
    # check that
    assert np.allclose(sample.ocean[0]["bar"].cpu().numpy(), expected_ocean)
    assert np.allclose(sample.atmosphere[0]["foo"].cpu().numpy(), expected_atmos)
