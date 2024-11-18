import datetime

import pytest

from fme.core.data_loading._xarray import XarrayDataset
from fme.core.data_loading.config import XarrayDataConfig
from fme.core.data_loading.requirements import DataRequirements

from .data_typing import CoupledDataset


def test_infer_timestep_error(mock_coupled_data):
    ocean_ds = XarrayDataset(
        XarrayDataConfig(data_path=mock_coupled_data.ocean_dir),
        DataRequirements(
            names=list(mock_coupled_data.ocean.data_vars),
            n_timesteps=2,
        ),
    )

    atmos_ds = XarrayDataset(
        XarrayDataConfig(
            data_path=mock_coupled_data.atmosphere_dir,
            infer_timestep=False,
        ),
        DataRequirements(
            names=list(mock_coupled_data.atmosphere.data_vars),
            n_timesteps=3,
        ),
    )
    ocean_timestep = datetime.timedelta(days=2)
    n_steps_fast = 2

    with pytest.raises(ValueError) as err:
        CoupledDataset(ocean_ds, atmos_ds, ocean_timestep, n_steps_fast)

    assert "must be inferred" in str(err.value)


def test_timestep_mismatch_error(mock_coupled_data):
    ocean_ds = XarrayDataset(
        XarrayDataConfig(data_path=mock_coupled_data.ocean_dir),
        DataRequirements(
            names=list(mock_coupled_data.ocean.data_vars),
            n_timesteps=2,
        ),
    )

    atmos_ds = XarrayDataset(
        XarrayDataConfig(
            data_path=mock_coupled_data.atmosphere_dir,
        ),
        DataRequirements(
            names=list(mock_coupled_data.atmosphere.data_vars),
            n_timesteps=3,
        ),
    )
    ocean_timestep = datetime.timedelta(days=1)
    n_steps_fast = 1

    with pytest.raises(ValueError) as err:
        CoupledDataset(ocean_ds, atmos_ds, ocean_timestep, n_steps_fast)

    assert "unexpected timestep" in str(err.value)
