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
    atmos_timestep = datetime.timedelta(days=1)

    with pytest.raises(ValueError) as err:
        CoupledDataset(ocean_ds, atmos_ds, ocean_timestep, atmos_timestep)

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
    atmos_timestep = datetime.timedelta(days=1)

    with pytest.raises(ValueError) as err:
        CoupledDataset(ocean_ds, atmos_ds, ocean_timestep, atmos_timestep)

    assert "unexpected timestep" in str(err.value)


def test_atmosphere_timestep_too_large_error(mock_coupled_data):
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
    ocean_timestep = datetime.timedelta(days=2)
    atmos_timestep = datetime.timedelta(days=1)

    with pytest.raises(ValueError) as err:
        # swap ocean and atmos args so "atmos" has larger timestep
        CoupledDataset(atmos_ds, ocean_ds, atmos_timestep, ocean_timestep)

    assert "must be no larger" in str(err.value)


def test_timestep_not_a_multiple_error(mock_coupled_data):
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
    ocean_timestep = datetime.timedelta(days=3)
    atmos_timestep = datetime.timedelta(days=2)

    # manually set the timesteps to trigger the error
    ocean_ds._timestep = ocean_timestep
    atmos_ds._timestep = atmos_timestep

    with pytest.raises(ValueError) as err:
        # swap ocean and atmos args so "atmos" has larger timestep
        CoupledDataset(ocean_ds, atmos_ds, ocean_timestep, atmos_timestep)

    assert "to be a multiple" in str(err.value)


def test_misconfigured_n_timesteps_error(mock_coupled_data):
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
            n_timesteps=2,
        ),
    )
    ocean_timestep = datetime.timedelta(days=2)
    atmos_timestep = datetime.timedelta(days=1)

    with pytest.raises(ValueError) as err:
        # swap ocean and atmos args so "atmos" has larger timestep
        CoupledDataset(ocean_ds, atmos_ds, ocean_timestep, atmos_timestep)

    assert "timepoints (including IC) per sample" in str(err.value)
