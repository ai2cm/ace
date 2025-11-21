from typing import Any, cast

import numpy as np
import pytest

from fme.ace.data_loading.inference import (
    ExplicitIndices,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.requirements import DataRequirements
from fme.coupled.data_loading.batch_data import CoupledBatchData
from fme.coupled.data_loading.config import CoupledDatasetWithOptionalOceanConfig
from fme.coupled.data_loading.inference import (
    InferenceDataLoaderConfig,
    InferenceDataset,
)
from fme.coupled.requirements import CoupledDataRequirements

from .test_data_loader import MockCoupledData, create_coupled_data_on_disk


def _setup(
    mock_data: MockCoupledData,
    total_coupled_steps: int,
    start_indices: InferenceInitialConditionIndices | ExplicitIndices | TimestampList,
    ocean_data_config_kwargs: dict[str, Any] | None = None,
    atmos_data_config_kwargs: dict[str, Any] | None = None,
) -> InferenceDataset:
    dataset_config = mock_data.get_dataset_config_with_kwargs(
        ocean_kwargs=ocean_data_config_kwargs,
        atmos_kwargs=atmos_data_config_kwargs,
    )
    # ocean timesteps include initial condition plus total_coupled_steps
    ocean_n_timesteps = total_coupled_steps + 1  # one forward window in memory
    n_inner_steps = int(mock_data.ocean.timestep / mock_data.atmosphere.timestep)
    atmos_n_timesteps = n_inner_steps * total_coupled_steps + 1
    ocean_names = list(mock_data.ocean.ds.data_vars)
    atmos_names = list(mock_data.atmosphere.ds.data_vars)
    coupled_requirements = CoupledDataRequirements(
        ocean_timestep=mock_data.ocean.timestep,
        ocean_requirements=DataRequirements(ocean_names, n_timesteps=ocean_n_timesteps),
        atmosphere_timestep=mock_data.atmosphere.timestep,
        atmosphere_requirements=DataRequirements(
            atmos_names, n_timesteps=atmos_n_timesteps
        ),
    )
    config = InferenceDataLoaderConfig(
        dataset=cast(CoupledDatasetWithOptionalOceanConfig, dataset_config),
        start_indices=start_indices,
    )
    dataset = InferenceDataset(
        config=config,
        total_coupled_steps=total_coupled_steps,
        requirements=coupled_requirements,
    )
    return dataset


_N_ICS = 2  # number of initial conditions
_IC_INDICES = [0, 2]
# these three are equivalent:
_EXPLICIT_INDICES = ExplicitIndices(list=_IC_INDICES)
_TIMESTAMPS = TimestampList(times=["1970-01-01T00:00:00", "1970-01-07T00:00:00"])
_IC_RANGE = InferenceInitialConditionIndices(
    n_initial_conditions=2, first=0, interval=2
)


@pytest.mark.parametrize("start_indices", [_EXPLICIT_INDICES, _TIMESTAMPS, _IC_RANGE])
@pytest.mark.parametrize("atmos_ic_time_offset", [0, 3, 2])
def test_inference_dataset(
    tmp_path,
    start_indices: InferenceInitialConditionIndices | ExplicitIndices | TimestampList,
    atmos_ic_time_offset: int,
):
    # synthetic data settings
    _N_FORWARD_OCEAN = 5
    _N_ATMOS_PER_OCEAN = 3
    _N_FORWARD_ATMOS = _N_FORWARD_OCEAN * _N_ATMOS_PER_OCEAN
    _OCEAN_NAME = "bar"
    _ATMOS_NAME = "foo"

    _N_STEPS = 2  # number of inference rollout steps

    def _check_batch(
        batch: CoupledBatchData,
        mock_data: MockCoupledData,
        atmos_ic_time_offset: int,
    ):
        ocean_data = batch.ocean_data.data[_OCEAN_NAME].cpu().numpy()
        atmos_data = batch.atmosphere_data.data[_ATMOS_NAME].cpu().numpy()
        # verify batch dimension
        assert ocean_data.shape[0] == _N_ICS
        assert atmos_data.shape[0] == _N_ICS
        # verify time dimension
        n_ocean_times = _N_STEPS + 1
        n_atmos_times = _N_STEPS * _N_ATMOS_PER_OCEAN + 1
        assert ocean_data.shape[1] == n_ocean_times
        assert atmos_data.shape[1] == n_atmos_times
        # verify tensor values
        ocean_idx_0 = _IC_INDICES[0]
        expected_ocean_0 = (
            mock_data.ocean.ds[_OCEAN_NAME]
            .isel(time=slice(ocean_idx_0, ocean_idx_0 + n_ocean_times))
            .values
        )
        atmos_idx_0 = atmos_ic_time_offset
        expected_atmos_0 = (
            mock_data.atmosphere.ds[_ATMOS_NAME]
            .isel(time=slice(atmos_idx_0, atmos_idx_0 + n_atmos_times))
            .values
        )
        ocean_idx_1 = _IC_INDICES[1]
        expected_ocean_1 = (
            mock_data.ocean.ds[_OCEAN_NAME]
            .isel(time=slice(ocean_idx_1, ocean_idx_1 + n_ocean_times))
            .values
        )
        atmos_idx_1 = atmos_ic_time_offset + ocean_idx_1 * n_ocean_times
        expected_atmos_1 = (
            mock_data.atmosphere.ds[_ATMOS_NAME]
            .isel(time=slice(atmos_idx_1, atmos_idx_1 + n_atmos_times))
            .values
        )
        np.testing.assert_allclose(ocean_data[0], expected_ocean_0, rtol=1e-6)
        np.testing.assert_allclose(ocean_data[1], expected_ocean_1, rtol=1e-6)
        np.testing.assert_allclose(atmos_data[0], expected_atmos_0, rtol=1e-6)
        np.testing.assert_allclose(atmos_data[1], expected_atmos_1, rtol=1e-6)

    # begin the actual test

    mock_data = create_coupled_data_on_disk(
        tmp_path,
        n_forward_times_ocean=_N_FORWARD_OCEAN,
        n_forward_times_atmosphere=_N_FORWARD_ATMOS,
        ocean_names=[_OCEAN_NAME],
        atmosphere_names=[_ATMOS_NAME],
        atmosphere_start_time_offset_from_ocean=atmos_ic_time_offset,
    )

    dataset = _setup(
        mock_data,
        total_coupled_steps=_N_STEPS,
        start_indices=start_indices,
    )
    _check_batch(dataset[0], mock_data, atmos_ic_time_offset)


@pytest.mark.parametrize(
    "start_indices,err_msg",
    [
        (_TIMESTAMPS, "were not found in the time index"),
        (_EXPLICIT_INDICES, "ocean dataset has an insufficient number of timepoints"),
        (_IC_RANGE, "ocean dataset has an insufficient number of timepoints"),
    ],
)
def test_validate_inference_length_ocean(
    tmp_path,
    start_indices: InferenceInitialConditionIndices | ExplicitIndices | TimestampList,
    err_msg: str,
):
    # ocean has too few steps
    _N_FORWARD_OCEAN = 5
    _N_ATMOS_PER_OCEAN = 3
    _N_FORWARD_ATMOS = _N_FORWARD_OCEAN * _N_ATMOS_PER_OCEAN
    _OCEAN_NAME = "bar"
    _ATMOS_NAME = "foo"
    mock_data = create_coupled_data_on_disk(
        tmp_path,
        n_forward_times_ocean=_N_FORWARD_OCEAN,
        n_forward_times_atmosphere=_N_FORWARD_ATMOS,
        ocean_names=[_OCEAN_NAME],
        atmosphere_names=[_ATMOS_NAME],
        atmosphere_start_time_offset_from_ocean=2,
    )

    _N_STEPS = 4

    with pytest.raises(ValueError, match=rf".*{err_msg}.*"):
        _ = _setup(
            mock_data,
            total_coupled_steps=_N_STEPS,
            start_indices=start_indices,
        )


@pytest.mark.parametrize(
    "start_indices",
    [_TIMESTAMPS, _EXPLICIT_INDICES, _IC_RANGE],
)
def test_validate_inference_length_atmos(
    tmp_path,
    start_indices: InferenceInitialConditionIndices | ExplicitIndices | TimestampList,
):
    # atmos has too few steps
    _N_FORWARD_ATMOS = 6
    _N_FORWARD_OCEAN = 6
    _OCEAN_TIMESTEP_SIZE = 2  # days
    _OCEAN_NAME = "bar"
    _ATMOS_NAME = "foo"
    mock_data = create_coupled_data_on_disk(
        tmp_path,
        n_forward_times_ocean=_N_FORWARD_OCEAN,
        n_forward_times_atmosphere=_N_FORWARD_ATMOS,
        ocean_names=[_OCEAN_NAME],
        atmosphere_names=[_ATMOS_NAME],
        atmosphere_start_time_offset_from_ocean=0,
        ocean_timestep_size_in_days=_OCEAN_TIMESTEP_SIZE,
    )

    _N_STEPS = 2
    _MSG = "atmosphere dataset has an insufficient number of timepoints"

    with pytest.raises(ValueError, match=rf".*{_MSG}.*"):
        _ = _setup(
            mock_data,
            total_coupled_steps=_N_STEPS,
            start_indices=start_indices,
        )
