import datetime
from typing import Any, cast

import cftime
import numpy as np
import pytest

from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.requirements import DataRequirements
from fme.core.dataset.xarray import XarrayDataConfig
from fme.coupled.data_loading.batch_data import CoupledBatchData, CoupledPrognosticState
from fme.coupled.data_loading.config import CoupledDatasetWithOptionalOceanConfig
from fme.coupled.data_loading.getters import get_forcing_data
from fme.coupled.data_loading.inference import (
    CoupledForcingDataLoaderConfig,
    InferenceDataLoaderConfig,
    InferenceDataset,
)
from fme.coupled.dataset_info import CoupledDatasetInfo
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


def _build_coupled_dataset_info(mock_data: MockCoupledData) -> CoupledDatasetInfo:
    """Construct a CoupledDatasetInfo directly from the mock on-disk data.

    This mimics what the stepper's training_dataset_info would provide in the
    no-target inference path (used by _make_dummy_ocean_forcing).
    """
    from fme.core.dataset_info import DatasetInfo

    ocean_info = DatasetInfo(
        horizontal_coordinates=mock_data.ocean.hcoord,
        vertical_coordinate=mock_data.ocean.vcoord,
        mask_provider=mock_data.ocean.mask_provider,
        timestep=mock_data.ocean.timestep,
    )
    atmos_info = DatasetInfo(
        horizontal_coordinates=mock_data.atmosphere.hcoord,
        vertical_coordinate=mock_data.atmosphere.vcoord,
        mask_provider=mock_data.atmosphere.mask_provider,
        timestep=mock_data.atmosphere.timestep,
    )
    return CoupledDatasetInfo(ocean=ocean_info, atmosphere=atmos_info)


def test_no_target_inference_with_n_repeats(tmp_path):
    """Test integration of n_repeats + update_subset alignment in no-target inference.

    - No ocean forcing dataset (forcing_loader.ocean is None).
    - Atmosphere forcing uses XarrayDataConfig with n_repeats > 1.
    - Ocean IC time is later than the atmosphere's first time, so
      update_subset(TimeSlice(start_time=ocean.first_time)) must shift the
      atmosphere subset.
    - The rollout extends past the end of the first repeat into a later
      repeat, exercising the integration of n_repeats with update_subset.
    """
    _OCEAN_NAME = "bar"
    _ATMOS_NAME = "foo"
    _N_FORWARD_OCEAN = 3  # 4 ocean times in source file
    _N_FORWARD_ATMOS = 12  # 13 atmos times in source file (1d step, 4d ocean step)
    _N_REPEATS = 3
    _TOTAL_COUPLED_STEPS = 8  # rollout = 8 ocean steps = 32 atmos days
    _COUPLED_STEPS_IN_MEMORY = 2  # multi-batch (matches production config)
    _N_INNER_STEPS = 4  # = ocean_timestep / atmos_timestep
    _CALENDAR = "proleptic_gregorian"

    mock_data = create_coupled_data_on_disk(
        tmp_path,
        n_forward_times_ocean=_N_FORWARD_OCEAN,
        n_forward_times_atmosphere=_N_FORWARD_ATMOS,
        ocean_names=[_OCEAN_NAME],
        atmosphere_names=[_ATMOS_NAME],
        atmosphere_start_time_offset_from_ocean=0,
    )
    # Sanity: ocean timestep is 4 days, atmos is 1 day.
    assert mock_data.ocean.timestep == datetime.timedelta(days=4)
    assert mock_data.atmosphere.timestep == datetime.timedelta(days=1)

    # IC at the second ocean time (1970-01-05): one ocean step past atmos start.
    ic_time = cftime.datetime(1970, 1, 5, calendar=_CALENDAR)
    atmos_initial_condition = BatchData.new_for_testing(
        names=[_ATMOS_NAME],
        n_samples=1,
        n_timesteps=1,
        t_initial=ic_time,
        calendar=_CALENDAR,
    )
    ocean_initial_condition = BatchData.new_for_testing(
        names=[_OCEAN_NAME],
        n_samples=1,
        n_timesteps=1,
        t_initial=ic_time,
        calendar=_CALENDAR,
    )

    # No-target forcing config: ocean is None, atmosphere has n_repeats > 1.
    config = CoupledForcingDataLoaderConfig(
        atmosphere=ForcingDataLoaderConfig(
            XarrayDataConfig(
                data_path=str(tmp_path / "atmos"),
                n_repeats=_N_REPEATS,
            ),
        ),
        ocean=None,
    )
    window_requirements = CoupledDataRequirements(
        ocean_timestep=mock_data.ocean.timestep,
        ocean_requirements=DataRequirements(
            [_OCEAN_NAME], n_timesteps=_COUPLED_STEPS_IN_MEMORY + 1
        ),
        atmosphere_timestep=mock_data.atmosphere.timestep,
        atmosphere_requirements=DataRequirements(
            [_ATMOS_NAME],
            n_timesteps=_COUPLED_STEPS_IN_MEMORY * _N_INNER_STEPS + 1,
        ),
    )

    dataset_info = _build_coupled_dataset_info(mock_data)
    initial_condition = CoupledPrognosticState(
        ocean_data=PrognosticState(ocean_initial_condition),
        atmosphere_data=PrognosticState(atmos_initial_condition),
    )

    data = get_forcing_data(
        config=config,
        total_coupled_steps=_TOTAL_COUPLED_STEPS,
        window_requirements=window_requirements,
        initial_condition=initial_condition,
        dataset_info=dataset_info,
    )

    # Initial time exposed by the loader matches the IC.
    initial_time_value = data.initial_time.values.flat[0]
    assert initial_time_value == ic_time

    # Loader yields the expected number of batches.
    batches = list(data.loader)
    expected_n_batches = _TOTAL_COUPLED_STEPS // _COUPLED_STEPS_IN_MEMORY
    assert (
        len(batches) == expected_n_batches
    ), f"Expected {expected_n_batches} batches, got {len(batches)}."

    n_atmos_per_batch = _COUPLED_STEPS_IN_MEMORY * _N_INNER_STEPS + 1
    n_ocean_per_batch = _COUPLED_STEPS_IN_MEMORY + 1
    source_n_atmos_times = _N_FORWARD_ATMOS + 1  # 13
    source_span = mock_data.atmosphere.timestep * source_n_atmos_times
    first_atmos_time_on_disk = cftime.datetime(1970, 1, 1, calendar=_CALENDAR)
    end_of_first_repeat = first_atmos_time_on_disk + source_span
    source_values = mock_data.atmosphere.ds[_ATMOS_NAME].values

    all_atmos_times_seen: set = set()
    for batch_idx, batch in enumerate(batches):
        assert isinstance(batch, CoupledBatchData)
        atmos_times = batch.atmosphere_data.time.isel(sample=0).values
        ocean_times = batch.ocean_data.time.isel(sample=0).values
        atmos_values = batch.atmosphere_data.data[_ATMOS_NAME][0].cpu().numpy()

        # Shapes match expected per-batch window length.
        assert atmos_times.shape == (n_atmos_per_batch,), (
            f"Batch {batch_idx}: atmos shape {atmos_times.shape} != "
            f"({n_atmos_per_batch},)"
        )
        assert ocean_times.shape == (n_ocean_per_batch,), (
            f"Batch {batch_idx}: ocean shape {ocean_times.shape} != "
            f"({n_ocean_per_batch},)"
        )

        # First atmos/ocean time of each batch aligns with the rollout position
        # implied by batch_idx (the alignment that update_subset is responsible
        # for).
        batch_first_time = (
            ic_time + batch_idx * _COUPLED_STEPS_IN_MEMORY * mock_data.ocean.timestep
        )
        assert atmos_times[0] == batch_first_time, (
            f"Batch {batch_idx}: first atmos time {atmos_times[0]} != "
            f"expected {batch_first_time}"
        )
        assert ocean_times[0] == batch_first_time, (
            f"Batch {batch_idx}: first ocean time {ocean_times[0]} != "
            f"expected {batch_first_time}"
        )

        # Atmos times are uniformly spaced (catches a bad seam between repeats).
        atmos_diffs = np.diff(atmos_times)
        assert np.all(atmos_diffs == mock_data.atmosphere.timestep), (
            f"Batch {batch_idx}: atmos times not uniformly spaced; "
            f"unique diffs = {set(atmos_diffs)}"
        )

        # Per-position value check: each batch value equals the source value at
        # (offset_from_source_start % source_n_atmos_times).
        for k, t in enumerate(atmos_times):
            offset_steps = (
                t - first_atmos_time_on_disk
            ) // mock_data.atmosphere.timestep
            expected_local_idx = int(offset_steps) % source_n_atmos_times
            np.testing.assert_allclose(
                atmos_values[k],
                source_values[expected_local_idx],
                rtol=1e-6,
                err_msg=(
                    f"Batch {batch_idx}, position {k} (time {t}): expected "
                    f"source index {expected_local_idx}."
                ),
            )

        all_atmos_times_seen.update(atmos_times.tolist())

    # Last atmos time of the last batch equals IC + total rollout (catches
    # off-by-one or shift introduced by update_subset).
    last_atmos_times = batches[-1].atmosphere_data.time.isel(sample=0).values
    expected_last = ic_time + _TOTAL_COUPLED_STEPS * mock_data.ocean.timestep
    assert last_atmos_times[-1] == expected_last

    # Rollout end falls into a later repeat (sanity check that n_repeats was
    # actually exercised, not silently truncated).
    assert last_atmos_times[-1] >= end_of_first_repeat, (
        "Rollout did not extend past the first repeat; n_repeats may not be "
        "exercised by this test configuration."
    )

    # Explicit wraparound value check: the first time of the second repeat
    # (1970-01-14) appears in some batch and corresponds to the source value at
    # 1970-01-01. (Redundant with 6, but states the property explicitly.)
    first_wrap_time = cftime.datetime(1970, 1, 14, calendar=_CALENDAR)
    assert first_wrap_time in all_atmos_times_seen, (
        f"First wraparound time {first_wrap_time} not seen in any batch; "
        f"this test configuration is not exercising the seam."
    )
    for batch in batches:
        atmos_times = batch.atmosphere_data.time.isel(sample=0).values
        atmos_values = batch.atmosphere_data.data[_ATMOS_NAME][0].cpu().numpy()
        wrap_positions = np.where(atmos_times == first_wrap_time)[0]
        if len(wrap_positions) == 0:
            continue
        wrap_idx = int(wrap_positions[0])
        np.testing.assert_allclose(
            atmos_values[wrap_idx],
            source_values[0],
            rtol=1e-6,
            err_msg=(
                f"Value at first wraparound time ({first_wrap_time}) does not "
                f"equal source value at {first_atmos_time_on_disk} (index 0)."
            ),
        )
