from collections.abc import Sequence
from datetime import timedelta

import cftime
import pytest
import torch
import xarray as xr

from .time_coarsen import coarsen_batch

DIM_SIZES = (2, 4, 2, 2)  # n_samples_in_batch, n_timesteps_in_window, n_lat, n_lon
VARNAME = "foo"


def get_windowed_batch(dim_sizes: Sequence[int], start_time: Sequence[int]):
    n_samples, n_timesteps, n_lat, n_lon = dim_sizes
    # get data varying only along time dim
    data = (
        torch.arange(n_timesteps, dtype=torch.float64)
        .repeat(n_samples, n_lat, n_lon, 1)
        .movedim(3, 1)
    )
    target = {VARNAME: data}
    time = get_batch_time(n_timesteps, start_time, n_samples=n_samples)
    return target, time


def get_batch_time(
    n_timesteps: int,
    start_time: Sequence[int],
    n_samples: int = 2,
    freq_hrs: int = 6,
):
    sample_times = [
        _get_time_array(start_time, n_timesteps, start_n_offset, freq_hrs)
        for start_n_offset in range(n_samples)
    ]
    return xr.DataArray(
        data=xr.concat(sample_times, dim="sample"),
        dims=["sample", "time"],
    )


def _get_time_array(
    start_time: Sequence[int], n_timesteps: int, start_n_offset: int, freq_hrs: int
):
    # start each sample one timestep behind the last
    start_time = cftime.DatetimeJulian(*start_time) + timedelta(
        hours=6 * start_n_offset
    )
    time_index = xr.date_range(
        start=start_time,
        periods=n_timesteps,
        freq=f"{freq_hrs}h",
        calendar="julian",
        use_cftime=True,
    )
    return xr.DataArray(data=time_index, dims=["time"]).drop_vars(["time"])


@pytest.mark.parametrize(
    [
        "coarsen_factor",
        "start_timestep",
        "expected_coarsened_data",
        "expected_coarsened_time",
        "expected_coarsened_start_timestep",
    ],
    [
        pytest.param(
            1,
            0,
            [0.0, 1.0, 2.0, 3.0],
            get_batch_time(start_time=(2020, 1, 1, 0, 0, 0), n_timesteps=4),
            0,
            id="coarsen_factor_1",
        ),
        pytest.param(
            2,
            0,
            [0.5, 2.5],
            get_batch_time(
                start_time=(2020, 1, 1, 3, 0, 0), n_timesteps=2, freq_hrs=12
            ),
            0,
            id="coarsen_factor_2",
        ),
        pytest.param(
            2,
            3,
            [0.5, 2.5],
            get_batch_time(
                start_time=(2020, 1, 1, 3, 0, 0), n_timesteps=2, freq_hrs=12
            ),
            1,
            id="coarsen_factor_2_start_timestep_3",
        ),
        pytest.param(
            4,
            0,
            [1.5],
            get_batch_time(
                start_time=(2020, 1, 1, 9, 0, 0), n_timesteps=1, freq_hrs=24
            ),
            0,
            id="coarsen_factor_4",
        ),
    ],
)
def test_time_coarsen(
    coarsen_factor: int,
    start_timestep: int,
    expected_coarsened_data: Sequence[float],
    expected_coarsened_time: Sequence[cftime.DatetimeJulian],
    expected_coarsened_start_timestep: int,
    dim_sizes: Sequence[int] = DIM_SIZES,
):
    target, time = get_windowed_batch(
        dim_sizes=dim_sizes, start_time=(2020, 1, 1, 0, 0, 0)
    )
    (
        target_coarsened,
        coarsened_start_timestep,
        time_coarsened,
    ) = coarsen_batch(
        data=target,
        start_timestep=start_timestep,
        batch_time=time,
        coarsen_factor=coarsen_factor,
    )
    # check the coarsened data time dim size
    assert target_coarsened[VARNAME].size(dim=1) == len(
        expected_coarsened_data
    ), "target coarsened time dim"
    assert time_coarsened.sizes["time"] == len(
        expected_coarsened_data
    ), "time coarsened time dim"
    # check the coarsened data values
    n_samples, _, n_lat, n_lon = dim_sizes
    (
        torch.testing.assert_close(
            target_coarsened[VARNAME],
            torch.tensor(expected_coarsened_data, dtype=torch.float64)
            .repeat(n_samples, n_lat, n_lon, 1)
            .movedim(3, 1),
        ),
        "target coarsened value",
    )
    # check the coarsened start timestep
    assert coarsened_start_timestep == expected_coarsened_start_timestep
    # check the coarsened data time coordinate values
    (
        xr.testing.assert_allclose(time_coarsened, expected_coarsened_time),
        "time initial condition value",
    )
