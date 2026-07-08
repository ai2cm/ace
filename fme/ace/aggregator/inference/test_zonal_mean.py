import numpy as np
import pytest
import torch

from fme.ace.aggregator.inference.data import InferenceBatchData, make_dummy_time
from fme.ace.aggregator.inference.zonal_mean import ZonalMeanAggregator
from fme.core import get_device

n_sample, n_time, ny, nx = 3, 6, 10, 20
zonal_mean_max_size = 2**12


def zonal_mean(data: torch.Tensor) -> torch.Tensor:
    return data.mean(dim=3)


def _make_batch_data(target_data, gen_data, i_time_start=0) -> InferenceBatchData:
    sample = next(iter(gen_data.values()))
    return InferenceBatchData(
        prediction=gen_data,
        prediction_norm={},
        target=target_data,
        target_norm=None,
        time=make_dummy_time(sample.shape[0], sample.shape[1]),
        i_time_start=i_time_start,
    )


def test_zonal_mean_dims():
    agg = ZonalMeanAggregator(
        zonal_mean,
        n_timesteps=n_time,
        zonal_mean_max_size=zonal_mean_max_size,
    )
    target_data = {"a": torch.randn(n_sample, n_time, ny, nx, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, ny, nx, device=get_device())}
    agg.record_batch(_make_batch_data(target_data, gen_data, i_time_start=0))
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        assert data["a"].size() == (
            n_sample,
            n_time,
            ny,
        )


def test_zonal_mean_lat_varying():
    agg = ZonalMeanAggregator(
        zonal_mean,
        n_timesteps=n_time,
        zonal_mean_max_size=zonal_mean_max_size,
    )
    arr = torch.arange(ny, dtype=torch.float32, device=get_device())
    arr = arr[None, None, :, None].expand(n_sample, n_time, -1, nx)
    agg.record_batch(_make_batch_data({"a": arr}, {"a": arr}, i_time_start=0))
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        torch.testing.assert_close(
            data["a"][0, 0, :],  # one time row of the zonal mean
            torch.arange(ny, dtype=torch.float32, device=get_device()),
        )


def test_zonal_mean_zonally_varying():
    agg = ZonalMeanAggregator(
        zonal_mean,
        n_timesteps=n_time,
        zonal_mean_max_size=zonal_mean_max_size,
    )
    arr = torch.arange(nx, dtype=torch.float32, device=get_device())
    arr = arr[None, None, None, :].expand(n_sample, n_time, ny, -1)
    agg.record_batch(_make_batch_data({"a": arr}, {"a": arr}, i_time_start=0))
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        torch.testing.assert_close(
            data["a"][0, 0, :],  # one time row of the zonal mean
            arr.mean() * torch.ones(ny, dtype=torch.float32, device=get_device()),
        )


def test_zonal_mean_batch_varying():
    agg = ZonalMeanAggregator(
        zonal_mean,
        n_timesteps=n_time,
        zonal_mean_max_size=zonal_mean_max_size,
    )
    for i in range(n_sample):  # assume one sample per batch
        arr = torch.tensor(i, dtype=torch.float32, device=get_device())
        arr = arr[None, None, None, None].expand(-1, n_time, ny, nx)
        agg.record_batch(_make_batch_data({"a": arr}, {"a": arr}, i_time_start=0))
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        torch.testing.assert_close(
            data["a"].sum(dim=0)[0, 0],  # sum over batches, then pick a time/lat point
            torch.arange(n_sample, dtype=torch.float32, device=get_device()).sum(),
            # should be same as sum over batches
        )


def test_zonal_mean_mulitple_time_slices():
    n_time_windows = 2
    n_time_in_memory = n_time // n_time_windows
    agg = ZonalMeanAggregator(
        zonal_mean,
        n_timesteps=n_time,
        zonal_mean_max_size=zonal_mean_max_size,
    )
    for i_time in range(0, n_time, n_time_in_memory):
        arr = torch.arange(ny, dtype=torch.float32, device=get_device())
        arr = arr[None, None, :, None].expand(n_sample, n_time_in_memory, ny, nx)
        agg.record_batch(_make_batch_data({"a": arr}, {"a": arr}, i_time_start=i_time))
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        torch.testing.assert_close(
            (data["a"] / agg._n_batches)[0, 0, :],
            torch.arange(ny, dtype=torch.float32, device=get_device()),
        )


@pytest.mark.parametrize("n_time", [40, 2**12 + 1, 2**13 + 7, 118260])
def test_zonal_mean_time_coarsening(n_time):
    n_sample, ny, nx = 3, 10, 20
    n_time_in_memory = 40

    agg = ZonalMeanAggregator(
        zonal_mean,
        n_timesteps=n_time,
        zonal_mean_max_size=2**12,
    )

    assert (
        agg.time_coarsening_factor == np.ceil(n_time / agg._max_size).astype(int)
        if n_time > agg._max_size
        else 1
    )

    assert n_time / agg.time_coarsening_factor <= agg._max_size

    chunks = np.ceil(n_time / n_time_in_memory).astype(int)
    for chunck in range(chunks):
        # This is how the inference loop acts
        remainder = n_time_in_memory
        if chunck * n_time_in_memory + n_time_in_memory > n_time:
            remainder = n_time - chunck * n_time_in_memory

        arr = torch.arange(ny, dtype=torch.float32, device=get_device())
        arr = arr[None, None, :, None].expand(n_sample, remainder, ny, nx)

        arr_t = torch.arange(
            start=chunck * n_time_in_memory,
            end=chunck * n_time_in_memory + remainder,
            dtype=torch.float32,
            device=get_device(),
        )
        arr_t = arr_t[None, :, None, None].expand(n_sample, remainder, ny, nx)
        arr = arr + arr_t  # make sure the zonal mean is not constant in time

        agg.record_batch(
            _make_batch_data(
                {"a": arr}, {"a": arr}, i_time_start=chunck * n_time_in_memory
            )
        )
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        # check that the time dim shape is a factor of time_coarsening_factor smaller
        assert data["a"].size() == (
            n_sample,
            n_time // agg.time_coarsening_factor,
            ny,
        )
        # check that the zonal mean time coarsening is correct
        expected_value = torch.stack(
            [
                torch.arange(ny, dtype=torch.float32, device=get_device()),
                torch.arange(
                    start=agg.time_coarsening_factor - 1,
                    end=ny + agg.time_coarsening_factor - 1,
                    dtype=torch.float32,
                    device=get_device(),
                ),
            ]
        ).mean(dim=0)
        torch.testing.assert_close(
            (data["a"] / agg._n_batches)[0, 0, :], expected_value
        )


@pytest.mark.parametrize("n_ic_steps", [1, 2])
def test_zonal_mean_time_coarsening_initial_condition_offset(n_ic_steps):
    """During inference the initial condition advances the global timestep
    counter by ``n_ic_steps`` before any window is recorded to this aggregator.
    The first recorded window therefore starts at a global time index that is
    not necessarily a multiple of the coarsening factor. This previously caused
    a size mismatch between the destination slice and the coarsened source.
    """
    n_sample, ny, nx = 2, 10, 20
    n_time_in_memory = 20
    n_time = 8761  # -> coarsening factor of 3 with a max size of 4096

    agg = ZonalMeanAggregator(
        zonal_mean,
        n_timesteps=n_time,
        zonal_mean_max_size=2**12,
    )
    factor = agg.time_coarsening_factor
    assert factor == 3

    n_forward = n_time - n_ic_steps
    for i_time in range(n_ic_steps, n_ic_steps + n_forward, n_time_in_memory):
        steps = min(n_time_in_memory, n_ic_steps + n_forward - i_time)
        # zonal mean of forward step k (0-indexed from the first recorded step)
        # is set equal to k, constant in latitude.
        forward_index = torch.arange(
            start=i_time - n_ic_steps,
            end=i_time - n_ic_steps + steps,
            dtype=torch.float32,
            device=get_device(),
        )
        arr = forward_index[None, :, None, None].expand(n_sample, steps, ny, nx)
        agg.record_batch(_make_batch_data({"a": arr}, {"a": arr}, i_time_start=i_time))

    # number of coarse columns actually filled by the recorded forward steps
    n_filled = n_forward // factor
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        assert data["a"].shape == (n_sample, n_time // factor, ny)
        coarsened = (data["a"] / agg._n_batches)[0, :n_filled, 0]
        # group g averages forward steps {factor*g, ..., factor*g + factor - 1}
        expected = (
            torch.arange(n_filled, dtype=torch.float32, device=get_device()) * factor
            + (factor - 1) / 2
        )
        torch.testing.assert_close(coarsened, expected)


@pytest.mark.parametrize("zonal_mean_max_size", [4, 2**14, 2**16])
def test_zonal_mean_time_coarsening_override(zonal_mean_max_size):
    n_time = 2**16
    agg = ZonalMeanAggregator(
        zonal_mean,
        n_timesteps=n_time,
        zonal_mean_max_size=zonal_mean_max_size,
    )
    assert agg._max_size <= agg._max_matplotlib_size
