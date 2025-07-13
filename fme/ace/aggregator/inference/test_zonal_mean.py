import torch

from fme.ace.aggregator.inference.zonal_mean import ZonalMeanAggregator
from fme.core import get_device

n_sample, n_time, ny, nx = 3, 6, 10, 20


def zonal_mean(data: torch.Tensor) -> torch.Tensor:
    return data.mean(dim=3)


def test_zonal_mean_dims():
    agg = ZonalMeanAggregator(zonal_mean, n_timesteps=n_time)
    target_data = {"a": torch.randn(n_sample, n_time, ny, nx, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, ny, nx, device=get_device())}
    agg.record_batch(target_data, gen_data, target_data, gen_data, i_time_start=0)
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        assert data["a"].size() == (
            n_sample,
            n_time,
            ny,
        )


def test_zonal_mean_lat_varying():
    agg = ZonalMeanAggregator(zonal_mean, n_timesteps=n_time)
    arr = torch.arange(ny, dtype=torch.float32, device=get_device())
    arr = arr[None, None, :, None].expand(n_sample, n_time, -1, nx)
    agg.record_batch({"a": arr}, {"a": arr}, {"a": arr}, {"a": arr}, i_time_start=0)
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        torch.testing.assert_close(
            data["a"][0, 0, :],  # one time row of the zonal mean
            torch.arange(ny, dtype=torch.float32, device=get_device()),
        )


def test_zonal_mean_zonally_varying():
    agg = ZonalMeanAggregator(zonal_mean, n_timesteps=n_time)
    arr = torch.arange(nx, dtype=torch.float32, device=get_device())
    arr = arr[None, None, None, :].expand(n_sample, n_time, ny, -1)
    agg.record_batch({"a": arr}, {"a": arr}, {"a": arr}, {"a": arr}, i_time_start=0)
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        torch.testing.assert_close(
            data["a"][0, 0, :],  # one time row of the zonal mean
            arr.mean() * torch.ones(ny, dtype=torch.float32, device=get_device()),
        )


def test_zonal_mean_batch_varying():
    agg = ZonalMeanAggregator(zonal_mean, n_timesteps=n_time)
    for i in range(n_sample):  # assume one sample per batch
        arr = torch.tensor(i, dtype=torch.float32, device=get_device())
        arr = arr[None, None, None, None].expand(-1, n_time, ny, nx)
        agg.record_batch({"a": arr}, {"a": arr}, {"a": arr}, {"a": arr}, i_time_start=0)
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
    agg = ZonalMeanAggregator(zonal_mean, n_timesteps=n_time)
    for i_time in range(0, n_time, n_time_in_memory):
        arr = torch.arange(ny, dtype=torch.float32, device=get_device())
        arr = arr[None, None, :, None].expand(n_sample, n_time_in_memory, ny, nx)
        agg.record_batch(
            {"a": arr}, {"a": arr}, {"a": arr}, {"a": arr}, i_time_start=i_time
        )
    for data in (agg._target_data, agg._gen_data):
        assert data is not None
        torch.testing.assert_close(
            (data["a"] / agg._n_batches)[0, 0, :],
            torch.arange(ny, dtype=torch.float32, device=get_device()),
        )
