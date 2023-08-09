from fme.core.aggregator.inference.zonal_mean import ZonalMeanAggregator

import torch

n_sample, n_time, ny, nx = 2, 5, 10, 20
loss = 1.0


def test_zonal_mean_dims():
    agg = ZonalMeanAggregator(n_timesteps=n_time)
    target_data = {"a": torch.randn(n_sample, n_time, ny, nx)}
    gen_data = {"a": torch.randn(n_sample, n_time, ny, nx)}
    agg.record_batch(loss, target_data, gen_data, target_data, gen_data, i_time_start=0)
    for data in (agg._target_data, agg._gen_data):
        assert data["a"].size() == (
            n_sample,
            n_time,
            ny,
        )


def test_zonal_mean_lat_varying():
    agg = ZonalMeanAggregator(n_timesteps=n_time)
    arr = torch.arange(ny, dtype=torch.float32)
    arr = arr[None, None, :, None].expand(n_sample, n_time, -1, nx)
    agg.record_batch(
        loss, {"a": arr}, {"a": arr}, {"a": arr}, {"a": arr}, i_time_start=0
    )
    for data in (agg._target_data, agg._gen_data):
        torch.testing.assert_close(
            data["a"][0, 0, :],  # one time row of the zonal mean
            torch.arange(ny, dtype=torch.float32),
        )


def test_zonal_mean_zonally_varying():
    agg = ZonalMeanAggregator(n_timesteps=n_time)
    arr = torch.arange(nx, dtype=torch.float32)
    arr = arr[None, None, None, :].expand(n_sample, n_time, ny, -1)
    agg.record_batch(
        loss, {"a": arr}, {"a": arr}, {"a": arr}, {"a": arr}, i_time_start=0
    )
    for data in (agg._target_data, agg._gen_data):
        torch.testing.assert_close(
            data["a"][0, 0, :],  # one time row of the zonal mean
            arr.mean() * torch.ones(ny, dtype=torch.float32),
        )
