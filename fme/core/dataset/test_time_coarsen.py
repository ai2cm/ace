import pytest
import torch
import xarray as xr

from fme.core.dataset.time_coarsen import TimeCoarsenConfig, TimeCoarsenDataset
from fme.core.testing import date_range
from fme.core.typing_ import TensorDict


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[tuple[TensorDict, xr.DataArray]]):
        if len(samples) == 0:
            raise ValueError("At least one sample is required")
        if len(set(len(sample[1]) for sample in samples)) != 1:
            raise ValueError("All samples must have the same number of timesteps")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[TensorDict, xr.DataArray]:
        return self.samples[idx]

    @property
    def sample_start_times(self) -> xr.CFTimeIndex:
        values = []
        for _, time in self.samples:
            values.append(time[0].values.item())
        return xr.CFTimeIndex(values)

    @property
    def sample_n_times(self) -> int:
        return self.samples[0][1].shape[0]


def get_sample(names: list[str], n_timesteps: int) -> tuple[TensorDict, xr.DataArray]:
    n_lat = 8
    n_lon = 16
    data = {name: torch.randn(n_timesteps, n_lat, n_lon) for name in names}
    times = date_range(n_timesteps)
    return (data, times)


@pytest.mark.parametrize("n_timesteps, factor", [(16, 2), (17, 2), (8, 1)])
def test_time_coarsen_dataset(n_timesteps: int, factor: int):
    config = TimeCoarsenConfig(
        factor=factor,
        snapshot_names=["snapshot"],
        window_names=["window"],
    )
    base_sample = get_sample(["snapshot", "window"], n_timesteps)
    base_data, base_time = base_sample
    base_dataset = BasicDataset([base_sample])
    dataset = TimeCoarsenDataset(base_dataset, config)
    assert len(dataset) == len(base_dataset)
    assert dataset.sample_n_times == base_dataset.sample_n_times // config.factor
    assert dataset.sample_start_times == base_dataset.sample_start_times
    for data, time in dataset:
        assert data["snapshot"].shape[0] == dataset.sample_n_times
        assert data["window"].shape[0] == dataset.sample_n_times
        assert time.shape[0] == dataset.sample_n_times
        assert time.values[0] == base_dataset.sample_start_times[0]
        assert time.values[0] == base_time.values[0]
        assert time.values[1] == base_time.values[config.factor]
        torch.testing.assert_close(
            data["snapshot"][0],
            base_data["snapshot"][config.factor - 1],
        )
        torch.testing.assert_close(
            data["snapshot"][1],
            base_data["snapshot"][config.factor * 2 - 1],
        )
        torch.testing.assert_close(
            data["window"][0],
            torch.mean(base_data["window"][: config.factor], dim=0),
        )
        torch.testing.assert_close(
            data["window"][1],
            torch.mean(base_data["window"][config.factor : 2 * config.factor], dim=0),
        )
