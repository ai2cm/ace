import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PairedData
from fme.core.device import get_device


def assert_metadata_equal(a: BatchData, b: BatchData):
    assert a.horizontal_dims == b.horizontal_dims


def get_batch_data(
    names: list[str],
    n_samples: int,
    n_times: int,
    horizontal_dims: list[str],
    n_lat: int = 8,
    n_lon: int = 16,
):
    device = get_device()
    return BatchData(
        data={
            name: torch.randn(n_samples, n_times, n_lat, n_lon, device=device)
            for name in names
        },
        time=xr.DataArray(np.random.rand(n_samples, n_times), dims=["sample", "time"]),
        horizontal_dims=horizontal_dims,
    )


@pytest.mark.parametrize(
    "names, prognostic_names",
    [
        pytest.param(["prog"], ["prog"], id="all prognostic"),
        pytest.param(["prog", "forcing"], ["prog"], id="some prognostic"),
        pytest.param(["forcing1", "forcing2"], [], id="no prognostic"),
    ],
)
@pytest.mark.parametrize("n_ic_timesteps", [1, 2])
def test_get_start(names: list[str], prognostic_names: list[str], n_ic_timesteps: int):
    n_samples = 2
    n_times = 5
    n_lat = 8
    n_lon = 16
    horizontal_dims = ["lat", "lon"]
    batch_data = get_batch_data(
        names=names,
        n_samples=n_samples,
        n_times=n_times,
        horizontal_dims=horizontal_dims,
        n_lat=n_lat,
        n_lon=n_lon,
    )
    start = batch_data.get_start(prognostic_names, n_ic_timesteps).as_batch_data()
    assert_metadata_equal(start, batch_data)
    assert start.time.equals(batch_data.time.isel(time=slice(0, n_ic_timesteps)))
    assert set(start.data.keys()) == set(prognostic_names)
    for name in prognostic_names:
        assert start.data[name].shape == (n_samples, n_ic_timesteps, n_lat, n_lon)
        np.testing.assert_allclose(
            start.data[name].cpu().numpy(),
            batch_data.data[name][:, :n_ic_timesteps, ...].cpu().numpy(),
        )


@pytest.mark.parametrize(
    "names, prognostic_names",
    [
        pytest.param(["prog"], ["prog"], id="all prognostic"),
        pytest.param(["prog", "forcing"], ["prog"], id="some prognostic"),
        pytest.param(["forcing1", "forcing2"], [], id="no prognostic"),
    ],
)
@pytest.mark.parametrize("n_ic_timesteps", [1, 2])
def test_get_end(names: list[str], prognostic_names: list[str], n_ic_timesteps: int):
    n_samples = 2
    n_times = 5
    n_lat = 8
    n_lon = 16
    horizontal_dims = ["lat", "lon"]
    batch_data = get_batch_data(
        names=names,
        n_samples=n_samples,
        n_times=n_times,
        horizontal_dims=horizontal_dims,
        n_lat=n_lat,
        n_lon=n_lon,
    )
    end = batch_data.get_end(prognostic_names, n_ic_timesteps).as_batch_data()
    assert_metadata_equal(end, batch_data)
    assert end.time.equals(batch_data.time.isel(time=slice(-n_ic_timesteps, None)))
    assert set(end.data.keys()) == set(prognostic_names)
    for name in prognostic_names:
        assert end.data[name].shape == (n_samples, n_ic_timesteps, n_lat, n_lon)
        np.testing.assert_allclose(
            end.data[name].cpu().numpy(),
            batch_data.data[name][:, -n_ic_timesteps:, ...].cpu().numpy(),
        )


@pytest.mark.parametrize(
    "names, prepend_names",
    [
        pytest.param(["prog"], ["prog"], id="all prepended"),
        pytest.param(["prog", "forcing"], ["prog"], id="some prepended"),
    ],
)
@pytest.mark.parametrize("n_ic_timesteps", [1, 2])
def test_prepend(names: list[str], prepend_names: list[str], n_ic_timesteps: int):
    n_samples = 2
    n_times = 5
    n_lat = 8
    n_lon = 16
    horizontal_dims = ["lat", "lon"]
    batch_data = get_batch_data(
        names=names,
        n_samples=n_samples,
        n_times=n_times,
        horizontal_dims=horizontal_dims,
        n_lat=n_lat,
        n_lon=n_lon,
    )
    start_data = batch_data.get_start(prepend_names, n_ic_timesteps)
    prepended = batch_data.prepend(start_data)
    start_batch_data = start_data.as_batch_data()
    assert_metadata_equal(prepended, batch_data)
    assert prepended.time.isel(time=slice(n_ic_timesteps, None)).equals(batch_data.time)
    assert set(prepended.data.keys()) == set(names)
    for name in names:
        np.testing.assert_allclose(
            prepended.data[name][:, n_ic_timesteps:, ...].cpu().numpy(),
            batch_data.data[name].cpu().numpy(),
        )
    for name in prepend_names:
        np.testing.assert_allclose(
            prepended.data[name][:, :n_ic_timesteps, ...].cpu().numpy(),
            start_batch_data.data[name].cpu().numpy(),
        )
    assert prepended.time.shape == (n_samples, n_times + n_ic_timesteps)
    for name in set(names) - set(prepend_names):
        assert np.all(
            np.isnan(prepended.data[name][:, :n_ic_timesteps, ...].cpu().numpy())
        )


def test_paired_data_forcing_target_data():
    n_samples = 2
    n_times = 5
    n_lat = 8
    n_lon = 16
    horizontal_dims = ["lat", "lon"]
    target_data = get_batch_data(
        names=["foo", "bar"],
        n_samples=n_samples,
        n_times=n_times,
        horizontal_dims=horizontal_dims,
        n_lat=n_lat,
        n_lon=n_lon,
    )
    gen_data = get_batch_data(
        names=["bar"],
        n_samples=n_samples,
        n_times=n_times,
        horizontal_dims=horizontal_dims,
        n_lat=n_lat,
        n_lon=n_lon,
    )
    gen_data.time = target_data.time
    paired_data = PairedData.from_batch_data(prediction=gen_data, reference=target_data)
    assert paired_data.forcing == {"foo": target_data.data["foo"]}
    assert paired_data.target == {"bar": target_data.data["bar"]}
