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
        labels=[set() for _ in range(n_samples)],
    )


def test_repeat_interleave_batch_dim_gives_correct_labels():
    batch = BatchData.new_for_testing(
        ["value"],
        n_samples=2,
        n_timesteps=3,
        labels=[{"0"}, {"1"}],
    )
    batch.data["value"][0] = 0.0
    batch.data["value"][1] = 1.0
    repeated = batch.repeat_interleave_batch_dim(2)
    assert repeated.labels == [{"0"}, {"0"}, {"1"}, {"1"}]
    torch.testing.assert_close(
        repeated.data["value"],
        torch.tensor([0.0, 0.0, 1.0, 1.0])[:, None, None, None].broadcast_to(
            4, 3, 9, 18
        ),
    )
    assert repeated.time.shape == (4, 3)


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


@pytest.mark.parametrize("n_ic_timesteps", [1, 2])
def test_remove_initial_condition(n_ic_timesteps: int):
    names = ["foo", "bar"]
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
    removed = batch_data.remove_initial_condition(n_ic_timesteps)
    assert_metadata_equal(removed, batch_data)
    assert removed.time.equals(batch_data.time.isel(time=slice(n_ic_timesteps, None)))
    assert set(removed.data.keys()) == set(names)
    for name in names:
        assert removed.data[name].shape == (
            n_samples,
            n_times - n_ic_timesteps,
            n_lat,
            n_lon,
        )
        np.testing.assert_allclose(
            removed.data[name].cpu().numpy(),
            batch_data.data[name][:, n_ic_timesteps:, ...].cpu().numpy(),
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


@pytest.mark.parametrize("n_ensemble", [1, 2, 3])
def test_broadcast_ensemble(n_ensemble):
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

    ensemble_target_data = target_data.broadcast_ensemble(n_ensemble=1)
    ensemble_gen_data = gen_data.broadcast_ensemble(n_ensemble=n_ensemble)

    # make sure original data is unchanged and that broadcasting with an_ensemble=1 just
    # copies the BatchData object
    assert target_data.data["bar"].shape == (n_samples, n_times, n_lat, n_lon)
    assert ensemble_target_data.data["bar"].shape == (n_samples, n_times, n_lat, n_lon)

    torch.testing.assert_close(
        target_data.data["bar"],
        ensemble_target_data.data["bar"],
    )

    # assert data is shapes are correct after ensembling
    assert ensemble_gen_data.data["bar"].shape == (
        n_ensemble * n_samples,
        n_times,
        n_lat,
        n_lon,
    )

    # assert metadata is shapes are correct after ensembling
    assert len(ensemble_gen_data.labels) == n_ensemble * n_samples
    assert len(ensemble_gen_data.time.sample) == n_ensemble * n_samples
    assert len(ensemble_gen_data.time.time) == n_times

    for i in range(n_ensemble):
        assert (
            ensemble_gen_data.labels[i * n_samples : (i * n_samples) + n_samples]
            == gen_data.labels
        )
        assert ensemble_gen_data.time[
            i * n_samples : (i * n_samples) + n_samples
        ].equals(gen_data.time)

    for i in range(n_samples):
        torch.testing.assert_close(
            ensemble_gen_data.data["bar"][i * n_ensemble],
            gen_data.data["bar"][i],
        )

    assert_metadata_equal(ensemble_gen_data, gen_data)


@pytest.mark.parametrize("n_ensemble", [1, 2, 3])
def test_ensemble_data_size(n_ensemble):
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

    ensemble_target_data = target_data.broadcast_ensemble(n_ensemble=1)
    ensemble_gen_data = gen_data.broadcast_ensemble(n_ensemble=n_ensemble)

    assert ensemble_gen_data.ensemble_data["bar"].shape == (
        n_samples,
        n_ensemble,
        n_times,
        n_lat,
        n_lon,
    )

    assert ensemble_target_data.ensemble_data["bar"].shape == (
        n_samples,
        1,
        n_times,
        n_lat,
        n_lon,
    )

    for ensemble_idx in range(n_ensemble - 1):
        torch.testing.assert_close(
            ensemble_gen_data.ensemble_data["bar"][:, ensemble_idx, ...],
            ensemble_gen_data.ensemble_data["bar"][:, ensemble_idx + 1, ...],
        )
