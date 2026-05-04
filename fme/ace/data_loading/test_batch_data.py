import dataclasses

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PairedData
from fme.core.device import get_device
from fme.core.labels import BatchLabels
from fme.core.typing_ import TensorDict


def assert_metadata_equal(
    a: BatchData,
    b: BatchData,
    check_labels: bool = True,
    check_n_ensemble: bool = True,
):
    assert a.horizontal_dims == b.horizontal_dims
    assert a.epoch == b.epoch
    if check_n_ensemble:
        assert a.n_ensemble == b.n_ensemble
    if check_labels:
        assert a.labels == b.labels


def assert_batchdata_equal_up_to_device(a: BatchData, b: BatchData) -> None:
    """
    Assert that two BatchData objects are the same, comparing tensor *values* on CPU
    so that `.device` differences on tensor fields are ignored. Compares all
    public dataclass fields so new fields (e.g. n_ensemble) are covered without
    having to update only ``assert_metadata_equal``.
    """
    for field in dataclasses.fields(BatchData):
        va = getattr(a, field.name)
        vb = getattr(b, field.name)
        if field.name == "data":
            assert set(va) == set(vb)
            for k in va:
                torch.testing.assert_close(
                    va[k].detach().cpu(),
                    vb[k].detach().cpu(),
                )
        elif field.name == "time":
            assert bool(va.equals(vb))
        elif field.name == "labels":
            if va is None or vb is None:
                assert va is None and vb is None
            else:
                assert va.names == vb.names
                torch.testing.assert_close(
                    va.tensor.cpu(),
                    vb.tensor.cpu(),
                )
        else:
            assert va == vb


def assert_paired_data_equal_up_to_device(a: PairedData, b: PairedData) -> None:
    """Like ``assert_batchdata_equal_up_to_device`` for ``PairedData``."""
    for field in dataclasses.fields(PairedData):
        va = getattr(a, field.name)
        vb = getattr(b, field.name)
        if field.name in ("prediction", "reference"):
            assert set(va) == set(vb)
            for k in va:
                torch.testing.assert_close(
                    va[k].detach().cpu(),
                    vb[k].detach().cpu(),
                )
        elif field.name == "time":
            assert bool(va.equals(vb))
        elif field.name == "labels":
            if va is None or vb is None:
                assert va is None and vb is None
            else:
                assert va.names == vb.names
                torch.testing.assert_close(
                    va.tensor.cpu(),
                    vb.tensor.cpu(),
                )
        else:
            assert va == vb


def get_batch_data(
    names: list[str],
    n_samples: int,
    n_times: int,
    horizontal_dims: list[str],
    n_lat: int = 8,
    n_lon: int = 16,
    n_labels: int = 0,
    n_ensemble: int = 1,
):
    device = get_device()
    if n_labels == 0:
        labels = None
    else:
        labels = BatchLabels(
            torch.zeros(n_samples, n_labels, device=device),
            [f"label_{i}" for i in range(n_labels)],
        )
    return BatchData(
        data={
            name: torch.randn(n_samples, n_times, n_lat, n_lon, device=device)
            for name in names
        },
        time=xr.DataArray(np.random.rand(n_samples, n_times), dims=["sample", "time"]),
        horizontal_dims=horizontal_dims,
        labels=labels,
        n_ensemble=n_ensemble,
    )


def test_to_device():
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
    device_data = batch_data.to_device()
    device = get_device()
    assert_metadata_equal(device_data, batch_data)
    for name in names:
        assert device_data.data[name].device == device
        np.testing.assert_allclose(
            device_data.data[name].cpu().numpy(),
            batch_data.data[name].cpu().numpy(),
        )
    if batch_data.labels is not None:
        assert device_data.labels.tensor.device == device
        np.testing.assert_allclose(
            device_data.labels.tensor.cpu().numpy(),
            batch_data.labels.tensor.cpu().numpy(),
        )


def test_to_cpu():
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
    cpu_data = batch_data.to_cpu()
    assert_metadata_equal(cpu_data, batch_data)
    for name in names:
        assert cpu_data.data[name].device.type == "cpu"
        np.testing.assert_allclose(
            cpu_data.data[name].cpu().numpy(),
            batch_data.data[name].cpu().numpy(),
        )
    if batch_data.labels is not None:
        assert cpu_data.labels.tensor.device.type == "cpu"
        np.testing.assert_allclose(
            cpu_data.labels.tensor.cpu().numpy(),
            batch_data.labels.tensor.cpu().numpy(),
        )


@pytest.mark.parametrize(
    "epoch_1, epoch_2",
    [
        (None, 0),
        (0, 1),
    ],
)
def test_from_sample_tuples_raises_on_mismatched_epochs(epoch_1: int, epoch_2: int):
    sample1 = ({"x": torch.zeros(2, 3)}, xr.DataArray([0]), None, epoch_1)
    sample2 = ({"x": torch.zeros(2, 3)}, xr.DataArray([0]), None, epoch_2)

    with pytest.raises(ValueError, match="same epoch"):
        BatchData.from_sample_tuples([sample1, sample2])


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
        n_labels=2,
    )

    gen_data = get_batch_data(
        names=["bar"],
        n_samples=n_samples,
        n_times=n_times,
        horizontal_dims=horizontal_dims,
        n_lat=n_lat,
        n_lon=n_lon,
        n_labels=2,
    )

    ensemble_target_data = target_data.broadcast_ensemble(n_ensemble=1)
    ensemble_gen_data = gen_data.broadcast_ensemble(n_ensemble=n_ensemble)

    assert ensemble_target_data.labels.tensor.shape == (n_samples, 2)
    assert ensemble_gen_data.labels.tensor.shape == (n_ensemble * n_samples, 2)
    assert ensemble_target_data.labels.names == ["label_0", "label_1"]
    assert ensemble_gen_data.labels.names == ["label_0", "label_1"]

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
    assert ensemble_gen_data.labels.tensor.shape[0] == n_ensemble * n_samples
    assert len(ensemble_gen_data.time.sample) == n_ensemble * n_samples
    assert len(ensemble_gen_data.time.time) == n_times

    for i in range(n_ensemble):
        torch.testing.assert_close(
            ensemble_gen_data.labels.tensor[
                i * n_samples : (i * n_samples) + n_samples
            ],
            gen_data.labels.tensor,
        )
        assert ensemble_gen_data.time[
            i * n_samples : (i * n_samples) + n_samples
        ].equals(gen_data.time)

    for i in range(n_samples):
        torch.testing.assert_close(
            ensemble_gen_data.data["bar"][i * n_ensemble],
            gen_data.data["bar"][i],
        )

    assert_metadata_equal(
        ensemble_gen_data, gen_data, check_labels=False, check_n_ensemble=False
    )  # n_ensemble and labels intentionally differ; labels checked above


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


def test_n_ensemble_preserved_across_batchdata_transforms():
    """
    Regression: any ``BatchData`` method that returns a new ``BatchData`` must
    forward ``n_ensemble`` (see dataclass and ``broadcast_ensemble``/dataloader
    invariants). Bundle these tests when splitting a focused PR for batch
    container fixes.
    """
    n_e = 3
    names = ["foo", "bar", "qux"]
    n_samples, n_times = 2, 5
    base = get_batch_data(
        names=names,
        n_samples=n_samples,
        n_times=n_times,
        horizontal_dims=["lat", "lon"],
        n_ensemble=n_e,
    )
    assert base.remove_initial_condition(1).n_ensemble == n_e
    assert base.select_time_slice(slice(0, 2)).n_ensemble == n_e
    assert base.subset_names(["foo", "qux"]).n_ensemble == n_e
    ic = base.get_start(["foo", "bar"], 2)
    assert ic.as_batch_data().n_ensemble == n_e
    assert base.prepend(ic).n_ensemble == n_e
    bcpu = base.to_cpu()
    if torch.cuda.is_available():
        bcpu.pin_memory()
    assert bcpu.n_ensemble == n_e
    assert base.to_device().n_ensemble == n_e
    assert base.to_cpu().n_ensemble == n_e

    derived = base.compute_derived_variables(lambda a, f: TensorDict({}), base)
    assert derived.n_ensemble == n_e
    assert_batchdata_equal_up_to_device(derived, base)
    a = get_batch_data(
        names=["x"],
        n_samples=1,
        n_times=2,
        horizontal_dims=["lat", "lon"],
        n_ensemble=n_e,
    )
    b = a.to_device().to_cpu()
    assert_batchdata_equal_up_to_device(a, b)


def test_batchdata_broadcast_ensemble_does_not_move_cpu_tensors_to_default_device():
    time = xr.DataArray(np.random.rand(2, 3), dims=["sample", "time"])
    base = BatchData(
        data={"a": torch.zeros(2, 3, 4, 4, device=torch.device("cpu"))},
        time=time,
        horizontal_dims=["lat", "lon"],
        n_ensemble=1,
    )
    out = base.broadcast_ensemble(2)
    for v in out.data.values():
        assert v.device == torch.device("cpu")


def test_paired_data_from_batch_data_copies_n_ensemble():
    n_e = 2
    t = get_batch_data(
        names=["foo", "bar"],
        n_samples=1,
        n_times=2,
        horizontal_dims=["lat", "lon"],
        n_ensemble=n_e,
    )
    g = get_batch_data(
        names=["bar"],
        n_samples=1,
        n_times=2,
        horizontal_dims=["lat", "lon"],
        n_ensemble=n_e,
    )
    g.time = t.time
    paired = PairedData.from_batch_data(prediction=g, reference=t)
    assert paired.n_ensemble == n_e


def test_paired_data_equal_up_to_device_helper_matches_new_on_device():
    """Exercises the field-by-field ``PairedData`` comparison helper."""
    t = get_batch_data(
        names=["foo", "bar"],
        n_samples=1,
        n_times=2,
        horizontal_dims=["lat", "lon"],
        n_ensemble=1,
    )
    g = get_batch_data(
        names=["bar"],
        n_samples=1,
        n_times=2,
        horizontal_dims=["lat", "lon"],
        n_ensemble=1,
    )
    g.time = t.time
    a = PairedData.from_batch_data(prediction=g, reference=t)
    b = PairedData.new_on_device(
        a.prediction, a.reference, a.time, a.labels, n_ensemble=a.n_ensemble
    )
    assert_paired_data_equal_up_to_device(a, b)


def test_paired_data_as_ensemble_tensor_dicts_shape():
    n_ensemble, n_s, n_t = 2, 1, 3
    t = get_batch_data(
        names=["foo", "bar"],
        n_samples=n_s * n_ensemble,
        n_times=n_t,
        horizontal_dims=["lat", "lon"],
        n_ensemble=1,
    )
    g = get_batch_data(
        names=["bar"],
        n_samples=n_s * n_ensemble,
        n_times=n_t,
        horizontal_dims=["lat", "lon"],
        n_ensemble=1,
    )
    g.time = t.time
    p = PairedData.from_batch_data(prediction=g, reference=t)
    u_ref, u_pred = p.as_ensemble_tensor_dicts(n_ensemble)
    for name in p.reference:
        assert u_ref[name].shape[:2] == (n_s, n_ensemble)
    for name in p.prediction:
        assert u_pred[name].shape[:2] == (n_s, n_ensemble)
