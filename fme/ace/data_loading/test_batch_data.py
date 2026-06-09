import dataclasses

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import (
    BatchData,
    PairedData,
    PrognosticState,
    _collate_with_masking,
)
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.labels import BatchLabels
from fme.core.typing_ import TensorDict

_METADATA_FIELDS = {"horizontal_dims", "epoch", "n_ensemble", "labels", "data_mask"}
_NON_METADATA_FIELDS = {"data", "time"}


def assert_metadata_equal(
    a: BatchData,
    b: BatchData,
    check_labels: bool = True,
    check_n_ensemble: bool = True,
    check_data_mask: bool = True,
):
    actual_fields = {f.name for f in dataclasses.fields(BatchData)}
    unexpected = actual_fields - _METADATA_FIELDS - _NON_METADATA_FIELDS
    if unexpected:
        raise AssertionError(
            f"BatchData has new fields {unexpected} not covered by "
            f"assert_metadata_equal. Update this helper to check them."
        )
    assert a.horizontal_dims == b.horizontal_dims
    assert a.epoch == b.epoch
    if check_n_ensemble:
        assert a.n_ensemble == b.n_ensemble
    if check_labels:
        assert a.labels == b.labels
    if check_data_mask:
        _assert_tensor_mapping_equal_up_to_device(
            dict(a.data_mask) if a.data_mask is not None else None,
            dict(b.data_mask) if b.data_mask is not None else None,
        )


def _assert_tensor_mapping_equal_up_to_device(va: dict | None, vb: dict | None) -> None:
    if va is None or vb is None:
        assert va is None and vb is None
    else:
        assert set(va) == set(vb)
        for k in va:
            torch.testing.assert_close(va[k].detach().cpu(), vb[k].detach().cpu())


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
        elif field.name == "data_mask":
            _assert_tensor_mapping_equal_up_to_device(va, vb)
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
        elif field.name == "data_mask":
            _assert_tensor_mapping_equal_up_to_device(va, vb)
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
    data_mask = {
        name: torch.ones(n_samples, dtype=torch.bool, device=device) for name in names
    }
    data_mask[names[-1]] = torch.zeros(n_samples, dtype=torch.bool, device=device)
    return BatchData(
        data={
            name: torch.randn(n_samples, n_times, n_lat, n_lon, device=device)
            for name in names
        },
        time=xr.DataArray(np.random.rand(n_samples, n_times), dims=["sample", "time"]),
        horizontal_dims=horizontal_dims,
        labels=labels,
        n_ensemble=n_ensemble,
        data_mask=data_mask,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory requires CUDA")
def test_pin_memory():
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
        n_labels=2,
    )
    cpu_data = batch_data.to_cpu()
    pinned = cpu_data.pin_memory()
    assert_metadata_equal(pinned, cpu_data)
    for name in names:
        assert pinned.data[name].is_pinned()
    assert pinned.labels is not None
    assert pinned.labels.tensor.is_pinned()
    assert pinned.data_mask is not None
    for name in names:
        assert pinned.data_mask[name].is_pinned()


@pytest.mark.parametrize(
    "epoch_1, epoch_2",
    [
        (None, 0),
        (0, 1),
    ],
)
def test_from_sample_tuples_raises_on_mismatched_epochs(epoch_1: int, epoch_2: int):
    sample1 = ({"x": torch.zeros(2, 3)}, xr.DataArray([0]), None, epoch_1, None)
    sample2 = ({"x": torch.zeros(2, 3)}, xr.DataArray([0]), None, epoch_2, None)

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
    assert_metadata_equal(start, batch_data, check_data_mask=False)
    assert start.time.equals(batch_data.time.isel(time=slice(0, n_ic_timesteps)))
    assert set(start.data.keys()) == set(prognostic_names)
    if batch_data.data_mask is not None:
        assert start.data_mask is not None
        for name in prognostic_names:
            torch.testing.assert_close(
                start.data_mask[name].cpu(), batch_data.data_mask[name].cpu()
            )
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
    assert_metadata_equal(end, batch_data, check_data_mask=False)
    assert end.time.equals(batch_data.time.isel(time=slice(-n_ic_timesteps, None)))
    assert set(end.data.keys()) == set(prognostic_names)
    if batch_data.data_mask is not None:
        assert end.data_mask is not None
        for name in prognostic_names:
            torch.testing.assert_close(
                end.data_mask[name].cpu(), batch_data.data_mask[name].cpu()
            )
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
        ensemble_gen_data,
        gen_data,
        check_labels=False,
        check_n_ensemble=False,
        check_data_mask=False,
    )

    assert ensemble_gen_data.data_mask is not None
    assert ensemble_gen_data.data_mask["bar"].shape == (n_ensemble * n_samples,)
    for i in range(n_samples):
        original_val = gen_data.data_mask["bar"][i].item()
        for e in range(n_ensemble):
            assert (
                ensemble_gen_data.data_mask["bar"][i * n_ensemble + e].item()
                == original_val
            )


@pytest.mark.parallel
def test_scatter_spatial():
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
    global_img_shape = (n_lat, n_lon)
    scattered = batch_data.scatter_spatial(global_img_shape)
    dist = Distributed.get_instance()
    local_slices = dist.get_local_slices(global_img_shape)
    assert_metadata_equal(scattered, batch_data)
    for name in names:
        expected = batch_data.data[name][(..., *local_slices)].contiguous()
        torch.testing.assert_close(scattered.data[name], expected)


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


def test_data_mask_raises_if_length_does_not_match_n_samples():
    base = get_batch_data(
        names=["foo"],
        n_samples=2,
        n_times=3,
        horizontal_dims=["lat", "lon"],
    )
    with pytest.raises(ValueError, match="data_mask for variable foo"):
        BatchData(
            data=base.data,
            time=base.time,
            data_mask={"foo": torch.ones(3, dtype=torch.bool)},
        )


def test_collate_with_masking_heterogeneous_variables():
    sample_a: TensorDict = {
        "x": torch.ones(2, 3),
        "y": torch.ones(2, 3) * 2,
    }
    sample_b: TensorDict = {
        "x": torch.ones(2, 3) * 3,
        "y": torch.full((2, 3), float("nan")),
    }
    batch_data, data_mask = _collate_with_masking(
        [sample_a, sample_b],
        sample_missing_names=[None, frozenset({"y"})],
    )
    assert set(batch_data.keys()) == {"x", "y"}
    assert batch_data["x"].shape == (2, 2, 3)
    torch.testing.assert_close(batch_data["x"][0], torch.ones(2, 3))
    torch.testing.assert_close(batch_data["x"][1], torch.ones(2, 3) * 3)
    torch.testing.assert_close(batch_data["y"][0], torch.ones(2, 3) * 2)
    torch.testing.assert_close(
        batch_data["y"][1], torch.full((2, 3), float("nan")), equal_nan=True
    )
    assert data_mask is not None
    torch.testing.assert_close(data_mask["x"], torch.tensor([True, True]))
    torch.testing.assert_close(data_mask["y"], torch.tensor([True, False]))


def test_collate_with_masking_all_present_returns_none_mask():
    sample_a: TensorDict = {"x": torch.ones(2, 3), "y": torch.ones(2, 3)}
    sample_b: TensorDict = {"x": torch.ones(2, 3), "y": torch.ones(2, 3)}
    batch_data, data_mask = _collate_with_masking(
        [sample_a, sample_b], sample_missing_names=[None, None]
    )
    assert data_mask is None
    assert batch_data["x"].shape == (2, 2, 3)


def test_from_sample_tuples_with_variable_masking():
    sample1 = (
        {"a": torch.ones(2, 3), "b": torch.ones(2, 3) * 2},
        xr.DataArray([0, 1]),
        None,
        0,
        None,
    )
    sample2 = (
        {"a": torch.ones(2, 3) * 3, "b": torch.full((2, 3), float("nan"))},
        xr.DataArray([0, 1]),
        None,
        0,
        frozenset({"b"}),
    )
    batch = BatchData.from_sample_tuples(
        [sample1, sample2], allow_missing_variables=True
    )
    assert "a" in batch.data
    assert "b" in batch.data
    assert batch.data_mask is not None
    torch.testing.assert_close(batch.data_mask["b"], torch.tensor([True, False]))
    torch.testing.assert_close(
        batch.data["b"][1], torch.full((2, 3), float("nan")), equal_nan=True
    )


def test_collate_with_masking_variable_missing_from_all_samples():
    sample_a: TensorDict = {
        "x": torch.ones(2, 3),
        "y": torch.full((2, 3), float("nan")),
    }
    sample_b: TensorDict = {
        "x": torch.ones(2, 3),
        "y": torch.full((2, 3), float("nan")),
    }
    batch_data, data_mask = _collate_with_masking(
        [sample_a, sample_b],
        sample_missing_names=[frozenset({"y"}), frozenset({"y"})],
    )
    assert "x" in batch_data
    assert "y" in batch_data
    assert data_mask is not None
    assert data_mask["x"].all()
    assert not data_mask["y"].any()


@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("with_data_mask", [False, True])
def test_batchdata_cat_and_split_roundtrip(with_labels: bool, with_data_mask: bool):
    names = ["foo", "bar"]
    n_times = 4
    horizontal_dims = ["lat", "lon"]
    n_labels = 2 if with_labels else 0
    a = get_batch_data(
        names=names,
        n_samples=2,
        n_times=n_times,
        horizontal_dims=horizontal_dims,
        n_labels=n_labels,
    )
    b = get_batch_data(
        names=names,
        n_samples=3,
        n_times=n_times,
        horizontal_dims=horizontal_dims,
        n_labels=n_labels,
    )
    # align time coords (cat does not require it, but split returns them so
    # the explicit indexing is well-defined for the round trip test)
    b.time = xr.DataArray(np.random.rand(3, n_times) + 100.0, dims=["sample", "time"])
    if not with_data_mask:
        a.data_mask = None
        b.data_mask = None
    cat = BatchData.cat([a, b])
    assert cat.time.shape == (5, n_times)
    for name in names:
        torch.testing.assert_close(cat.data[name][:2].cpu(), a.data[name].cpu())
        torch.testing.assert_close(cat.data[name][2:].cpu(), b.data[name].cpu())
    if with_labels:
        assert cat.labels is not None
        assert cat.labels.tensor.shape == (5, n_labels)
    if with_data_mask:
        assert cat.data_mask is not None
        for name in names:
            assert cat.data_mask[name].shape == (5,)
    pieces = cat.split([2, 3])
    assert len(pieces) == 2
    assert_batchdata_equal_up_to_device(pieces[0], a)
    assert_batchdata_equal_up_to_device(pieces[1], b)


def test_batchdata_cat_empty_raises():
    with pytest.raises(ValueError, match="empty sequence"):
        BatchData.cat([])


def test_batchdata_cat_single_returns_input():
    a = get_batch_data(
        names=["foo"],
        n_samples=2,
        n_times=3,
        horizontal_dims=["lat", "lon"],
    )
    assert BatchData.cat([a]) is a


def test_batchdata_cat_mismatched_n_timesteps_raises():
    a = get_batch_data(
        names=["foo"], n_samples=2, n_times=3, horizontal_dims=["lat", "lon"]
    )
    b = get_batch_data(
        names=["foo"], n_samples=2, n_times=4, horizontal_dims=["lat", "lon"]
    )
    with pytest.raises(ValueError, match="n_timesteps"):
        BatchData.cat([a, b])


def test_batchdata_cat_mismatched_n_ensemble_raises():
    a = get_batch_data(
        names=["foo"],
        n_samples=2,
        n_times=3,
        horizontal_dims=["lat", "lon"],
        n_ensemble=1,
    )
    b = get_batch_data(
        names=["foo"],
        n_samples=2,
        n_times=3,
        horizontal_dims=["lat", "lon"],
        n_ensemble=2,
    )
    with pytest.raises(ValueError, match="n_ensemble"):
        BatchData.cat([a, b])


def test_batchdata_cat_unions_different_label_names():
    """Cat is permitted across batches with different label encodings."""
    device = get_device()
    a = get_batch_data(
        names=["foo"], n_samples=2, n_times=3, horizontal_dims=["lat", "lon"]
    )
    b = get_batch_data(
        names=["foo"], n_samples=3, n_times=3, horizontal_dims=["lat", "lon"]
    )
    a.labels = BatchLabels(torch.ones(2, 2, device=device), ["x", "y"])
    b.labels = BatchLabels(torch.ones(3, 2, device=device), ["y", "z"])
    cat = BatchData.cat([a, b])
    assert cat.labels is not None
    assert cat.labels.names == ["x", "y", "z"]
    # Sample 0 was originally "x"+"y" → ["x","y","z"] = [1,1,0]
    # Sample 2 was originally "y"+"z" → ["x","y","z"] = [0,1,1]
    expected = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        device=device,
    )
    torch.testing.assert_close(cat.labels.tensor, expected)


def test_batchdata_split_wrong_sum_raises():
    a = get_batch_data(
        names=["foo"], n_samples=2, n_times=3, horizontal_dims=["lat", "lon"]
    )
    with pytest.raises(ValueError, match="sample_sizes"):
        a.split([1, 2])


def test_prognostic_state_cat_and_split_roundtrip():
    a = get_batch_data(
        names=["foo"], n_samples=2, n_times=2, horizontal_dims=["lat", "lon"]
    )
    b = get_batch_data(
        names=["foo"], n_samples=3, n_times=2, horizontal_dims=["lat", "lon"]
    )
    sa = PrognosticState(a)
    sb = PrognosticState(b)
    cat = PrognosticState.cat([sa, sb])
    assert cat.as_batch_data().time.shape == (5, 2)
    pieces = cat.split([2, 3])
    assert len(pieces) == 2
    assert_batchdata_equal_up_to_device(pieces[0].as_batch_data(), a)
    assert_batchdata_equal_up_to_device(pieces[1].as_batch_data(), b)


def _paired(
    prediction_names: list[str],
    reference_names: list[str],
    n_samples: int,
    n_times: int = 3,
    n_labels: int = 0,
) -> PairedData:
    device = get_device()
    if n_labels == 0:
        labels = None
    else:
        labels = BatchLabels(
            torch.zeros(n_samples, n_labels, device=device),
            [f"label_{i}" for i in range(n_labels)],
        )
    return PairedData(
        prediction={
            k: torch.randn(n_samples, n_times, 4, 8, device=device)
            for k in prediction_names
        },
        reference={
            k: torch.randn(n_samples, n_times, 4, 8, device=device)
            for k in reference_names
        },
        time=xr.DataArray(np.random.rand(n_samples, n_times), dims=["sample", "time"]),
        labels=labels,
    )


@pytest.mark.parametrize("with_labels", [False, True])
def test_paired_data_cat_and_split_roundtrip(with_labels: bool):
    n_labels = 2 if with_labels else 0
    a = _paired(["bar"], ["foo", "bar"], n_samples=2, n_labels=n_labels)
    b = _paired(["bar"], ["foo", "bar"], n_samples=3, n_labels=n_labels)
    cat = PairedData.cat([a, b])
    assert cat.time.shape == (5, 3)
    torch.testing.assert_close(
        cat.prediction["bar"][:2].cpu(), a.prediction["bar"].cpu()
    )
    torch.testing.assert_close(cat.reference["foo"][2:].cpu(), b.reference["foo"].cpu())
    pieces = cat.split([2, 3])
    assert len(pieces) == 2
    assert_paired_data_equal_up_to_device(pieces[0], a)
    assert_paired_data_equal_up_to_device(pieces[1], b)


def test_paired_data_cat_mismatched_prediction_names_raises():
    a = _paired(["bar"], ["foo", "bar"], n_samples=2)
    b = _paired(["foo"], ["foo", "bar"], n_samples=2)
    with pytest.raises(ValueError, match="prediction variables"):
        PairedData.cat([a, b])
