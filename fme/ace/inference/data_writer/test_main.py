import dataclasses
import datetime
import os
import tempfile

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PairedData
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.file_writer import FileWriterConfig
from fme.ace.inference.data_writer.main import (
    DataWriterConfig,
    _gather_tensor_dict,
    _gather_time,
    _write,
)
from fme.ace.inference.data_writer.raw import RawDataWriter
from fme.ace.inference.data_writer.step_diagnostics import StepDiagnosticsWriter
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.step.step_diagnostics import StepDiagnostics

_WRITER_ENABLING_FIELDS = {
    "save_prediction_files",
    "save_monthly_files",
    "save_step_diagnostics",
    "files",
}


@pytest.mark.parametrize("field", sorted(_WRITER_ENABLING_FIELDS))
def test_has_subwriters_enabled_covers_field(field: str):
    """Turning on one writer-enabling field makes has_subwriters_enabled True."""
    kwargs: dict = {f: False for f in _WRITER_ENABLING_FIELDS if f != "files"}
    kwargs["files"] = None
    if field == "files":
        kwargs["files"] = [FileWriterConfig(label="test")]
    else:
        kwargs[field] = True
    config = DataWriterConfig(**kwargs)
    assert config.has_subwriters_enabled


def test_has_subwriters_enabled_all_off():
    config = DataWriterConfig(
        save_prediction_files=False,
        save_monthly_files=False,
        save_step_diagnostics=False,
        files=None,
    )
    assert not config.has_subwriters_enabled


def test_has_subwriters_enabled_field_set_is_current():
    """Fail if a new boolean or list[*Config] field is added that might enable
    a writer but isn't listed in _WRITER_ENABLING_FIELDS."""
    non_writer_fields = {"names", "time_coarsen"}
    known = _WRITER_ENABLING_FIELDS | non_writer_fields
    actual = {f.name for f in dataclasses.fields(DataWriterConfig)}
    unknown = actual - known
    assert not unknown, (
        f"DataWriterConfig has new field(s) {unknown} not accounted for in "
        f"test_has_subwriters_enabled_field_set_is_current. Add each to "
        f"_WRITER_ENABLING_FIELDS or non_writer_fields."
    )


def test_write_single_timestep():
    n_samples = 2
    n_lat = 4
    n_lon = 5
    n_time = 1
    batch = BatchData.new_on_cpu(
        data={"air_temperature": torch.rand((n_samples, n_time, n_lat, n_lon))},
        time=xr.DataArray(np.random.rand(n_samples, n_time), dims=["sample", "time"]),
        horizontal_dims=["lat", "lon"],
        labels=None,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _write(
            data=batch,
            path=tmpdir,
            filename="initial_condition.nc",
            variable_metadata={
                "air_temperature": VariableMetadata(
                    long_name="Air Temperature", units="K"
                )
            },
            coords={"lat": np.arange(n_lat), "lon": np.arange(n_lon)},
            dataset_metadata=DatasetMetadata(
                history={"created": "2023-10-01T00:00:00"}
            ),
        )
        filename = os.path.join(tmpdir, "initial_condition.nc")
        assert os.path.exists(filename)
        with xr.open_dataset(filename, decode_timedelta=False) as ds:
            assert "air_temperature" in ds
            assert ds.air_temperature.shape == (n_samples, n_lat, n_lon)
            assert ds.time.shape == (n_samples,)
            assert ds.air_temperature.dims == ("sample", "lat", "lon")
            xr.testing.assert_allclose(ds.time, batch.time.isel(time=0))
            np.testing.assert_allclose(
                ds.air_temperature.values,
                batch.data["air_temperature"].squeeze(dim=1).cpu().numpy(),
            )
            np.testing.assert_allclose(ds.coords["lat"].values, np.arange(n_lat))
            np.testing.assert_allclose(ds.coords["lon"].values, np.arange(n_lon))
            assert ds.air_temperature.attrs["long_name"] == "Air Temperature"
            assert ds.air_temperature.attrs["units"] == "K"
            assert ds.attrs["history.created"] == "2023-10-01T00:00:00"


def test_write_multiple_timesteps():
    n_samples = 2
    n_lat = 4
    n_lon = 5
    n_time = 2
    batch = BatchData.new_on_cpu(
        data={"air_temperature": torch.rand((n_samples, n_time, n_lat, n_lon))},
        time=xr.DataArray(np.random.rand(n_samples, n_time), dims=["sample", "time"]),
        horizontal_dims=["lat", "lon"],
        labels=None,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _write(
            data=batch,
            path=tmpdir,
            filename="initial_condition.nc",
            variable_metadata={
                "air_temperature": VariableMetadata(
                    long_name="Air Temperature", units="K"
                )
            },
            coords={"lat": np.arange(n_lat), "lon": np.arange(n_lon)},
            dataset_metadata=DatasetMetadata(),
        )
        filename = os.path.join(tmpdir, "initial_condition.nc")
        assert os.path.exists(filename)
        with xr.open_dataset(filename, decode_timedelta=False) as ds:
            assert "air_temperature" in ds
            assert ds.air_temperature.shape == (n_samples, n_time, n_lat, n_lon)
            assert ds.time.shape == (n_samples, n_time)
            assert ds.air_temperature.dims == ("sample", "time", "lat", "lon")
            np.testing.assert_allclose(ds.time.values, batch.time.values)
            np.testing.assert_allclose(
                ds.air_temperature.values, batch.data["air_temperature"].cpu().numpy()
            )
            np.testing.assert_allclose(ds.coords["lat"].values, np.arange(n_lat))
            np.testing.assert_allclose(ds.coords["lon"].values, np.arange(n_lon))
            assert ds.air_temperature.attrs["long_name"] == "Air Temperature"
            assert ds.air_temperature.attrs["units"] == "K"


def _get_step_diagnostics_setup(n_samples=2, n_times=4, n_lat=4, n_lon=5):
    times = xr.DataArray(
        np.stack(
            n_samples
            * [
                xr.date_range(
                    "2020-01-01",
                    periods=n_times,
                    freq="6h",
                    calendar="julian",
                    use_cftime=True,
                )
            ]
        ),
        dims=["sample", "time"],
    )
    initial_condition_times = times.values[:, 0]
    device = get_device()
    delta = {
        "a": torch.arange(
            n_samples * n_times * n_lat * n_lon, dtype=torch.float32
        ).reshape(n_samples, n_times, n_lat, n_lon),
        "b": torch.ones(n_samples, n_times, n_lat, n_lon),
    }
    batch = PairedData(
        prediction={
            "a": torch.rand(n_samples, n_times, n_lat, n_lon, device=device),
            "b": torch.rand(n_samples, n_times, n_lat, n_lon, device=device),
        },
        reference={
            "a": torch.rand(n_samples, n_times, n_lat, n_lon, device=device),
            "b": torch.rand(n_samples, n_times, n_lat, n_lon, device=device),
        },
        time=times,
        step_diagnostics=StepDiagnostics(
            delta={k: v.to(device) for k, v in delta.items()}
        ),
    )
    build_kwargs = dict(
        initial_condition_times=initial_condition_times,
        n_timesteps=n_times,
        timestep=datetime.timedelta(hours=6),
        variable_metadata={},
        coords={"lat": np.arange(n_lat), "lon": np.arange(n_lon)},
        dataset_metadata=DatasetMetadata(),
    )
    config = DataWriterConfig(
        save_prediction_files=False,
        save_monthly_files=False,
        save_step_diagnostics=True,
    )
    return config, build_kwargs, batch, delta


@pytest.mark.parametrize("builder", ["build_paired", "build"])
@pytest.mark.parametrize("save_names", [None, ["a"]])
@pytest.mark.parametrize("coarsen_factor", [1, 2])
def test_step_diagnostics_writer_writes_delta_series(
    builder, save_names, coarsen_factor
):
    config, build_kwargs, batch, delta = _get_step_diagnostics_setup()
    config.names = save_names
    if coarsen_factor > 1:
        config.time_coarsen = TimeCoarsenConfig(coarsen_factor=coarsen_factor)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = getattr(config, builder)(experiment_dir=tmpdir, **build_kwargs)
        writer.append_batch(batch)
        writer.finalize()
        filename = os.path.join(tmpdir, "step_diagnostics", "correction_deltas.nc")
        assert os.path.exists(filename)
        assert not os.path.exists(
            os.path.join(tmpdir, "autoregressive_step_diagnostics.nc")
        )
        with xr.open_dataset(filename, decode_timedelta=False) as ds:
            expected_names = {"a"} if save_names == ["a"] else {"a", "b"}
            assert expected_names.issubset(set(ds.data_vars))
            if save_names == ["a"]:
                assert "b" not in ds.data_vars
            for name in expected_names:
                expected = delta[name]
                if coarsen_factor > 1:
                    expected = expected.unfold(
                        dimension=1, size=coarsen_factor, step=coarsen_factor
                    ).mean(dim=-1)
                assert ds[name].shape == tuple(expected.shape)
                np.testing.assert_allclose(ds[name].values, expected.numpy(), rtol=1e-6)


def test_no_step_diagnostics_output_by_default_or_without_corrector():
    config, build_kwargs, batch, _ = _get_step_diagnostics_setup()

    # flag off: no step-diagnostics output at all
    config.save_step_diagnostics = False
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = config.build_paired(experiment_dir=tmpdir, **build_kwargs)
        writer.append_batch(batch)
        writer.finalize()
        assert not os.path.exists(os.path.join(tmpdir, "step_diagnostics"))

    # flag on, but no diagnostics on the batch: no step-diagnostics output
    config.save_step_diagnostics = True
    batch_no_diagnostics = dataclasses.replace(batch, step_diagnostics=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = config.build_paired(experiment_dir=tmpdir, **build_kwargs)
        writer.append_batch(batch_no_diagnostics)
        writer.finalize()
        assert not os.path.exists(os.path.join(tmpdir, "step_diagnostics"))

    # flag on, but an empty delta: no step-diagnostics output
    batch_empty_delta = dataclasses.replace(
        batch, step_diagnostics=StepDiagnostics(delta={})
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = config.build_paired(experiment_dir=tmpdir, **build_kwargs)
        writer.append_batch(batch_empty_delta)
        writer.finalize()
        assert not os.path.exists(os.path.join(tmpdir, "step_diagnostics"))


def test_step_diagnostics_writer_writes_one_file_per_named_dataset():
    _, build_kwargs, batch, _ = _get_step_diagnostics_setup()
    n_samples, n_times, n_lat, n_lon = 2, 4, 4, 5

    def make_dataset(value: float) -> xr.Dataset:
        return xr.Dataset(
            {
                "a": xr.DataArray(
                    np.full((n_samples, n_times, n_lat, n_lon), value),
                    dims=["sample", "time", "lat", "lon"],
                )
            }
        ).assign_coords(valid_time=(("sample", "time"), batch.time.values))

    with tempfile.TemporaryDirectory() as tmpdir:

        def writer_factory(name: str) -> RawDataWriter:
            path = os.path.join(tmpdir, "step_diagnostics")
            os.makedirs(path, exist_ok=True)
            return RawDataWriter(
                path=path,
                label=name,
                initial_condition_times=build_kwargs["initial_condition_times"],
                save_names=None,
                variable_metadata={},
                coords=build_kwargs["coords"],
                dataset_metadata=DatasetMetadata(),
            )

        writer = StepDiagnosticsWriter(writer_factory)
        writer.append_batch(
            {
                "first": make_dataset(1.0),
                "second": make_dataset(2.0),
                "empty": xr.Dataset(),
            }
        )
        writer.finalize()
        for name, expected_value in [("first", 1.0), ("second", 2.0)]:
            filename = os.path.join(tmpdir, "step_diagnostics", f"{name}.nc")
            assert os.path.exists(filename)
            with xr.open_dataset(filename, decode_timedelta=False) as ds:
                assert ds["a"].shape == (n_samples, n_times, n_lat, n_lon)
                np.testing.assert_allclose(ds["a"].values, expected_value)
        assert not os.path.exists(os.path.join(tmpdir, "step_diagnostics", "empty.nc"))


def test_gather_tensor_dict_single_rank():
    """Under a single data-parallel rank, _gather_tensor_dict returns the input."""
    data = {
        "b": torch.tensor([[1.0, 2.0]]),
        "a": torch.tensor([[3.0, 4.0]]),
    }
    dist = Distributed.get_instance()
    result = _gather_tensor_dict(data, dist)
    assert result is not None
    assert set(result.keys()) == {"a", "b"}
    torch.testing.assert_close(result["a"], data["a"], check_device=False)
    torch.testing.assert_close(result["b"], data["b"], check_device=False)


def test_gather_time_single_rank():
    """Under a single data-parallel rank, _gather_time returns the input."""
    time = xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]), dims=["sample", "time"])
    dist = Distributed.get_instance()
    result = _gather_time(time, dist)
    assert result is not None
    xr.testing.assert_equal(result, time)


def test_netcdf_build_paired_under_multi_gpu_builds_writers_on_root():
    """build_paired with netCDF-only config succeeds and creates subwriters
    under simulated single-rank (root). This verifies the zarr guard is not
    triggered by netCDF writers."""
    n_lat, n_lon = 4, 5
    n_samples = 2
    n_timesteps = 4
    times = xr.DataArray(
        np.stack(
            n_samples
            * [
                xr.date_range(
                    "2020-01-01",
                    periods=n_timesteps,
                    freq="6h",
                    calendar="julian",
                    use_cftime=True,
                )
            ]
        ),
        dims=["sample", "time"],
    )
    initial_condition_times = times.values[:, 0]
    config = DataWriterConfig(
        save_prediction_files=True,
        save_monthly_files=False,
        save_step_diagnostics=False,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = config.build_paired(
            experiment_dir=tmpdir,
            initial_condition_times=initial_condition_times,
            n_timesteps=n_timesteps,
            timestep=datetime.timedelta(hours=6),
            variable_metadata={},
            coords={"lat": np.arange(n_lat), "lon": np.arange(n_lon)},
            dataset_metadata=DatasetMetadata(),
        )
        # Under single rank (the test default), subwriters should be built.
        assert len(writer._writers) == 1


@pytest.mark.parallel
def test_gather_tensor_dict_parallel():
    """Each rank's tensors are gathered and concatenated on root."""
    dist = Distributed.get_instance()
    rank = dist.data_parallel_rank
    data = {"x": torch.full((1, 3), float(rank))}
    result = _gather_tensor_dict(data, dist)
    if dist.is_data_parallel_root():
        assert result is not None
        assert result["x"].shape[0] == dist.total_data_parallel_ranks
        for r in range(dist.total_data_parallel_ranks):
            torch.testing.assert_close(
                result["x"][r], torch.full((3,), float(r)), check_device=False
            )
    else:
        assert result is None


@pytest.mark.parallel
def test_gather_time_parallel():
    """Each rank's time is gathered and concatenated on root."""
    import cftime

    dist = Distributed.get_instance()
    rank = dist.data_parallel_rank
    time = xr.DataArray(
        [[cftime.DatetimeProlepticGregorian(2020, 1, 1 + rank)]],
        dims=["sample", "time"],
    )
    result = _gather_time(time, dist)
    if dist.is_data_parallel_root():
        assert result is not None
        assert result.sizes["sample"] == dist.total_data_parallel_ranks
    else:
        assert result is None


@pytest.mark.parallel
def test_paired_data_writer_gathers_and_writes(tmp_path):
    """Under multi-GPU, PairedDataWriter gathers data to root and writes
    the full sample set. Non-root ranks produce no output files."""
    import cftime

    from fme.ace.inference.data_writer.main import PairedDataWriter
    from fme.ace.inference.data_writer.raw import PairedRawDataWriter

    dist = Distributed.get_instance()
    rank = dist.data_parallel_rank
    n_ranks = dist.total_data_parallel_ranks
    n_local = 2
    n_global = n_local * n_ranks
    n_lat, n_lon = 3, 4

    global_ic_times = np.array(
        [cftime.DatetimeProlepticGregorian(2020, 1, 1 + i) for i in range(n_global)]
    )
    experiment_dir = str(tmp_path / f"rank_{rank}")
    os.makedirs(experiment_dir, exist_ok=True)

    coords = {"lat": np.arange(n_lat), "lon": np.arange(n_lon)}

    if dist.is_data_parallel_root():
        raw_writer = PairedRawDataWriter(
            path=experiment_dir,
            initial_condition_times=global_ic_times,
            save_names=None,
            variable_metadata={},
            coords=coords,
            dataset_metadata=DatasetMetadata(),
        )
        writers = [raw_writer]
    else:
        writers = []

    writer = PairedDataWriter(
        writers=writers,
        path=experiment_dir,
        variable_metadata={},
        coords=coords,
        dataset_metadata=DatasetMetadata(),
    )

    prediction = {"temp": torch.full((n_local, 1, n_lat, n_lon), float(rank))}
    target = {"temp": torch.full((n_local, 1, n_lat, n_lon), float(rank) + 100)}
    time = xr.DataArray(
        [[cftime.DatetimeProlepticGregorian(2020, 2, 1 + s)] for s in range(n_local)],
        dims=["sample", "time"],
    )
    batch = PairedData(
        prediction=prediction,
        reference=target,
        time=time,
    )
    writer.append_batch(batch)
    writer.flush()
    writer.finalize()
    dist.barrier()

    if dist.is_data_parallel_root():
        pred_path = os.path.join(experiment_dir, "autoregressive_predictions.nc")
        assert os.path.exists(pred_path)
        ds = xr.open_dataset(pred_path, decode_timedelta=False)
        assert ds.sizes["sample"] == n_global
        for r in range(n_ranks):
            np.testing.assert_allclose(
                ds["temp"].values[r * n_local : (r + 1) * n_local], float(r)
            )
