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
from fme.ace.inference.data_writer.main import DataWriterConfig, _write
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.step.step_diagnostics import StepDiagnostics


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
        filename = os.path.join(tmpdir, "autoregressive_step_diagnostics.nc")
        assert os.path.exists(filename)
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


def test_no_step_diagnostics_file_by_default_or_without_corrector():
    config, build_kwargs, batch, _ = _get_step_diagnostics_setup()

    # flag off: no file at all
    config.save_step_diagnostics = False
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = config.build_paired(experiment_dir=tmpdir, **build_kwargs)
        writer.append_batch(batch)
        writer.finalize()
        assert not os.path.exists(
            os.path.join(tmpdir, "autoregressive_step_diagnostics.nc")
        )

    # flag on, but no diagnostics on the batch: no diagnostics content
    config.save_step_diagnostics = True
    batch_no_diagnostics = dataclasses.replace(batch, step_diagnostics=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = config.build_paired(experiment_dir=tmpdir, **build_kwargs)
        writer.append_batch(batch_no_diagnostics)
        writer.finalize()
        filename = os.path.join(tmpdir, "autoregressive_step_diagnostics.nc")
        # the file's time variables were never filled, so skip time decoding
        with xr.open_dataset(
            filename, decode_times=False, decode_timedelta=False
        ) as ds:
            assert len(ds.data_vars.keys() - {"valid_time", "init_time"}) == 0
