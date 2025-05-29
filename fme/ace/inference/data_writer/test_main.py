import os
import tempfile

import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.main import _write
from fme.core.dataset.data_typing import VariableMetadata


def test_write_single_timestep():
    n_samples = 2
    n_lat = 4
    n_lon = 5
    n_time = 1
    batch = BatchData.new_on_cpu(
        data={"air_temperature": torch.rand((n_samples, n_time, n_lat, n_lon))},
        time=xr.DataArray(np.random.rand(n_samples, n_time), dims=["sample", "time"]),
        horizontal_dims=["lat", "lon"],
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
