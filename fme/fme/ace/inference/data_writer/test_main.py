import os
import tempfile

import numpy as np
import torch
import xarray as xr

from fme.ace.inference.data_writer.main import save_initial_condition
from fme.core.data_loading.batch_data import BatchData
from fme.core.data_loading.data_typing import VariableMetadata


def test_save_initial_condition_single_timestep():
    n_samples = 2
    n_lat = 4
    n_lon = 5
    n_time = 1
    batch = BatchData(
        data={"air_temperature": torch.rand((n_samples, n_time, n_lat, n_lon))},
        times=xr.DataArray(np.random.rand(n_samples, n_time), dims=["sample", "time"]),
        horizontal_dims=["lat", "lon"],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        save_initial_condition(
            ic_data=batch,
            path=tmpdir,
            prognostic_names=["air_temperature"],
            metadata={
                "air_temperature": VariableMetadata(
                    long_name="Air Temperature", units="K"
                )
            },
            coords={"lat": np.arange(n_lat), "lon": np.arange(n_lon)},
        )
        filename = os.path.join(tmpdir, "initial_condition.nc")
        assert os.path.exists(filename)
        with xr.open_dataset(filename) as ds:
            assert "air_temperature" in ds
            assert ds.air_temperature.shape == (n_samples, n_lat, n_lon)
            assert ds.time.shape == (n_samples,)
            assert ds.air_temperature.dims == ("sample", "lat", "lon")
            xr.testing.assert_allclose(ds.time, batch.times.isel(time=0))
            np.testing.assert_allclose(
                ds.air_temperature.values,
                batch.data["air_temperature"].squeeze(dim=1).cpu().numpy(),
            )
            np.testing.assert_allclose(ds.coords["lat"].values, np.arange(n_lat))
            np.testing.assert_allclose(ds.coords["lon"].values, np.arange(n_lon))
            assert ds.air_temperature.attrs["long_name"] == "Air Temperature"
            assert ds.air_temperature.attrs["units"] == "K"


def test_save_initial_condition_multiple_timesteps():
    n_samples = 2
    n_lat = 4
    n_lon = 5
    n_time = 2
    batch = BatchData(
        data={"air_temperature": torch.rand((n_samples, n_time, n_lat, n_lon))},
        times=xr.DataArray(np.random.rand(n_samples, n_time), dims=["sample", "time"]),
        horizontal_dims=["lat", "lon"],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        save_initial_condition(
            ic_data=batch,
            path=tmpdir,
            prognostic_names=["air_temperature"],
            metadata={
                "air_temperature": VariableMetadata(
                    long_name="Air Temperature", units="K"
                )
            },
            coords={"lat": np.arange(n_lat), "lon": np.arange(n_lon)},
        )
        filename = os.path.join(tmpdir, "initial_condition.nc")
        assert os.path.exists(filename)
        with xr.open_dataset(filename) as ds:
            assert "air_temperature" in ds
            assert ds.air_temperature.shape == (n_samples, n_time, n_lat, n_lon)
            assert ds.time.shape == (n_samples, n_time)
            assert ds.air_temperature.dims == ("sample", "time", "lat", "lon")
            np.testing.assert_allclose(ds.time.values, batch.times.values)
            np.testing.assert_allclose(
                ds.air_temperature.values, batch.data["air_temperature"].cpu().numpy()
            )
            np.testing.assert_allclose(ds.coords["lat"].values, np.arange(n_lat))
            np.testing.assert_allclose(ds.coords["lon"].values, np.arange(n_lon))
            assert ds.air_temperature.attrs["long_name"] == "Air Temperature"
            assert ds.air_temperature.attrs["units"] == "K"