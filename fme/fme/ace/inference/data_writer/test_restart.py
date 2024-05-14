import numpy as np
import torch
import xarray as xr

from fme.ace.inference.data_writer.restart import RestartWriter
from fme.core.data_loading.data_typing import VariableMetadata


def test_restart_saves_last_step(tmpdir):
    """
    If multiple steps are configured as restart steps, the last one should be saved.
    """
    n_sample: int = 3
    n_time: int = 2
    n_lat = 10
    n_lon = 20
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    writer = RestartWriter(
        path=tmpdir,
        is_restart_step=lambda i: True,
        prognostic_names=["a", "b"],
        metadata={"a": VariableMetadata(long_name="var_a", units="m")},
        coords={"lon": lon, "lat": lat},
    )
    data = {
        "a": torch.randn(n_sample, n_time, n_lat, n_lon),
        "b": torch.randn(n_sample, n_time, n_lat, n_lon),
    }
    batch_times = xr.DataArray(
        data=np.random.uniform(size=(n_sample, n_time)),
        dims=(
            "sample",
            "time",
        ),
    )
    writer.append_batch(
        prediction=data,
        start_timestep=0,
        batch_times=batch_times,
    )
    ds = xr.open_dataset(str(tmpdir / "restart.nc"))
    np.testing.assert_allclose(ds.a.values, data["a"][:, -1].cpu().numpy())
    np.testing.assert_allclose(ds.b.values, data["b"][:, -1].cpu().numpy())
    np.testing.assert_allclose(ds.time.values, batch_times[:, -1].values)
    assert len(ds.b.attrs) == 0
    assert len(ds.a.attrs) == 2
    assert ds.a.attrs["long_name"] == "var_a"
    assert ds.a.attrs["units"] == "m"
    np.testing.assert_allclose(ds.lon.values, lon)
    np.testing.assert_allclose(ds.lat.values, lat)
    assert ds.attrs["timestep"] == 1


def test_restart_saves_configured_step(tmpdir):
    """
    If a specific step is configured as a restart step, that step should be saved.
    """
    n_sample: int = 3
    i_time_start = 4
    i_time_target = 6
    n_time: int = 4
    n_lat = 10
    n_lon = 20
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    writer = RestartWriter(
        path=tmpdir,
        is_restart_step=lambda i: i == i_time_target,
        prognostic_names=["a", "b"],
        metadata={"a": VariableMetadata(long_name="var_a", units="m")},
        coords={"lon": lon, "lat": lat},
    )
    data = {
        "a": torch.randn(n_sample, n_time, n_lat, n_lon),
        "b": torch.randn(n_sample, n_time, n_lat, n_lon),
    }
    batch_times = xr.DataArray(
        data=np.random.uniform(size=(n_sample, n_time)),
        dims=(
            "sample",
            "time",
        ),
    )
    writer.append_batch(
        prediction=data,
        start_timestep=i_time_start,
        batch_times=batch_times,
    )
    ds = xr.open_dataset(str(tmpdir / "restart.nc"))
    np.testing.assert_allclose(
        ds.a.values, data["a"][:, i_time_target - i_time_start].cpu().numpy()
    )
    np.testing.assert_allclose(
        ds.b.values, data["b"][:, i_time_target - i_time_start].cpu().numpy()
    )
    np.testing.assert_allclose(
        ds.time.values, batch_times[:, i_time_target - i_time_start].values
    )
    assert len(ds.b.attrs) == 0
    assert len(ds.a.attrs) == 2
    assert ds.a.attrs["long_name"] == "var_a"
    assert ds.a.attrs["units"] == "m"
    np.testing.assert_allclose(ds.lon.values, lon)
    np.testing.assert_allclose(ds.lat.values, lat)
    assert ds.attrs["timestep"] == i_time_target


def test_restart_does_not_save(tmpdir):
    """
    If no step is configured to save as restart, no restart should be saved.
    """
    n_sample: int = 3
    n_time: int = 2
    n_lat = 10
    n_lon = 20
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    writer = RestartWriter(
        path=tmpdir,
        is_restart_step=lambda i: False,
        prognostic_names=["a", "b"],
        metadata={"a": VariableMetadata(long_name="var_a", units="m")},
        coords={"lon": lon, "lat": lat},
    )
    data = {
        "a": torch.randn(n_sample, n_time, n_lat, n_lon),
        "b": torch.randn(n_sample, n_time, n_lat, n_lon),
    }
    batch_times = xr.DataArray(
        data=np.random.uniform(size=(n_sample, n_time)),
        dims=(
            "sample",
            "time",
        ),
    )
    writer.append_batch(
        prediction=data,
        start_timestep=0,
        batch_times=batch_times,
    )
    assert not (tmpdir / "restart.nc").exists()
