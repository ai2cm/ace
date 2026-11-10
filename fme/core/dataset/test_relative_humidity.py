import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset.relative_humidity import (
    RELATIVE_HUMIDITY_METADATA,
    DerivedRelativeHumidity,
)
from fme.core.dataset.schedule import IntSchedule
from fme.core.dataset.xarray import XarrayDataConfig, XarrayDataset
from fme.core.humidity import saturation_specific_humidity

N_TIME, N_LAT, N_LON = 4, 3, 5
AK = [0.0, 5000.0, 0.0]
BK = [0.0, 0.3, 1.0]


def _expected_rh(t, q, ps, k):
    p_top = AK[k] + BK[k] * ps
    p_bottom = AK[k + 1] + BK[k + 1] * ps
    qsat = saturation_specific_humidity(t, 0.5 * (p_top + p_bottom))
    return q / qsat


@pytest.fixture
def humidity_data_path(tmp_path):
    rng = np.random.default_rng(0)
    shape = (N_TIME, N_LAT, N_LON)
    dims = ("time", "lat", "lon")
    data_vars: dict = {f"ak_{i}": AK[i] for i in range(3)}
    data_vars.update({f"bk_{i}": BK[i] for i in range(3)})
    data_vars["PRESsfc"] = (dims, rng.uniform(9.0e4, 1.02e5, shape).astype("f4"))
    for k, (t_mean, q_mean) in enumerate([(230.0, 1e-4), (285.0, 8e-3)]):
        data_vars[f"air_temperature_{k}"] = (
            dims,
            rng.uniform(t_mean - 10, t_mean + 10, shape).astype("f4"),
            {"units": "K"},
        )
        data_vars[f"specific_total_water_{k}"] = (
            dims,
            rng.uniform(0.2 * q_mean, 1.5 * q_mean, shape).astype("f4"),
            {"units": "kg/kg"},
        )
    time = xr.date_range("2000-01-01", periods=N_TIME, freq="D", use_cftime=True)
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time,
            "lat": np.linspace(-60, 60, N_LAT, dtype="f4"),
            "lon": np.linspace(0, 288, N_LON, dtype="f4"),
        },
    )
    ds.to_netcdf(tmp_path / "data.nc", unlimited_dims=["time"], format="NETCDF4")
    return tmp_path, ds


def _build(path, names, derive=True):
    config = XarrayDataConfig(data_path=str(path), derive_relative_humidity=derive)
    return XarrayDataset(config, names, IntSchedule.from_constant(2))


def test_derived_relative_humidity_matches_formula(humidity_data_path):
    path, ds = humidity_data_path
    names = ["relative_humidity_0", "relative_humidity_1", "air_temperature_1"]
    dataset = _build(path, names)
    tensors = dataset.get_sample_by_time_slice(slice(0, N_TIME))[0]
    # only requested names are returned; derivation-only sources are dropped
    assert set(tensors) == set(names)
    ps = torch.as_tensor(ds["PRESsfc"].values)
    for k in range(2):
        expected = _expected_rh(
            torch.as_tensor(ds[f"air_temperature_{k}"].values),
            torch.as_tensor(ds[f"specific_total_water_{k}"].values),
            ps,
            k,
        )
        torch.testing.assert_close(tensors[f"relative_humidity_{k}"], expected)
        assert tensors[f"relative_humidity_{k}"].dtype == torch.float32
    torch.testing.assert_close(
        tensors["air_temperature_1"],
        torch.as_tensor(ds["air_temperature_1"].values),
    )
    metadata = dataset.properties.variable_metadata
    assert metadata["relative_humidity_0"] == RELATIVE_HUMIDITY_METADATA
    assert "PRESsfc" not in metadata
    assert metadata["air_temperature_1"].units == "K"


def test_derived_relative_humidity_not_clipped(humidity_data_path):
    path, ds = humidity_data_path
    ds = ds.copy()
    ds["specific_total_water_1"] = ds["specific_total_water_1"] * 0 + 0.5
    ds["specific_total_water_0"] = ds["specific_total_water_0"] * 0 - 1e-5
    ds.to_netcdf(path / "data.nc", unlimited_dims=["time"], format="NETCDF4")
    dataset = _build(path, ["relative_humidity_0", "relative_humidity_1"])
    tensors = dataset.get_sample_by_time_slice(slice(0, N_TIME))[0]
    assert (tensors["relative_humidity_1"] > 1).all()
    assert (tensors["relative_humidity_0"] < 0).all()


def test_requested_source_fields_are_kept(humidity_data_path):
    path, _ = humidity_data_path
    names = ["relative_humidity_0", "PRESsfc", "specific_total_water_0"]
    dataset = _build(path, names)
    tensors = dataset.get_sample_by_time_slice(slice(0, 2))[0]
    assert set(tensors) == set(names)


def test_relative_humidity_requires_flag(humidity_data_path):
    path, _ = humidity_data_path
    with pytest.raises(ValueError, match="relative_humidity_0"):
        _build(path, ["relative_humidity_0"], derive=False)


def test_flag_without_relative_humidity_is_noop(humidity_data_path):
    path, ds = humidity_data_path
    dataset = _build(path, ["air_temperature_0"])
    tensors = dataset.get_sample_by_time_slice(slice(0, 2))[0]
    assert set(tensors) == {"air_temperature_0"}


def test_relative_humidity_level_beyond_vertical_coordinate(humidity_data_path):
    path, _ = humidity_data_path
    with pytest.raises(ValueError):
        _build(path, ["relative_humidity_2"])


def test_from_requested_names():
    derived = DerivedRelativeHumidity.from_requested_names(
        ["relative_humidity_3", "relative_humidity_1", "air_temperature_1", "foo"]
    )
    assert derived.levels == (1, 3)
    assert derived.names == ["relative_humidity_1", "relative_humidity_3"]
    assert derived.extra_names == {
        "PRESsfc",
        "specific_total_water_1",
        "air_temperature_3",
        "specific_total_water_3",
    }
    assert set(
        derived.load_names(["relative_humidity_3", "relative_humidity_1", "foo"])
    ) == {
        "foo",
        "PRESsfc",
        "air_temperature_1",
        "specific_total_water_1",
        "air_temperature_3",
        "specific_total_water_3",
    }


def test_validate_rejects_non_hybrid_coordinate():
    from fme.core.coordinates import DepthCoordinate

    derived = DerivedRelativeHumidity.from_requested_names(["relative_humidity_0"])
    coordinate = DepthCoordinate(
        idepth=torch.tensor([0.0, 1.0]), mask=torch.ones(1, 1, 1)
    )
    with pytest.raises(ValueError, match="hybrid sigma-pressure"):
        derived.validate(coordinate)
    derived.validate(
        HybridSigmaPressureCoordinate(ak=torch.tensor(AK), bk=torch.tensor(BK))
    )
