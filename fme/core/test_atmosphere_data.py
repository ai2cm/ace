import datetime

import pytest
import torch

from fme.core.atmosphere_data import (
    AtmosphereData,
    _height_at_interface,
    compute_layer_thickness,
)
from fme.core.constants import LATENT_HEAT_OF_FREEZING, LATENT_HEAT_OF_VAPORIZATION
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.derived_variables import compute_derived_quantities
from fme.core.humidity import saturation_specific_humidity


def test_compute_layer_thickness():
    pressure_at_interface = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
        ]
    )
    air_temperature = torch.tensor(
        [
            [[300, 310], [300, 310]],
            [[300, 310], [300, 310]],
        ]
    )
    specific_total_water = torch.full((2, 2, 2), 0.1)
    dz = compute_layer_thickness(
        pressure_at_interface, air_temperature, specific_total_water
    )
    assert dz.shape == (2, 2, 2)
    assert torch.all(dz >= 0.0)


def test_backpropagate_through_compute_layer_thickness():
    # Regression test to ensure we can properly handle a TOA interface pressure
    # of 0.0, as is present in the ERA5 dataset. The key for this test to be
    # meaningful is the presence of zeros in the pressure_at_interface tensor.
    pressure_at_interface = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        ],
        requires_grad=True,
    )
    air_temperature = torch.tensor(
        [
            [[300.0, 310.0], [300.0, 310.0]],
            [[300.0, 310.0], [300.0, 310.0]],
        ],
        requires_grad=True,
    )
    specific_total_water = torch.full((2, 2, 2), 0.1, requires_grad=True)
    dz = compute_layer_thickness(
        pressure_at_interface, air_temperature, specific_total_water
    )
    with torch.autograd.detect_anomaly():
        # Take the sum to reduce dz to a scalar prior to backpropagation to
        # avoid the need to provide a gradient argument to backward.
        dz.sum().backward()


def test__height_at_interface():
    layer_thickness = torch.tensor([[[3, 2], [1, 0.5]], [[3, 2], [1, 0.5]]])
    height_at_surface = torch.tensor([[10, 20], [10, 20]])
    height_at_interface = _height_at_interface(layer_thickness, height_at_surface)
    assert height_at_interface.shape == (2, 2, 3)
    assert torch.equal(
        height_at_interface,
        torch.tensor(
            [[[15, 12, 10], [21.5, 20.5, 20]], [[15, 12, 10], [21.5, 20.5, 20]]]
        ),
    )


@pytest.mark.parametrize("has_water_variable", [True, False])
def test_missing_specific_total_water(has_water_variable):
    """Check shape of specific total water and make sure that it returns None
    when it can't be computed."""
    n_samples, n_time_steps, nlat, nlon = 2, 3, 4, 8

    def _get_data(water: bool):
        if water:
            return {
                "x": torch.rand(n_samples, n_time_steps, nlat, nlon),
                "specific_total_water_0": torch.rand(
                    n_samples, n_time_steps, nlat, nlon
                ),
                "specific_total_water_1": torch.rand(
                    n_samples, n_time_steps, nlat, nlon
                ),
                "PRESsfc": torch.rand(n_samples, n_time_steps, nlat, nlon),
            }
        else:
            return {
                "x": torch.rand(n_samples, n_time_steps, nlat, nlon),
                "PRESsfc": torch.rand(n_samples, n_time_steps, nlat, nlon),
            }

    atmos_data = AtmosphereData(_get_data(water=has_water_variable))

    if has_water_variable:
        assert atmos_data.specific_total_water is not None
        assert atmos_data.specific_total_water.shape == (
            n_samples,
            n_time_steps,
            nlat,
            nlon,
            2,
        )
    else:
        with pytest.raises(KeyError):
            _ = atmos_data.specific_total_water


@pytest.mark.parametrize("missing_water_layer", [True, False])
def test_keyerror_when_missing_specific_total_water_layer(missing_water_layer: bool):
    """Check shape of specific total water and make sure that it returns None
    when it can't be computed."""
    n_samples, n_time_steps, nlat, nlon = 2, 3, 4, 8

    def _get_data(missing_water_layer: bool):
        data = {
            "x": torch.rand(n_samples, n_time_steps, nlat, nlon),
            "specific_total_water_1": torch.rand(n_samples, n_time_steps, nlat, nlon),
            "PRESsfc": torch.rand(n_samples, n_time_steps, nlat, nlon),
        }
        if not missing_water_layer:
            data["specific_total_water_0"] = torch.rand(
                n_samples, n_time_steps, nlat, nlon
            )
        return data

    atmos_data = AtmosphereData(_get_data(missing_water_layer))

    if not missing_water_layer:
        assert atmos_data.specific_total_water is not None
        assert atmos_data.specific_total_water.shape == (
            n_samples,
            n_time_steps,
            nlat,
            nlon,
            2,
        )
    else:
        with pytest.raises(ValueError):
            _ = atmos_data.specific_total_water


@pytest.mark.parametrize("has_frozen_precipitation", [False, True])
def test_net_surface_energy_flux(has_frozen_precipitation):
    n_samples, n_time_steps, nlat, nlon = 2, 3, 4, 8
    shape = (n_samples, n_time_steps, nlat, nlon)

    def _get_data(
        shape: tuple[int, ...], has_frozen_precipitation: bool
    ) -> AtmosphereData:
        ones = torch.ones(shape)
        surface_pressure = {"PRESsfc": ones}
        energy_fluxes = {
            "LHTFLsfc": ones,
            "SHTFLsfc": ones,
            "ULWRFsfc": ones,
            "DLWRFsfc": ones,
            "USWRFsfc": ones,
            "DSWRFsfc": ones,
        }
        if has_frozen_precipitation:
            frozen_precipitation = {
                "GRAUPELsfc": ones / LATENT_HEAT_OF_FREEZING,
                "ICEsfc": ones / LATENT_HEAT_OF_FREEZING,
                "SNOWsfc": ones / LATENT_HEAT_OF_FREEZING,
            }
        else:
            frozen_precipitation = {}
        data = surface_pressure | energy_fluxes | frozen_precipitation
        return AtmosphereData(data)

    atmosphere_data = _get_data(shape, has_frozen_precipitation)

    if has_frozen_precipitation:
        expected = torch.full(shape, -5)
    else:
        expected = torch.full(shape, -2)

    result = atmosphere_data.net_surface_energy_flux
    torch.testing.assert_close(result, expected, check_dtype=False)


def test_net_energy_flux_into_atmosphere():
    n_samples, n_time_steps, nlat, nlon = 2, 3, 4, 8
    shape = (n_samples, n_time_steps, nlat, nlon)

    def _get_data(shape: tuple[int, ...]) -> AtmosphereData:
        ones = torch.ones(shape, dtype=torch.float32)
        surface_pressure = {"PRESsfc": ones}
        energy_fluxes = {
            "LHTFLsfc": ones,
            "SHTFLsfc": ones,
            "ULWRFsfc": ones,
            "DLWRFsfc": ones,
            "USWRFsfc": ones,
            "DSWRFsfc": ones,
            "ULWRFtoa": ones,
            "USWRFtoa": ones,
            "DSWRFtoa": ones,
        }
        data = surface_pressure | energy_fluxes
        return AtmosphereData(data)

    atmosphere_data = _get_data(shape)

    expected = torch.full(shape, 1.0, dtype=torch.float32)

    result = atmosphere_data.net_energy_flux_into_atmosphere
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("fields", "expected_sum"),
    [
        (("GRAUPELsfc", "ICEsfc", "SNOWsfc"), 3),
        (("GRAUPELsfc", "ICEsfc"), 0),
        (("total_frozen_precipitation_rate",), 1),
        ((), 0),
    ],
    ids=lambda x: f"{x!r}",
)
def test_frozen_precipitation_rate(fields, expected_sum):
    n_samples, n_time_steps, nlat, nlon = 2, 3, 4, 8
    shape = (n_samples, n_time_steps, nlat, nlon)

    def _get_data(shape: tuple[int, ...]) -> AtmosphereData:
        ones = torch.ones(shape)
        surface_pressure = {"PRESsfc": ones}
        frozen_precipitation = {}
        for field in fields:
            frozen_precipitation[field] = ones
        data = surface_pressure | frozen_precipitation
        return AtmosphereData(data)

    atmosphere_data = _get_data(shape)
    expected = torch.full(shape, expected_sum)

    result = atmosphere_data.frozen_precipitation_rate
    torch.testing.assert_close(result, expected, check_dtype=False)


def test_modified_data_empty_before_any_write():
    shape = (2, 4)
    atmosphere_data = AtmosphereData({"PRESsfc": torch.ones(shape)})
    assert atmosphere_data.modified_data == {}


def test_modified_data_records_concrete_key():
    shape = (2, 4)
    new_pressure = 2 * torch.ones(shape)
    atmosphere_data = AtmosphereData({"PRESsfc": torch.ones(shape)})
    atmosphere_data.set_surface_pressure(new_pressure)
    modified = atmosphere_data.modified_data
    assert set(modified) == {"PRESsfc"}
    torch.testing.assert_close(modified["PRESsfc"], new_pressure)


def test_modified_data_records_storage_key_not_standard_name():
    # set_evaporation_rate writes the latent heat flux field, so the recorded
    # key must be the concrete storage key, not "evaporation_rate".
    shape = (2, 4)
    atmosphere_data = AtmosphereData({"LHTFLsfc": torch.ones(shape)})
    evaporation = 3 * torch.ones(shape)
    atmosphere_data.set_evaporation_rate(evaporation)
    modified = atmosphere_data.modified_data
    assert set(modified) == {"LHTFLsfc"}
    torch.testing.assert_close(
        modified["LHTFLsfc"], evaporation * LATENT_HEAT_OF_VAPORIZATION
    )


def test_modified_data_records_multiple_writes():
    shape = (2, 4)
    atmosphere_data = AtmosphereData(
        {
            "PRESsfc": torch.ones(shape),
            "PRATEsfc": torch.ones(shape),
            "tendency_of_total_water_path_due_to_advection": torch.ones(shape),
        }
    )
    atmosphere_data.set_precipitation_rate(2 * torch.ones(shape))
    atmosphere_data.set_tendency_of_total_water_path_due_to_advection(
        3 * torch.ones(shape)
    )
    modified = atmosphere_data.modified_data
    assert set(modified) == {
        "PRATEsfc",
        "tendency_of_total_water_path_due_to_advection",
    }


def test_modified_data_returns_references():
    # The contract is out-of-place correction: modified_data returns the live
    # tensors, not clones, so the caller diffs them against its own snapshot.
    shape = (2, 4)
    new_pressure = 2 * torch.ones(shape)
    atmosphere_data = AtmosphereData({"PRESsfc": torch.ones(shape)})
    atmosphere_data.set_surface_pressure(new_pressure)
    assert atmosphere_data.modified_data["PRESsfc"] is atmosphere_data.data["PRESsfc"]


def test_windspeed_at_10m():
    shape = (2, 4)
    data_dict = {"UGRD10m": torch.ones(shape), "VGRD10m": -2 * torch.ones(shape)}
    atmosphere_data = AtmosphereData(data_dict)
    expected = torch.sqrt(torch.full(shape, 5))
    result = atmosphere_data.windspeed_at_10m
    torch.testing.assert_close(result, expected)


def test_specific_total_water_reconstructed_from_relative_humidity():
    torch.manual_seed(0)
    n_levels = 3
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.tensor([0.0, 5000.0, 10000.0, 0.0]),
        bk=torch.tensor([0.0, 0.1, 0.5, 1.0]),
    )
    shape = (2, 4, 3, 5)
    surface_pressure = 9.0e4 + 1.0e4 * torch.rand(shape)
    temperature = {
        f"air_temperature_{k}": 220.0 + 30.0 * k + 10.0 * torch.rand(shape)
        for k in range(n_levels)
    }
    q = {
        f"specific_total_water_{k}": 1e-3 * (k + 1) * torch.rand(shape)
        for k in range(n_levels)
    }
    q_data = {"PRESsfc": surface_pressure, **temperature, **q}
    interface = vertical_coordinate.interface_pressure(surface_pressure)
    layer = 0.5 * (interface[..., :-1] + interface[..., 1:])
    rh = {
        f"relative_humidity_{k}": q[f"specific_total_water_{k}"]
        / saturation_specific_humidity(
            temperature[f"air_temperature_{k}"], layer[..., k]
        )
        for k in range(n_levels)
    }
    rh_data = {"PRESsfc": surface_pressure, **temperature, **rh}
    expected = AtmosphereData(q_data, vertical_coordinate)
    reconstructed = AtmosphereData(rh_data, vertical_coordinate)
    torch.testing.assert_close(
        reconstructed.specific_total_water, expected.specific_total_water
    )
    torch.testing.assert_close(
        reconstructed.total_water_path, expected.total_water_path
    )
    derived = compute_derived_quantities(
        rh_data, vertical_coordinate, datetime.timedelta(days=1)
    )
    torch.testing.assert_close(derived["total_water_path"], expected.total_water_path)
    # no vertical coordinate: q cannot be reconstructed, reported as missing
    with pytest.raises(KeyError):
        AtmosphereData(rh_data).specific_total_water
