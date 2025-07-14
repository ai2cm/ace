import dataclasses
import datetime
from collections.abc import Callable

import dacite
import numpy as np
import pytest
import torch

from fme.core import AtmosphereData, metrics
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.derived_variables import total_water_path_budget_residual
from fme.core.gridded_ops import GriddedOperations, HEALPixOperations, LatLonOperations
from fme.core.typing_ import TensorMapping

from .atmosphere import (
    AtmosphereCorrector,
    AtmosphereCorrectorConfig,
    EnergyBudgetConfig,
    _force_conserve_dry_air,
    _force_conserve_moisture,
    _force_conserve_total_energy,
    _force_zero_global_mean_moisture_advection,
)
from .utils import force_positive

TIMESTEP = datetime.timedelta(hours=6)


def test_config_dataclass_round_trip():
    config = AtmosphereCorrectorConfig()
    new_config = dacite.from_dict(
        data_class=AtmosphereCorrectorConfig,
        data=dataclasses.asdict(config),
        config=dacite.Config(strict=True),
    )
    assert config == new_config


def compute_dry_air_absolute_differences(
    atmosphere_data: AtmosphereData,
    area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Computes the absolute value of the dry air tendency of each time step.

    Args:
        atmosphere_data: AtmosphereData object.
        area_weighted_mean: Function which returns an area-weighted mean.
        vertical_coordinate: The vertical coordinate of the model.

    Returns:
        A tensor of shape (time,) of the absolute value of the dry air tendency
            of each time step.
    """
    try:
        surface_pressure = atmosphere_data.surface_pressure
        total_water_path = atmosphere_data.total_water_path
    except KeyError:
        return torch.tensor([torch.nan])
    ps_dry = metrics.surface_pressure_due_to_dry_air(surface_pressure, total_water_path)
    ps_dry_mean = area_weighted_mean(ps_dry)
    return ps_dry_mean.diff(dim=-1).abs().mean(dim=0)


def get_dry_air_nonconservation(
    data: TensorMapping,
    area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
    vertical_coordinate: HybridSigmaPressureCoordinate,
):
    """
    Computes the time-average one-step absolute difference in surface pressure due to
    changes in globally integrated dry air.

    Args:
        data: A mapping from variable name to tensor of shape
            [sample, time, lat, lon], in physical units. specific_total_water in kg/kg
            and surface_pressure in Pa must be present.
        area_weighted_mean: Computes the area-weighted mean of a tensor, removing the
            horizontal dimensions.
        vertical_coordinate: The vertical coordinates of the model.
    """
    return compute_dry_air_absolute_differences(
        AtmosphereData(data, vertical_coordinate),
        area_weighted_mean=area_weighted_mean,
    ).mean()


def test_force_no_global_mean_moisture_advection():
    torch.random.manual_seed(0)
    data = {
        "tendency_of_total_water_path_due_to_advection": torch.rand(size=(3, 2, 5, 5)),
    }
    area_weights = 1.0 + torch.rand(size=(5, 1)).broadcast_to(size=(5, 5))
    original_mean = metrics.weighted_mean(
        data["tendency_of_total_water_path_due_to_advection"],
        weights=area_weights,
        dim=[-2, -1],
    )
    assert (original_mean.abs() > 0.0).all()
    fixed_data = _force_zero_global_mean_moisture_advection(
        data,
        area_weighted_mean=LatLonOperations(area_weights).area_weighted_mean,
    )
    new_mean = metrics.weighted_mean(
        fixed_data["tendency_of_total_water_path_due_to_advection"],
        weights=area_weights,
        dim=[-2, -1],
    )
    assert (new_mean.abs() < original_mean.abs()).all()
    np.testing.assert_almost_equal(new_mean.cpu().numpy(), 0.0, decimal=6)


@pytest.mark.parametrize(
    "size, use_area",
    [
        pytest.param((3, 2, 5, 5), True, id="latlon"),
        pytest.param((3, 12, 2, 3, 3), False, id="healpix"),
    ],
)
def test_force_conserve_dry_air(size: tuple[int, ...], use_area: bool):
    torch.random.manual_seed(0)
    data = {
        "PRESsfc": 10.0 + torch.rand(size=size),
        "specific_total_water_0": torch.rand(size=size),
        "specific_total_water_1": torch.rand(size=size),
    }
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.asarray([3.0, 1.0, 0.0]), bk=torch.asarray([0.0, 0.6, 1.0])
    )
    if use_area:
        area_weights: torch.Tensor | None = 1.0 + torch.rand(
            size=(size[-2], 1)
        ).broadcast_to(size=size[-2:])
    else:
        area_weights = None
    if area_weights is not None:
        gridded_operations: GriddedOperations = LatLonOperations(area_weights)
    else:
        gridded_operations = HEALPixOperations()
    original_nonconservation = get_dry_air_nonconservation(
        data,
        vertical_coordinate=vertical_coordinate,
        area_weighted_mean=gridded_operations.area_weighted_mean,
    )
    assert original_nonconservation > 0.0
    in_data = {k: v.select(dim=1, index=0) for k, v in data.items()}
    out_data = {k: v.select(dim=1, index=1) for k, v in data.items()}
    fixed_out_data = _force_conserve_dry_air(
        in_data,
        out_data,
        vertical_coordinate=vertical_coordinate,
        area_weighted_mean=gridded_operations.area_weighted_mean,
    )
    new_data = {
        k: torch.stack([v, fixed_out_data[k]], dim=1) for k, v in in_data.items()
    }
    new_nonconservation = get_dry_air_nonconservation(
        new_data,
        vertical_coordinate=vertical_coordinate,
        area_weighted_mean=gridded_operations.area_weighted_mean,
    )
    assert new_nonconservation < original_nonconservation
    np.testing.assert_almost_equal(new_nonconservation.cpu().numpy(), 0.0, decimal=6)


@pytest.mark.parametrize("dataset", ["fv3", "e3sm"])
@pytest.mark.parametrize(
    "global_only, terms_to_modify",
    [
        (True, "precipitation"),
        (True, "evaporation"),
        (False, "advection_and_precipitation"),
        (False, "advection_and_evaporation"),
    ],
)
@pytest.mark.parametrize(
    "size, use_area",
    [
        pytest.param((3, 2, 5, 5), True, id="latlon"),
        pytest.param((3, 12, 2, 3, 3), False, id="healpix"),
    ],
)
def test_force_conserve_moisture(
    dataset: str,
    global_only: bool,
    terms_to_modify,
    size: tuple[int, ...],
    use_area: bool,
):
    torch.random.manual_seed(0)
    if dataset == "fv3":
        data = {
            "PRESsfc": 10.0 + torch.rand(size=size),
            "specific_total_water_0": torch.rand(size=size),
            "specific_total_water_1": torch.rand(size=size),
            "PRATEsfc": torch.rand(size=size),
            "LHTFLsfc": torch.rand(size=size),
            "tendency_of_total_water_path_due_to_advection": torch.rand(size=size),
        }
    if dataset == "e3sm":
        data = {
            "PS": 10.0 + torch.rand(size=size),
            "specific_total_water_0": torch.rand(size=size),
            "specific_total_water_1": torch.rand(size=size),
            "surface_precipitation_rate": torch.rand(size=size),
            "LHFLX": torch.rand(size=size),
            "tendency_of_total_water_path_due_to_advection": torch.rand(size=size),
        }
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.asarray([3.0, 1.0, 0.0]), bk=torch.asarray([0.0, 0.6, 1.0])
    )
    if use_area:
        ops: GriddedOperations = LatLonOperations(
            1.0 + torch.rand(size=(5, 1)).broadcast_to(size=(5, 5))
        )
    else:
        ops = HEALPixOperations()
    data["tendency_of_total_water_path_due_to_advection"] -= ops.area_weighted_mean(
        data["tendency_of_total_water_path_due_to_advection"], keepdim=True
    )
    original_budget_residual = total_water_path_budget_residual(
        AtmosphereData(data, vertical_coordinate),
        timestep=TIMESTEP,
    )[:, 1]  # no meaning for initial value data, want first timestep
    if global_only:
        original_budget_residual = ops.area_weighted_mean(
            original_budget_residual, keepdim=True
        )
    original_budget_residual = original_budget_residual.cpu().numpy()
    original_dry_air = (
        AtmosphereData(data, vertical_coordinate)
        .surface_pressure_due_to_dry_air.cpu()
        .numpy()
    )
    assert np.any(np.abs(original_budget_residual) > 0.0)
    in_data = {k: v.select(dim=1, index=0) for k, v in data.items()}
    out_data = {k: v.select(dim=1, index=1) for k, v in data.items()}
    fixed_out_data = _force_conserve_moisture(
        in_data,
        out_data,
        vertical_coordinate=vertical_coordinate,
        area_weighted_mean=ops.area_weighted_mean,
        timestep_seconds=TIMESTEP.total_seconds(),
        terms_to_modify=terms_to_modify,
    )
    new_data = {
        k: torch.stack([v, fixed_out_data[k]], dim=1) for k, v in in_data.items()
    }
    new_budget_residual = total_water_path_budget_residual(
        AtmosphereData(new_data, vertical_coordinate),
        timestep=TIMESTEP,
    )[:, 1]  # no meaning for initial value data, want first timestep
    new_dry_air = (
        AtmosphereData(data, vertical_coordinate)
        .surface_pressure_due_to_dry_air.cpu()
        .numpy()
    )

    global_budget_residual = ops.area_weighted_mean(new_budget_residual).cpu().numpy()
    np.testing.assert_almost_equal(global_budget_residual, 0.0, decimal=6)

    if not global_only:
        new_budget_residual = new_budget_residual.cpu().numpy()
        assert np.all(np.abs(new_budget_residual) < np.abs(original_budget_residual))
        np.testing.assert_almost_equal(new_budget_residual, 0.0, decimal=6)

    np.testing.assert_almost_equal(new_dry_air, original_dry_air, decimal=6)


def test_force_positive():
    data = {
        "foo": torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]]),
        "bar": torch.tensor([[-1.0, 0.0], [0.0, -3.0], [1.0, 2.0]]),
    }
    original_min = torch.min(data["foo"])
    assert original_min < 0.0
    fixed_data = force_positive(data, ["foo"])
    new_min = torch.min(fixed_data["foo"])
    # Ensure the minimum value of 'foo' is now 0
    torch.testing.assert_close(new_min, torch.tensor(0.0))
    # Ensure other variables are not modified
    torch.testing.assert_close(fixed_data["bar"], data["bar"])


def _get_corrector_test_input(tensor_shape):
    """Generate test input that has necessary variables for all correctors."""
    # debugging a bit easier with realistic scales for input variables
    # tuples of (name, mean, std) for each variable
    variables = [
        ("PRESsfc", 100000, 1000),
        ("HGTsfc", 500, 50),
        ("SHTFLsfc", 100, 50),
        ("LHTFLsfc", 100, 50),
        ("specific_total_water_0", 0.001, 0.0001),
        ("specific_total_water_1", 0.001, 0.0001),
        ("air_temperature_0", 300, 10),
        ("air_temperature_1", 300, 10),
        ("DSWRFtoa", 500, 100),
        ("USWRFtoa", 500, 100),
        ("ULWRFtoa", 500, 100),
        ("USWRFsfc", 500, 100),
        ("DSWRFsfc", 500, 100),
        ("ULWRFsfc", 500, 100),
        ("DLWRFsfc", 500, 100),
        ("PRATEsfc", 1e-5, 1e-6),
        ("tendency_of_total_water_path_due_to_advection", 1e-5, 1e-6),
    ]
    forcing_names = ["HGTsfc", "DSWRFtoa"]
    non_forcing_names = [name for name, _, _ in variables if name not in forcing_names]
    input_data = {
        name: centering + scale * torch.rand(size=tensor_shape)
        for name, centering, scale in variables
    }
    gen_data = {
        name: centering + scale * torch.rand(size=tensor_shape)
        for name, centering, scale in variables
        if name in non_forcing_names
    }
    forcing_data = {
        name: centering + scale * torch.rand(size=tensor_shape)
        for name, centering, scale in variables
        if name in forcing_names
    }
    vertical_coord = HybridSigmaPressureCoordinate(
        ak=torch.asarray([3.0, 1.0, 0.0]), bk=torch.asarray([0.0, 0.6, 1.0])
    )
    return input_data, gen_data, forcing_data, vertical_coord


@pytest.mark.parametrize("negative_pressure", [True, False])
def test__force_conserve_total_energy(negative_pressure: bool):
    tensor_shape = (5, 5)

    ops = LatLonOperations(
        0.5 + torch.rand(size=(tensor_shape[-2], 1)).broadcast_to(size=tensor_shape)
    )
    timestep = datetime.timedelta(seconds=3600)
    input_data, gen_data, forcing_data, vertical_coord = _get_corrector_test_input(
        tensor_shape
    )
    extra_heating = 10.0

    if negative_pressure:
        input_data["PRESsfc"] = -1 * input_data["PRESsfc"]

    correction_expected = not negative_pressure

    corrected_gen_data = _force_conserve_total_energy(
        input_data=input_data,
        gen_data=gen_data,
        forcing_data=forcing_data,
        area_weighted_mean=ops.area_weighted_mean,
        vertical_coordinate=vertical_coord,
        timestep_seconds=timestep.total_seconds(),
        unaccounted_heating=extra_heating,
    )

    # ensure only temperature is modified
    for name in gen_data:
        if "air_temperature" in name:
            if correction_expected:
                assert not torch.allclose(
                    corrected_gen_data[name], gen_data[name], rtol=1e-6
                )
            else:
                assert torch.allclose(
                    corrected_gen_data[name], gen_data[name], rtol=1e-6
                )
        else:
            torch.testing.assert_close(corrected_gen_data[name], gen_data[name])

    # ensure forcing variables are not in the corrected data
    for name in forcing_data:
        assert name not in corrected_gen_data

    # ensure the corrected global mean MSE path is what we expect
    input = AtmosphereData(input_data, vertical_coord)
    corrected_gen = AtmosphereData(corrected_gen_data | forcing_data, vertical_coord)
    input_gm_mse = ops.area_weighted_mean(input.total_energy_ace2_path)
    corrected_gen_gm_mse = ops.area_weighted_mean(corrected_gen.total_energy_ace2_path)
    predicted_mse_tendency = (
        ops.area_weighted_mean(corrected_gen.net_energy_flux_into_atmosphere)
        + extra_heating
    )
    if correction_expected:
        expected_gm_mse = (
            input_gm_mse + predicted_mse_tendency * timestep.total_seconds()
        )
        torch.testing.assert_close(corrected_gen_gm_mse, expected_gm_mse)

    # ensure the temperature correction is constant
    corrected_gen_temperature = corrected_gen_data["air_temperature_1"]
    initial_gen_temperature = gen_data["air_temperature_1"]
    temperature_1_correction = corrected_gen_temperature - initial_gen_temperature
    assert torch.all(torch.eq(temperature_1_correction, temperature_1_correction[0, 0]))
    temperature_correction_0 = (
        corrected_gen_data["air_temperature_0"] - gen_data["air_temperature_0"]
    )
    torch.testing.assert_close(temperature_correction_0, temperature_1_correction)


def test__force_conserve_energy_doesnt_clobber():
    tensor_shape = (5, 5)

    ops = LatLonOperations(
        0.5 + torch.rand(size=(tensor_shape[-2], 1)).broadcast_to(size=tensor_shape)
    )
    timestep = datetime.timedelta(seconds=3600)
    input_data, gen_data, forcing_data, vertical_coord = _get_corrector_test_input(
        tensor_shape
    )
    # add a prognostic variable to the forcing data
    forcing_data["PRESsfc"] = 10.0 + torch.rand(size=tensor_shape)

    corrected_gen_data = _force_conserve_total_energy(
        input_data=input_data,
        gen_data=gen_data,
        forcing_data=forcing_data,
        area_weighted_mean=ops.area_weighted_mean,
        vertical_coordinate=vertical_coord,
        timestep_seconds=timestep.total_seconds(),
    )
    torch.testing.assert_close(corrected_gen_data["PRESsfc"], gen_data["PRESsfc"])


def test_corrector_integration():
    """Ensures that the corrector can be called with all methods active
    but doesn't check results."""
    config = AtmosphereCorrectorConfig(
        conserve_dry_air=True,
        zero_global_mean_moisture_advection=True,
        moisture_budget_correction="advection_and_precipitation",
        force_positive_names=["PRESsfc"],
        total_energy_budget_correction=EnergyBudgetConfig("constant_temperature", 1.0),
    )
    tensor_shape = (5, 5)
    test_input = _get_corrector_test_input(tensor_shape)
    input_data, gen_data, forcing_data, vertical_coord = test_input
    ops = LatLonOperations(
        0.5 + torch.rand(size=(tensor_shape[-2], 1)).broadcast_to(size=tensor_shape)
    )
    timestep = datetime.timedelta(seconds=3600)
    corrector = AtmosphereCorrector(config, ops, vertical_coord, timestep)
    corrector(input_data, gen_data, forcing_data)
