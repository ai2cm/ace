import dataclasses
import datetime
import math
import pathlib

import dacite
import numpy as np
import pytest
import torch
import xarray as xr

from fme.core import AtmosphereData
from fme.core.coordinates import NullVerticalCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import LatLonOperations
from fme.core.registry.corrector import CorrectorSelector

from .atmosphere import AtmosphereCorrectorConfig, EnergyBudgetConfig
from .co2_relaxation import (
    CO2_NAME_ATTR,
    CO2_REF_ATTR,
    CO2RegressionRelaxationConfig,
    intercept_name,
    slope_name,
)
from .test_atmosphere import _get_corrector_test_input

TIMESTEP = datetime.timedelta(hours=6)
CO2_NAME = "global_mean_co2"
CO2_REF = 400e-6


def _loaded_config(
    timescale_days: dict[str, float],
    intercept: dict[str, torch.Tensor],
    slope: dict[str, torch.Tensor],
    co2_ref: float = CO2_REF,
) -> CO2RegressionRelaxationConfig:
    return CO2RegressionRelaxationConfig(
        timescale_days=timescale_days,
        co2_name=CO2_NAME,
        intercept=intercept,
        slope=slope,
        co2_ref=co2_ref,
    )


def _write_reference(
    path: pathlib.Path,
    intercept: dict[str, np.ndarray],
    slope: dict[str, np.ndarray],
    co2_ref: float = CO2_REF,
):
    data_vars = {}
    for name in intercept:
        data_vars[intercept_name(name)] = (["y", "x"], intercept[name])
        data_vars[slope_name(name)] = (["y", "x"], slope[name])
    ds = xr.Dataset(data_vars, attrs={CO2_NAME_ATTR: CO2_NAME, CO2_REF_ATTR: co2_ref})
    ds.to_netcdf(path)


def _co2(shape, value: float) -> dict[str, torch.Tensor]:
    return {CO2_NAME: torch.full(shape, value)}


def test_relaxation_exponential_update():
    shape = (2, 3, 4)
    torch.manual_seed(0)
    intercept = torch.rand(shape[-2:])
    slope = torch.rand(shape[-2:])
    config = _loaded_config({"a": 2.0}, {"a": intercept}, {"a": slope})
    relaxation = config.build(TIMESTEP)
    relaxation.train(False)
    x = torch.rand(shape)
    co2 = 2 * CO2_REF
    relaxed, _ = relaxation({"a": x}, {"a": x}, _co2(shape, co2), None)
    target = intercept + slope * math.log(2.0)
    decay = math.exp(-TIMESTEP.total_seconds() / (2.0 * 86400.0))
    torch.testing.assert_close(relaxed["a"], target + (x - target) * decay)


def test_relaxation_spatially_varying_co2():
    shape = (2, 3, 4)
    intercept = torch.zeros(shape[-2:])
    slope = torch.ones(shape[-2:])
    config = _loaded_config({"a": 1.0}, {"a": intercept}, {"a": slope})
    relaxation = config.build(TIMESTEP)
    relaxation.train(False)
    x = torch.zeros(shape)
    co2 = CO2_REF * (1.0 + torch.rand(shape))
    relaxed, _ = relaxation({"a": x}, {"a": x}, {CO2_NAME: co2}, None)
    decay = math.exp(-0.25)
    torch.testing.assert_close(relaxed["a"], torch.log(co2 / CO2_REF) * (1 - decay))


def test_relaxation_short_timescale_is_bounded_and_preserves_positivity():
    shape = (2, 3, 4)
    target_value = 1e-3
    config = _loaded_config(
        {"q": 0.01},  # much shorter than the 6-hour timestep
        {"q": torch.full(shape[-2:], target_value)},
        {"q": torch.zeros(shape[-2:])},
    )
    relaxation = config.build(TIMESTEP)
    relaxation.train(False)
    x = 10.0 * torch.rand(shape) + 1e-8
    relaxed, _ = relaxation({"q": x}, {"q": x}, _co2(shape, 3 * CO2_REF), None)
    lower = torch.minimum(x, torch.tensor(target_value))
    upper = torch.maximum(x, torch.tensor(target_value))
    assert torch.all(relaxed["q"] >= lower)
    assert torch.all(relaxed["q"] <= upper)
    assert torch.all(relaxed["q"] > 0)
    # tau << dt relaxes (nearly) all the way to the target
    torch.testing.assert_close(relaxed["q"], torch.full(shape, target_value))


def test_relaxation_skipped_in_train_mode():
    shape = (2, 3, 4)
    config = _loaded_config(
        {"a": 1.0}, {"a": torch.ones(shape[-2:])}, {"a": torch.ones(shape[-2:])}
    )
    relaxation = config.build(TIMESTEP)
    x = torch.zeros(shape)
    # train mode is the initial mode
    relaxed, _ = relaxation({"a": x}, {"a": x}, {}, None)
    assert relaxed == {}
    relaxation.train(False)
    relaxed, _ = relaxation({"a": x}, {"a": x}, _co2(shape, CO2_REF), None)
    assert set(relaxed) == {"a"}
    relaxation.train(True)
    relaxed, _ = relaxation({"a": x}, {"a": x}, _co2(shape, CO2_REF), None)
    assert relaxed == {}


def test_relaxation_missing_co2_raises():
    shape = (2, 3, 4)
    config = _loaded_config(
        {"a": 1.0}, {"a": torch.ones(shape[-2:])}, {"a": torch.ones(shape[-2:])}
    )
    relaxation = config.build(TIMESTEP)
    relaxation.train(False)
    x = torch.zeros(shape)
    with pytest.raises(ValueError, match=f"CO2 forcing variable '{CO2_NAME}'"):
        relaxation({"a": x}, {"a": x}, {"other": x}, None)


def test_relaxation_shape_mismatch_raises():
    shape = (2, 3, 4)
    config = _loaded_config(
        {"a": 1.0}, {"a": torch.ones(4, 3)}, {"a": torch.ones(4, 3)}
    )
    relaxation = config.build(TIMESTEP)
    relaxation.train(False)
    x = torch.zeros(shape)
    with pytest.raises(ValueError, match="horizontal shape"):
        relaxation({"a": x}, {"a": x}, _co2(shape, CO2_REF), None)


def test_relaxation_non_prognostic_field_raises():
    shape = (2, 3, 4)
    config = _loaded_config(
        {"a": 1.0}, {"a": torch.ones(3, 4)}, {"a": torch.ones(3, 4)}
    )
    relaxation = config.build(TIMESTEP)
    relaxation.train(False)
    x = torch.zeros(shape)
    # "a" is generated but not in the input, so it is a diagnostic
    with pytest.raises(ValueError, match="generated prognostic"):
        relaxation({}, {"a": x}, _co2(shape, CO2_REF), None)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "Exactly one of"),
        ({"path": "ref.nc", "co2_ref": 1.0}, "must be given together"),
        (
            {
                "path": "ref.nc",
                "intercept": {"a": torch.ones(1)},
                "slope": {"a": torch.ones(1)},
                "co2_ref": 1.0,
            },
            "Exactly one of",
        ),
        (
            {
                "intercept": {"b": torch.ones(1)},
                "slope": {"b": torch.ones(1)},
                "co2_ref": 1.0,
            },
            "do not match",
        ),
    ],
)
def test_relaxation_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CO2RegressionRelaxationConfig(
            timescale_days={"a": 1.0}, co2_name=CO2_NAME, **kwargs
        )


def test_relaxation_config_rejects_non_positive_timescale():
    with pytest.raises(ValueError, match="positive"):
        CO2RegressionRelaxationConfig(
            timescale_days={"a": 0.0}, co2_name=CO2_NAME, path="ref.nc"
        )


def test_relaxation_config_load_from_file(tmp_path: pathlib.Path):
    intercept = np.random.rand(3, 4).astype(np.float32)
    slope = np.random.rand(3, 4).astype(np.float32)
    path = tmp_path / "ref.nc"
    _write_reference(path, {"a": intercept}, {"a": slope}, co2_ref=350e-6)
    config = CO2RegressionRelaxationConfig(
        timescale_days={"a": 1.0}, co2_name=CO2_NAME, path=str(path)
    )
    config.load()
    assert config.path is None
    assert config.co2_ref == pytest.approx(350e-6)
    assert config.intercept is not None and config.slope is not None
    np.testing.assert_array_equal(config.intercept["a"].numpy(), intercept)
    np.testing.assert_array_equal(config.slope["a"].numpy(), slope)
    # the loaded form round-trips through strict dacite
    state = dataclasses.asdict(config)
    reloaded = dacite.from_dict(
        CO2RegressionRelaxationConfig, state, config=dacite.Config(strict=True)
    )
    assert reloaded.intercept is not None
    torch.testing.assert_close(reloaded.intercept["a"], config.intercept["a"])


def test_relaxation_config_missing_variable_in_file_raises(tmp_path: pathlib.Path):
    path = tmp_path / "ref.nc"
    _write_reference(path, {"a": np.zeros((3, 4))}, {"a": np.zeros((3, 4))})
    config = CO2RegressionRelaxationConfig(
        timescale_days={"b": 1.0}, co2_name=CO2_NAME, path=str(path)
    )
    with pytest.raises(ValueError, match="b_intercept"):
        config.load()


def _get_dataset_info(img_shape: tuple[int, int]) -> DatasetInfo:
    return DatasetInfo(
        vertical_coordinate=NullVerticalCoordinate(),
        gridded_operations=LatLonOperations(area_weights=torch.ones(img_shape)),
        img_shape=img_shape,
        timestep=TIMESTEP,
    )


def test_corrector_selector_load_persists_reference(tmp_path: pathlib.Path):
    img_shape = (3, 4)
    path = tmp_path / "ref.nc"
    _write_reference(path, {"a": np.ones(img_shape)}, {"a": np.ones(img_shape)})
    selector = CorrectorSelector(
        type="atmosphere_corrector",
        config={
            "co2_regression_relaxation": {
                "timescale_days": {"a": 1.0},
                "co2_name": CO2_NAME,
                "path": str(path),
            }
        },
    )
    selector.load()
    # the loaded reference is in the serialized form of the selector
    state = dataclasses.asdict(selector)
    relaxation_state = state["config"]["co2_regression_relaxation"]
    assert relaxation_state["path"] is None
    assert relaxation_state["co2_ref"] == pytest.approx(CO2_REF)
    path.unlink()
    reloaded = CorrectorSelector.from_state(state)
    corrector = reloaded.get_corrector(_get_dataset_info(img_shape))
    corrector.eval()
    shape = (2, *img_shape)
    x = torch.zeros(shape)
    result = corrector({"a": x}, {"a": x}, _co2(shape, CO2_REF), None)
    # target is the intercept (ones) at co2 == co2_ref
    torch.testing.assert_close(
        result.corrected["a"], torch.full(shape, 1 - math.exp(-0.25))
    )


@pytest.mark.parametrize("scheduled", [False, True])
def test_atmosphere_corrector_relaxation_follows_mode(scheduled: bool):
    img_shape = (3, 4)
    config = AtmosphereCorrectorConfig(
        co2_regression_relaxation=_loaded_config(
            {"a": 1.0}, {"a": torch.ones(img_shape)}, {"a": torch.zeros(img_shape)}
        ),
        corrector_disabled_epochs=1 if scheduled else 0,
    )
    corrector = config.get_corrector(_get_dataset_info(img_shape))
    shape = (2, *img_shape)
    x = torch.zeros(shape)
    forcing = _co2(shape, CO2_REF)
    result = corrector({"a": x}, {"a": x}, forcing, None)
    torch.testing.assert_close(result.corrected["a"], x)
    assert "a" not in result.modified_names
    corrector.eval()
    result = corrector({"a": x}, {"a": x}, forcing, None)
    assert "a" in result.modified_names
    assert torch.all(result.corrected["a"] > 0)
    corrector.train()
    result = corrector({"a": x}, {"a": x}, forcing, None)
    torch.testing.assert_close(result.corrected["a"], x)


def test_relaxation_precedes_energy_correction():
    """The energy correction must close the budget of the relaxed state, so
    the relaxation has to run first."""
    torch.manual_seed(0)
    tensor_shape = (2, 5, 5)
    input_data, gen_data, forcing_data, vertical_coord = _get_corrector_test_input(
        tensor_shape
    )
    forcing_data = {**forcing_data, **_co2(tensor_shape, 2 * CO2_REF)}
    ops = LatLonOperations(
        0.5 + torch.rand(size=(tensor_shape[-2], 1)).broadcast_to(size=tensor_shape)
    )
    timestep = datetime.timedelta(seconds=3600)
    config = AtmosphereCorrectorConfig(
        total_energy_budget_correction=EnergyBudgetConfig("constant_temperature"),
        co2_regression_relaxation=_loaded_config(
            {"air_temperature_0": 0.01},
            {"air_temperature_0": torch.full(tensor_shape[-2:], 250.0)},
            {"air_temperature_0": torch.full(tensor_shape[-2:], 10.0)},
        ),
    )
    corrector = config._build(ops, vertical_coord, timestep)
    corrector.eval()
    corrected = corrector(input_data, gen_data, forcing_data, None).corrected
    # the relaxation has an effect beyond the energy correction alone
    energy_only = dataclasses.replace(config, co2_regression_relaxation=None)
    energy_only_corrected = energy_only._build(ops, vertical_coord, timestep)(
        input_data, gen_data, forcing_data, None
    ).corrected
    assert not torch.allclose(
        corrected["air_temperature_0"], energy_only_corrected["air_temperature_0"]
    )
    # energy is conserved with respect to the final, relaxed state
    input = AtmosphereData(input_data, vertical_coord)
    corrected_gen = AtmosphereData(corrected | forcing_data, vertical_coord)
    input_gm_energy = ops.area_weighted_mean(input.total_energy_ace2_path)
    corrected_gm_energy = ops.area_weighted_mean(corrected_gen.total_energy_ace2_path)
    predicted_tendency = ops.area_weighted_mean(
        corrected_gen.net_energy_flux_into_atmosphere
    )
    expected_gm_energy = input_gm_energy + predicted_tendency * timestep.total_seconds()
    torch.testing.assert_close(corrected_gm_energy, expected_gm_energy)
