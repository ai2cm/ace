import datetime
import os

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.requirements import DataRequirements
from fme.ace.stepper.insolation.cm4 import (
    AUTUMNAL_EQUINOX,
    _convert_lat_lon_from_degrees_to_radians,
)
from fme.ace.stepper.insolation.config import InsolationConfig, NameConfig, ValueConfig
from fme.core.coordinates import HorizontalCoordinates, LatLonCoordinates
from fme.core.device import get_device
from fme.core.testing import validate_tensor
from fme.core.typing_ import TensorMapping

INSOLATION_NAME = "DSWRFtoa"
SOLAR_CONSTANT_NAME = "solar_constant"
N_LAT = 8
N_LON = 16
LAT = torch.linspace(-90.0, 90.0, N_LAT)
LON = torch.linspace(0.0, 360.0 - 360.0 / N_LON, N_LON)
HORIZONTAL_COORDINATES = LatLonCoordinates(lat=LAT, lon=LON)
TIMESTEP = datetime.timedelta(hours=6)
S0 = 1360.0  # Physically realistic solar constant value
SOLAR_CONSTANT_AS_NAME = NameConfig(SOLAR_CONSTANT_NAME)
SOLAR_CONSTANT_AS_VALUE = ValueConfig(S0)
DIR = os.path.abspath(os.path.dirname(__file__))


def validate_insolation(
    forcing_dict: TensorMapping,
    insolation_config: InsolationConfig,
    horizontal_coordinates: HorizontalCoordinates,
    solar_constant: float,
    expected_shape: tuple[int, ...],
):
    insolation_name = insolation_config.insolation_name
    insolation = forcing_dict[insolation_name]
    assert insolation.min() == 0.0
    assert insolation.max() > 1000.0
    assert insolation.shape == expected_shape

    if isinstance(insolation_config.solar_constant, NameConfig):
        solar_constant_name = insolation_config.solar_constant.name
        assert solar_constant_name in forcing_dict
        expected_dtype = forcing_dict[solar_constant_name].dtype
    else:
        expected_dtype = insolation_config.solar_constant.torch_dtype
    assert insolation.dtype == expected_dtype

    # Check that global mean insolation falls somewhere within ±10 percent of
    # solar_constant / 4, to account for the variability of the global mean
    # insolation over the year for the current eccentricity, with some generous
    # tolerance built in for the reduced grid resolution of the test.
    expected_mean = solar_constant / 4
    gridded_operations = horizontal_coordinates.get_gridded_operations()
    mean = gridded_operations.area_weighted_mean(insolation)
    expected_mean = torch.full(mean.shape, expected_mean, device=get_device())
    torch.testing.assert_close(mean, expected_mean, rtol=0.1, atol=0.0)


@pytest.mark.parametrize(
    ("insolation", "input_names", "expected_names"),
    [
        pytest.param(
            InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_NAME),
            [INSOLATION_NAME],
            [SOLAR_CONSTANT_NAME],
            id="insolation-required-solar-constant-as-name",
        ),
        pytest.param(
            InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_VALUE),
            [INSOLATION_NAME],
            [],
            id="insolation-required-solar-constant-as-value",
        ),
        pytest.param(
            InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_NAME),
            [],
            [],
            id="insolation-not-required-solar-constant-as-name",
        ),
        pytest.param(
            InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_VALUE),
            [],
            [],
            id="insolation-not-required-solar-constant-as-value",
        ),
    ],
)
def test_update_requirements(
    insolation: InsolationConfig,
    input_names: list[str],
    expected_names: list[str],
):
    n_timesteps = 2
    requirements = DataRequirements(names=input_names, n_timesteps=n_timesteps)
    result = insolation.update_requirements(requirements)
    assert result.names == expected_names
    assert result.n_timesteps == n_timesteps


@pytest.mark.parametrize(
    ("solar_constant_config", "test_id"),
    [
        pytest.param(
            SOLAR_CONSTANT_AS_NAME,
            "solar-constant-as-name",
            id="solar-constant-as-name",
        ),
        pytest.param(
            SOLAR_CONSTANT_AS_VALUE,
            "solar-constant-as-value",
            id="solar-constant-as-value",
        ),
    ],
)
def test_insolation_compute(
    solar_constant_config: NameConfig | ValueConfig, test_id: str
):
    n_timesteps = 2
    n_samples = 3
    index = xr.date_range("2000", freq="6h", periods=n_timesteps, use_cftime=True)
    time = xr.DataArray(np.stack(n_samples * [index]), dims=["sample", "time"])
    expected_shape = (n_samples, n_timesteps, N_LAT, N_LON)
    insolation_config = InsolationConfig(INSOLATION_NAME, solar_constant_config)
    insolation = insolation_config.build(TIMESTEP, HORIZONTAL_COORDINATES)

    tensors = {}
    if isinstance(solar_constant_config, NameConfig):
        tensors[SOLAR_CONSTANT_NAME] = torch.full(expected_shape, S0)
    result = insolation.compute(time, tensors)

    validate_insolation(
        result, insolation_config, HORIZONTAL_COORDINATES, S0, expected_shape
    )

    # Check that we did not mutate the input tensor dictionary.
    assert tensors != result

    reference_filename = os.path.join(DIR, f"testdata/{test_id}.pt")
    validate_tensor(result[INSOLATION_NAME], reference_filename, rtol=1e-4, atol=0.0)


@pytest.mark.parametrize("eccentricity", [0.0, 0.0167])
def test_longitude_of_perhelion_and_eccentricity(eccentricity: float):
    times = [cftime.DatetimeGregorian(*AUTUMNAL_EQUINOX)]
    time = xr.DataArray(times, dims=["time"], coords=[times], name="time")

    perhelion_config = InsolationConfig(
        insolation_name=INSOLATION_NAME,
        solar_constant=SOLAR_CONSTANT_AS_VALUE,
        eccentricity=eccentricity,
        longitude_of_perhelion=0.0,
    )
    perhelion_insolation = perhelion_config.build(TIMESTEP, HORIZONTAL_COORDINATES)

    aphelion_config = InsolationConfig(
        insolation_name=INSOLATION_NAME,
        solar_constant=SOLAR_CONSTANT_AS_VALUE,
        eccentricity=eccentricity,
        longitude_of_perhelion=180.0,
    )
    aphelion_insolation = aphelion_config.build(TIMESTEP, HORIZONTAL_COORDINATES)

    tensors: TensorMapping = {}
    perhelion_tensors = perhelion_insolation.compute(time, tensors)
    aphelion_tensors = aphelion_insolation.compute(time, tensors)

    # Check that the mean insolation at perhelion is larger than the mean insolation at
    # aphelion if the eccentricity is greater than 0.0; otherwise it will be equal.
    gridded_operations = HORIZONTAL_COORDINATES.get_gridded_operations()
    perhelion_mean = gridded_operations.area_weighted_mean(
        perhelion_tensors[INSOLATION_NAME]
    ).squeeze()
    aphelion_mean = gridded_operations.area_weighted_mean(
        aphelion_tensors[INSOLATION_NAME]
    ).squeeze()

    if eccentricity > 0.0:
        assert perhelion_mean > aphelion_mean
    else:
        assert perhelion_mean == aphelion_mean


@pytest.mark.parametrize(("obliquity", "raises"), [(0.0, False), (23.439, True)])
def test_obliquity(obliquity: float, raises: bool):
    # Intentionally create times at like times of day, ignoring the underlying
    # timestep of ACE.
    n_times = 2
    index = xr.date_range("2000", freq="D", periods=n_times, use_cftime=True)
    time = xr.DataArray(index, dims=["time"], coords=[index], name="time")

    insolation_config = InsolationConfig(
        insolation_name=INSOLATION_NAME,
        solar_constant=SOLAR_CONSTANT_AS_VALUE,
        eccentricity=0.0,
        obliquity=obliquity,
    )
    insolation = insolation_config.build(TIMESTEP, HORIZONTAL_COORDINATES)

    tensors: TensorMapping = {}
    tensors = insolation.compute(time, tensors)
    result = tensors[INSOLATION_NAME]

    # Check that for like time of day, the insolation is identical when the
    # eccentricity and obliquity are set to 0.0; if obliquity is non-zero they
    # will differ due to the seasonal cycle.
    if raises:
        with pytest.raises(AssertionError):
            torch.testing.assert_close(result[0], result[1], atol=0.0, rtol=0.0)
    else:
        torch.testing.assert_close(result[0], result[1], atol=0.0, rtol=0.0)


def test_timestep_error():
    n_times = 2
    index = xr.date_range("2000", freq="D", periods=n_times, use_cftime=True)
    time = xr.DataArray(index, dims=["time"], coords=[index], name="time")

    timestep = datetime.timedelta(hours=12)
    insolation_config = InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_VALUE)
    insolation = insolation_config.build(timestep, HORIZONTAL_COORDINATES)

    tensors: TensorMapping = {}
    with pytest.raises(NotImplementedError, match="timestep"):
        insolation.compute(time, tensors)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        pytest.param(
            torch.tensor([-0.5, 0.5]),
            torch.tensor([0.5, 359.5]),
            id="lat-triggers-warning",
        ),
        pytest.param(
            torch.tensor([-89.5, 89.5]),
            torch.tensor([-2.0, 2.0]),
            id="lon-triggers-warning",
        ),
    ],
)
def test_horizontal_coordinates_warning(lat: torch.tensor, lon: torch.tensor):
    with pytest.warns(match="degrees"):
        _convert_lat_lon_from_degrees_to_radians(lat, lon)
