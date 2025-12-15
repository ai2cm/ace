import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.requirements import DataRequirements
from fme.ace.stepper.derived_forcings import DerivedForcingsConfig, ForcingDeriver
from fme.ace.stepper.insolation.config import InsolationConfig, NameConfig, ValueConfig
from fme.ace.stepper.insolation.test_insolation import validate_insolation
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo

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
            None,
            [INSOLATION_NAME],
            [INSOLATION_NAME],
            id="insolation-required-but-not-derived",
        ),
    ],
)
def test_update_requirements(
    insolation: InsolationConfig | None,
    input_names: list[str],
    expected_names: list[str],
):
    derived_forcings = DerivedForcingsConfig(insolation=insolation)
    n_timesteps = 2
    requirements = DataRequirements(names=input_names, n_timesteps=n_timesteps)
    result = derived_forcings.update_requirements(requirements)
    assert result.names == expected_names
    assert result.n_timesteps == n_timesteps


@pytest.mark.parametrize(
    ("replacement_insolation", "raises", "match"),
    [
        pytest.param(
            InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_VALUE, obliquity=0.0),
            False,
            None,
            id="valid-replacement",
        ),
        pytest.param(
            InsolationConfig("Other", SOLAR_CONSTANT_AS_VALUE),
            True,
            "insolation_name",
            id="incompatible-insolation-name",
        ),
    ],
)
def test_validate_replacement(
    replacement_insolation: InsolationConfig,
    raises: bool,
    match: str | None,
):
    original_insolation = InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_VALUE)
    derived_forcings = DerivedForcingsConfig(insolation=original_insolation)
    replacement = DerivedForcingsConfig(insolation=replacement_insolation)

    if raises:
        with pytest.raises(ValueError, match=match):
            derived_forcings.validate_replacement(replacement)
    else:
        derived_forcings.validate_replacement(replacement)


@pytest.mark.parametrize(
    "insolation",
    [
        pytest.param(
            InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_NAME),
            id="insolation-derived",
        ),
        pytest.param(None, id="insolation-not-derived"),
    ],
)
def test_build(
    insolation: InsolationConfig | None,
):
    dataset_info = DatasetInfo(
        timestep=TIMESTEP,
        horizontal_coordinates=HORIZONTAL_COORDINATES,
    )
    derived_forcings = DerivedForcingsConfig(insolation=insolation)
    forcing_deriver = derived_forcings.build(dataset_info)
    assert isinstance(forcing_deriver, ForcingDeriver)
    assert dataset_info.horizontal_coordinates == dataset_info.horizontal_coordinates


@pytest.mark.parametrize(
    "insolation",
    [
        pytest.param(
            None,
            id="no-derived-insolation",
        ),
        pytest.param(
            InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_NAME),
            id="solar-constant-as-name",
        ),
        pytest.param(
            InsolationConfig(INSOLATION_NAME, SOLAR_CONSTANT_AS_VALUE),
            id="solar-constant-as-value",
        ),
    ],
)
def test_forcing_deriver(insolation: InsolationConfig | None):
    n_timesteps = 2
    n_samples = 3
    index = xr.date_range("2000", freq="6h", periods=n_timesteps, use_cftime=True)
    time = xr.DataArray(np.stack(n_samples * [index]), dims=["sample", "time"])
    expected_shape = (n_samples, n_timesteps, N_LAT, N_LON)

    dataset_info = DatasetInfo(
        timestep=TIMESTEP,
        horizontal_coordinates=HORIZONTAL_COORDINATES,
        variable_metadata={},
    )
    derived_forcings = DerivedForcingsConfig(insolation=insolation)
    forcing_deriver = derived_forcings.build(dataset_info)

    forcing_dict = {}
    if insolation is not None:
        if isinstance(insolation.solar_constant, NameConfig):
            solar_constant_name = insolation.solar_constant.name
            forcing_dict[solar_constant_name] = torch.full(expected_shape, S0)

    forcing = BatchData(forcing_dict, time, labels=None)
    result = forcing_deriver(forcing)

    if insolation is None:
        assert result.data == {}
    else:
        validate_insolation(
            result.data,
            insolation,
            HORIZONTAL_COORDINATES,
            S0,
            expected_shape,
        )

        # Check that we did not mutate the input tensor dictionary.
        assert forcing_dict != result.data
