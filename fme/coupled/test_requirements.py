import datetime

import pytest

from fme.ace.requirements import DataRequirements

from .requirements import CoupledDataRequirements


@pytest.mark.parametrize(
    "requirements, n_steps_fast",
    [
        (
            CoupledDataRequirements(
                ocean_timestep=datetime.timedelta(days=2),
                ocean_requirements=DataRequirements(names=[], n_timesteps=3),
                atmosphere_timestep=datetime.timedelta(days=1),
                atmosphere_requirements=DataRequirements(names=[], n_timesteps=5),
            ),
            2,
        ),
        (
            CoupledDataRequirements(
                ocean_timestep=datetime.timedelta(days=3),
                ocean_requirements=DataRequirements(names=[], n_timesteps=2),
                atmosphere_timestep=datetime.timedelta(days=1),
                atmosphere_requirements=DataRequirements(names=[], n_timesteps=4),
            ),
            3,
        ),
    ],
)
def test_n_steps_fast(requirements: CoupledDataRequirements, n_steps_fast: int):
    assert requirements.n_steps_fast == n_steps_fast


def test_timestep_mismatch_error():
    ocean_names = ["o_exog", "o_prog", "o_sfc"]
    atmos_names = ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"]
    ocean_requirements = DataRequirements(
        names=ocean_names,
        n_timesteps=2,
    )
    atmosphere_requirements = DataRequirements(
        names=atmos_names,
        n_timesteps=3,
    )
    ocean_timestep = datetime.timedelta(days=1)
    atmos_timestep = datetime.timedelta(days=1)

    with pytest.raises(ValueError) as err:
        CoupledDataRequirements(
            ocean_timestep=ocean_timestep,
            ocean_requirements=ocean_requirements,
            atmosphere_timestep=atmos_timestep,
            atmosphere_requirements=atmosphere_requirements,
        )

    assert "timepoints (including IC) per sample" in str(err.value)


def test_atmosphere_timestep_too_large_error():
    ocean_names = ["o_exog", "o_prog", "o_sfc"]
    atmos_names = ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"]
    ocean_requirements = DataRequirements(
        names=ocean_names,
        n_timesteps=2,
    )
    atmosphere_requirements = DataRequirements(
        names=atmos_names,
        n_timesteps=2,
    )
    ocean_timestep = datetime.timedelta(days=1)
    atmos_timestep = datetime.timedelta(days=2)

    with pytest.raises(ValueError) as err:
        CoupledDataRequirements(
            ocean_timestep=ocean_timestep,
            ocean_requirements=ocean_requirements,
            atmosphere_timestep=atmos_timestep,
            atmosphere_requirements=atmosphere_requirements,
        )

    assert "must be no larger" in str(err.value)


def test_timestep_not_a_multiple_error():
    ocean_names = ["o_exog", "o_prog", "o_sfc"]
    atmos_names = ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"]
    ocean_requirements = DataRequirements(
        names=ocean_names,
        n_timesteps=2,
    )
    atmosphere_requirements = DataRequirements(
        names=atmos_names,
        n_timesteps=2,
    )
    ocean_timestep = datetime.timedelta(days=3)
    atmos_timestep = datetime.timedelta(days=2)

    with pytest.raises(ValueError) as err:
        CoupledDataRequirements(
            ocean_timestep=ocean_timestep,
            ocean_requirements=ocean_requirements,
            atmosphere_timestep=atmos_timestep,
            atmosphere_requirements=atmosphere_requirements,
        )

    assert "to be a multiple" in str(err.value)


def test_misconfigured_n_timesteps_error():
    ocean_names = ["o_exog", "o_prog", "o_sfc"]
    atmos_names = ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"]
    ocean_requirements = DataRequirements(
        names=ocean_names,
        n_timesteps=2,
    )
    atmosphere_requirements = DataRequirements(
        names=atmos_names,
        n_timesteps=2,
    )
    ocean_timestep = datetime.timedelta(days=2)
    atmos_timestep = datetime.timedelta(days=1)

    with pytest.raises(ValueError) as err:
        CoupledDataRequirements(
            ocean_timestep=ocean_timestep,
            ocean_requirements=ocean_requirements,
            atmosphere_timestep=atmos_timestep,
            atmosphere_requirements=atmosphere_requirements,
        )

    assert "timepoints (including IC) per sample" in str(err.value)


def test_valid_n_timesteps():
    ocean_names = ["o_exog", "o_prog", "o_sfc"]
    atmos_names = ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"]
    ocean_requirements = DataRequirements(
        names=ocean_names,
        n_timesteps=3,
    )
    atmosphere_requirements = DataRequirements(
        names=atmos_names,
        n_timesteps=5,
    )
    ocean_timestep = datetime.timedelta(days=2)
    atmos_timestep = datetime.timedelta(days=1)

    CoupledDataRequirements(
        ocean_timestep=ocean_timestep,
        ocean_requirements=ocean_requirements,
        atmosphere_timestep=atmos_timestep,
        atmosphere_requirements=atmosphere_requirements,
    )
