from collections import namedtuple
from typing import Iterable, List, Literal, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData
from fme.ace.stepper import SingleModuleStepperConfig
from fme.core.coordinates import (
    DepthCoordinate,
    HybridSigmaPressureCoordinate,
    VerticalCoordinate,
)
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.ocean import OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.registry.corrector import CorrectorSelector
from fme.core.registry.module import ModuleSelector

from .data_loading.batch_data import (
    CoupledBatchData,
    CoupledPairedData,
    CoupledPrognosticState,
)
from .data_loading.data_typing import CoupledVerticalCoordinate
from .stepper import ComponentConfig, CoupledStepper, CoupledStepperConfig

DEVICE = fme.get_device()
NZ = 3  # number of vertical interface levels in mock data from get_data
N_LAT = 5
N_LON = 5

ATMOS_STEPPER_CONFIG = SingleModuleStepperConfig(
    builder=MagicMock(),
    in_names=["a", "f"],
    out_names=["a"],
    normalization=MagicMock(),
    loss=MagicMock(),
    ocean=OceanConfig(
        surface_temperature_name="a",
        ocean_fraction_name="f",
    ),
)

OCEAN_STEPPER_CONFIG = SingleModuleStepperConfig(
    builder=MagicMock(),
    in_names=["sst"],
    out_names=["sst"],
    normalization=MagicMock(),
    loss=MagicMock(),
)


ForcingInputs = namedtuple(
    "ForcingInputs", ["atmos_in", "atmos_out", "ocean_in", "ocean_out"]
)
ForcingExpectations = namedtuple(
    "ForcingExpectations",
    [
        "all_atmos",
        "atmos_exog",
        "atmos_to_ocean_forcings",
        "all_ocean",
        "ocean_exog",
        "ocean_to_atmos_forcings",
    ],
)

FORCING_TEST_PARAMS = [
    (
        ForcingInputs(["a", "c"], ["a", "b"], ["a", "c"], ["c", "d"]),
        ForcingExpectations(["a", "b"], [], ["a"], ["c", "d"], [], ["c"]),
    ),
    (
        ForcingInputs(["a", "c", "f"], ["a", "b"], ["a", "c"], ["c", "d"]),
        ForcingExpectations(["a", "b", "f"], ["f"], ["a"], ["c", "d"], [], ["c"]),
    ),
    (
        ForcingInputs(["a", "c"], ["a", "b"], ["a", "c", "f"], ["c", "d"]),
        ForcingExpectations(["a", "b"], [], ["a"], ["c", "d", "f"], ["f"], ["c"]),
    ),
    (
        ForcingInputs(["a", "c", "f"], ["a", "b"], ["b", "c", "f"], ["d"]),
        ForcingExpectations(
            ["a", "b", "c", "f"], ["c", "f"], ["b"], ["c", "d", "f"], ["c", "f"], []
        ),
    ),
    (
        ForcingInputs(["a", "c", "f"], ["a", "b"], ["b", "f"], ["d"]),
        ForcingExpectations(
            ["a", "b", "c", "f"], ["c", "f"], ["b"], ["d", "f"], ["f"], []
        ),
    ),
]


@pytest.mark.parametrize(
    "inputs, expectations",
    FORCING_TEST_PARAMS,
)
def test_config_names(inputs, expectations):
    # atmosphere and ocean have the same surface temperature name
    atmos_in = inputs.atmos_in + ["a_sfc_temp", "frac"]
    atmos_out = inputs.atmos_out + ["a_sfc_temp"]
    atmosphere = ComponentConfig(
        timedelta="6h",
        stepper=SingleModuleStepperConfig(
            builder=MagicMock(),
            in_names=atmos_in,
            out_names=atmos_out,
            normalization=MagicMock(),
            loss=MagicMock(),
            ocean=OceanConfig(
                surface_temperature_name="a_sfc_temp",
                ocean_fraction_name="frac",
            ),
        ),
    )
    ocean_in = inputs.ocean_in + ["o_sfc_temp"]
    ocean_out = inputs.ocean_out + ["o_sfc_temp"]
    ocean = ComponentConfig(
        timedelta="12h",
        stepper=SingleModuleStepperConfig(
            builder=MagicMock(),
            in_names=ocean_in,
            out_names=ocean_out,
            next_step_forcing_names=expectations.atmos_to_ocean_forcings,
            normalization=MagicMock(),
            loss=MagicMock(),
        ),
    )
    config = CoupledStepperConfig(
        atmosphere=atmosphere, ocean=ocean, sst_name="o_sfc_temp"
    )
    assert sorted(config.atmosphere_forcing_exogenous_names) == sorted(
        expectations.atmos_exog
        + [
            "frac",
        ]
    )
    assert sorted(config.atmosphere_to_ocean_forcing_names) == sorted(
        expectations.atmos_to_ocean_forcings
    )
    assert sorted(config.ocean_forcing_exogenous_names) == sorted(
        expectations.ocean_exog
    )
    assert sorted(config.ocean_to_atmosphere_forcing_names) == sorted(
        expectations.ocean_to_atmos_forcings + ["o_sfc_temp"]
    )
    assert sorted(config._all_ocean_names) == sorted(
        expectations.all_ocean + ["o_sfc_temp"]
    )
    assert sorted(config._all_atmosphere_names) == sorted(
        expectations.all_atmos
        + [
            "frac",
            "a_sfc_temp",
        ]
    )


@pytest.mark.parametrize(
    "inputs, expectations",
    FORCING_TEST_PARAMS,
)
def test_config_names_diff_sfc_temp_names(inputs, expectations):
    # atmosphere and ocean have different surface temperature name
    atmos_in = inputs.atmos_in + ["atmos_surface_temp", "frac"]
    atmos_out = inputs.atmos_out + ["atmos_surface_temp"]
    atmosphere = ComponentConfig(
        timedelta="6h",
        stepper=SingleModuleStepperConfig(
            builder=MagicMock(),
            in_names=atmos_in,
            out_names=atmos_out,
            normalization=MagicMock(),
            loss=MagicMock(),
            ocean=OceanConfig(
                surface_temperature_name="atmos_surface_temp",
                ocean_fraction_name="frac",
            ),
        ),
    )
    ocean_in = inputs.ocean_in + ["ocean_surface_temp"]
    ocean_out = inputs.ocean_out + ["ocean_surface_temp"]
    ocean = ComponentConfig(
        timedelta="12h",
        stepper=SingleModuleStepperConfig(
            builder=MagicMock(),
            in_names=ocean_in,
            out_names=ocean_out,
            next_step_forcing_names=expectations.atmos_to_ocean_forcings,
            normalization=MagicMock(),
            loss=MagicMock(),
        ),
    )
    config = CoupledStepperConfig(
        atmosphere=atmosphere, ocean=ocean, sst_name="ocean_surface_temp"
    )
    assert sorted(config.atmosphere_forcing_exogenous_names) == sorted(
        expectations.atmos_exog + ["frac"]
    )
    assert sorted(config.atmosphere_to_ocean_forcing_names) == sorted(
        expectations.atmos_to_ocean_forcings
    )
    assert sorted(config.ocean_forcing_exogenous_names) == sorted(
        expectations.ocean_exog
    )
    assert sorted(config.ocean_to_atmosphere_forcing_names) == sorted(
        expectations.ocean_to_atmos_forcings + ["ocean_surface_temp"]
    )
    assert sorted(config._all_ocean_names) == sorted(
        expectations.all_ocean + ["ocean_surface_temp"]
    )
    assert sorted(config._all_atmosphere_names) == sorted(
        expectations.all_atmos
        + [
            "frac",
            "atmos_surface_temp",
        ]
    )


@pytest.mark.parametrize(
    "timedelta_atmos, timedelta_ocean, expected_n_inner_steps",
    [
        ("6h", "5D", 20),
        ("0.25 days", "1 w", 28),
    ],
)
def test_config_n_inner_steps(timedelta_atmos, timedelta_ocean, expected_n_inner_steps):
    atmosphere = ComponentConfig(
        timedelta=timedelta_atmos,
        stepper=ATMOS_STEPPER_CONFIG,
    )
    ocean = ComponentConfig(
        timedelta=timedelta_ocean,
        stepper=OCEAN_STEPPER_CONFIG,
    )
    config = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)
    assert config.n_inner_steps == expected_n_inner_steps


def test_config_init_atmos_stepper_missing_ocean_error():
    # atmosphere is required to have stepper.ocean attribute
    atmosphere = ComponentConfig(
        timedelta="6h",
        stepper=OCEAN_STEPPER_CONFIG,
    )
    ocean = ComponentConfig(
        timedelta="1h",
        stepper=OCEAN_STEPPER_CONFIG,
    )

    with pytest.raises(
        ValueError, match=r".* atmosphere stepper 'ocean' config is missing .*"
    ):
        _ = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)


def test_config_init_timedelta_comparison_error():
    # atmosphere timedelta > ocean timedelta raises error
    atmosphere = ComponentConfig(
        timedelta="6h",
        stepper=ATMOS_STEPPER_CONFIG,
    )
    ocean = ComponentConfig(
        timedelta="1h",
        stepper=OCEAN_STEPPER_CONFIG,
    )

    with pytest.raises(
        ValueError, match=r"Atmosphere timedelta must not be larger than ocean's."
    ):
        _ = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)


def test_config_init_incompatible_timedelta_error():
    # ocean timestep % atmosphere timestep != 0 raises error
    atmosphere = ComponentConfig(
        timedelta="2h",
        stepper=ATMOS_STEPPER_CONFIG,
    )
    ocean = ComponentConfig(
        timedelta="5h",
        stepper=OCEAN_STEPPER_CONFIG,
    )
    with pytest.raises(
        ValueError, match="Ocean timedelta must be a multiple of the atmosphere's."
    ):
        _ = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)


def test_config_missing_next_step_forcings_error():
    # ocean stepper config with input names overlapping with atmosphere output
    # names but missing next_step_forcing_names raises error
    atmosphere = ComponentConfig(
        timedelta="6h",
        stepper=ATMOS_STEPPER_CONFIG,
    )
    ocean = ComponentConfig(
        timedelta="5D",
        stepper=SingleModuleStepperConfig(
            builder=MagicMock(),
            in_names=["sst", "a", "b"],
            out_names=["sst"],
            next_step_forcing_names=["b"],
            normalization=MagicMock(),
            loss=MagicMock(),
        ),
    )
    with pytest.raises(ValueError, match=r".* next_step_forcing_names: \['a'\]\."):
        _ = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)


SphericalData = namedtuple(
    "SphericalData",
    [
        "data",
        "area_weights",
        "vertical_coord",
    ],
)


def get_data(
    names: Iterable[str], n_samples, n_time, realm: Literal["atmosphere", "ocean"]
) -> SphericalData:
    data_dict = {}

    lats = torch.linspace(-89.5, 89.5, N_LAT)  # arbitary choice
    for name in names:
        data_dict[name] = torch.rand(n_samples, n_time, N_LAT, N_LON, device=DEVICE)
    area_weights = fme.spherical_area_weights(lats, N_LON).to(DEVICE)
    vertical_coord: VerticalCoordinate
    if realm == "atmosphere":
        ak, bk = torch.arange(NZ), torch.arange(NZ)
        vertical_coord = HybridSigmaPressureCoordinate(ak, bk)
    elif realm == "ocean":
        vertical_coord = DepthCoordinate(
            torch.arange(NZ), torch.ones(N_LAT, N_LON, NZ - 1)
        )
    data = BatchData.new_on_device(
        data=data_dict,
        time=xr.DataArray(
            np.zeros((n_samples, n_time)),
            dims=["sample", "time"],
        ),
    )
    return SphericalData(data, area_weights, vertical_coord)


def get_coupled_data(
    ocean_names: List[str],
    atmosphere_names: List[str],
    n_forward_times_ocean: int,
    n_forward_times_atmosphere: int,
    n_samples: int,
) -> SphericalData:
    ocean_data = get_data(ocean_names, n_samples, n_forward_times_ocean + 1, "ocean")
    atmos_data = get_data(
        atmosphere_names, n_samples, n_forward_times_atmosphere + 1, "atmosphere"
    )
    data = CoupledBatchData(ocean_data=ocean_data.data, atmosphere_data=atmos_data.data)
    nz = len(atmos_data.vertical_coord)
    assert nz == NZ, f"expected 7 interfaces in mock data vertical coord but got {nz}"
    return SphericalData(
        data,
        atmos_data.area_weights,
        CoupledVerticalCoordinate(
            ocean=ocean_data.vertical_coord, atmosphere=atmos_data.vertical_coord
        ),
    )


# default atmosphere module
class AddOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


# default ocean module
class TimesTwo(torch.nn.Module):
    def forward(self, x):
        return 2 * x


def get_stepper_config(
    ocean_in_names: List[str],
    ocean_out_names: List[str],
    atmosphere_in_names: List[str],
    atmosphere_out_names: List[str],
    sst_name_in_ocean_data: str = "sst",
    sst_mask_name_in_ocean_data: str = "mask_0",
    sfc_temp_name_in_atmosphere_data: str = "surface_temperature",
    ocean_fraction_name: str = "ocean_fraction",
    ocean_builder: Optional[ModuleSelector] = None,
    atmosphere_builder: Optional[ModuleSelector] = None,
    ocean_timedelta: str = "2D",
    atmosphere_timedelta: str = "1D",
):
    # CoupledStepper requires that both component datasets include prognostic
    # surface temperature variables and that the atmosphere data includes an
    # ocean fraction forcing variable
    assert sst_name_in_ocean_data in ocean_in_names
    assert sst_name_in_ocean_data in ocean_out_names
    assert sfc_temp_name_in_atmosphere_data in atmosphere_in_names
    assert sfc_temp_name_in_atmosphere_data in atmosphere_out_names
    assert sst_mask_name_in_ocean_data in ocean_in_names
    assert ocean_fraction_name in atmosphere_in_names

    ocean_norm_names = set(ocean_in_names + ocean_out_names)
    atmos_norm_names = set(atmosphere_in_names + atmosphere_out_names)
    next_step_forcing_names = list(set(atmosphere_out_names) & set(ocean_in_names))

    if atmosphere_builder is None:
        atmosphere_builder = ModuleSelector(
            type="prebuilt", config={"module": AddOne()}
        )
    if ocean_builder is None:
        ocean_builder = ModuleSelector(type="prebuilt", config={"module": TimesTwo()})

    config = CoupledStepperConfig(
        atmosphere=ComponentConfig(
            timedelta=atmosphere_timedelta,
            stepper=SingleModuleStepperConfig(
                builder=atmosphere_builder,
                in_names=atmosphere_in_names,
                out_names=atmosphere_out_names,
                normalization=NormalizationConfig(
                    means={name: 0.0 for name in atmos_norm_names},
                    stds={name: 1.0 for name in atmos_norm_names},
                ),
                loss=WeightedMappingLossConfig(type="MSE"),
                ocean=OceanConfig(
                    surface_temperature_name=sfc_temp_name_in_atmosphere_data,
                    ocean_fraction_name=ocean_fraction_name,
                ),
            ),
        ),
        ocean=ComponentConfig(
            timedelta=ocean_timedelta,
            stepper=SingleModuleStepperConfig(
                builder=ocean_builder,
                in_names=ocean_in_names,
                out_names=ocean_out_names,
                next_step_forcing_names=next_step_forcing_names,
                normalization=NormalizationConfig(
                    means={name: 0.0 for name in ocean_norm_names},
                    stds={name: 1.0 for name in ocean_norm_names},
                ),
                loss=WeightedMappingLossConfig(type="MSE"),
                corrector=CorrectorSelector("ocean_corrector", {}),
            ),
        ),
        sst_name=sst_name_in_ocean_data,
        sst_mask_name=sst_mask_name_in_ocean_data,
    )
    return config


def get_stepper_and_batch(
    ocean_in_names: List[str],
    ocean_out_names: List[str],
    atmosphere_in_names: List[str],
    atmosphere_out_names: List[str],
    n_forward_times_ocean: int,
    n_forward_times_atmosphere: int,
    n_samples: int,
    sst_name_in_ocean_data: str = "sst",
    sst_mask_name_in_ocean_data: str = "mask_0",
    sfc_temp_name_in_atmosphere_data: str = "surface_temperature",
    ocean_fraction_name: str = "ocean_fraction",
    ocean_builder: Optional[ModuleSelector] = None,
    atmosphere_builder: Optional[ModuleSelector] = None,
):
    all_ocean_names = set(ocean_in_names + ocean_out_names)
    all_atmos_names = set(atmosphere_in_names + atmosphere_out_names)

    # variables with larger ocean timestep
    ocean_names = list(all_ocean_names - set(atmosphere_out_names))
    # variables with smaller atmosphere timestep
    atmos_names = list(all_atmos_names - set(ocean_out_names))

    # get the dataset
    coupled_data = get_coupled_data(
        ocean_names,
        atmos_names,
        n_forward_times_ocean=n_forward_times_ocean,
        n_forward_times_atmosphere=n_forward_times_atmosphere,
        n_samples=n_samples,
    )

    config = get_stepper_config(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmosphere_in_names,
        atmosphere_out_names=atmosphere_out_names,
        sst_name_in_ocean_data=sst_name_in_ocean_data,
        sst_mask_name_in_ocean_data=sst_mask_name_in_ocean_data,
        sfc_temp_name_in_atmosphere_data=sfc_temp_name_in_atmosphere_data,
        ocean_fraction_name=ocean_fraction_name,
        ocean_builder=ocean_builder,
        atmosphere_builder=atmosphere_builder,
        # NOTE: these values don't actually matter here because they aren't used
        # when stepping the batch forward... if you need consistency between the
        # timedeltas and batch time dims then you should use
        # n_forward_times_atmosphere = 2 * n_forward_times_ocean
        ocean_timedelta="2D",
        atmosphere_timedelta="1D",
    )

    coupler = config.get_stepper(
        img_shape=(N_LAT, N_LON),
        gridded_operations=LatLonOperations(coupled_data.area_weights),
        vertical_coordinate=coupled_data.vertical_coord,
    )
    return coupler, coupled_data


def test_predict_paired():
    ocean_in_names = ["o_prog", "o_sfc_temp", "o_mask", "a_diag"]
    ocean_out_names = ["o_prog", "o_sfc_temp", "o_diag"]
    atmos_in_names = ["a_prog", "a_sfc_temp", "ocean_frac", "o_prog"]
    atmos_out_names = ["a_prog", "a_sfc_temp", "a_diag"]

    class Ocean(torch.nn.Module):
        def forward(self, x):
            # in: [o_prog, o_sfc_temp, a_diag_avg]
            # out: [o_prog + 1, o_sfc_temp + 1, o_diag = o_prog + a_diag_avg + 1]
            o_prog = x[:, :1]
            o_sfc_temp = x[:, :1]
            o_diag = o_prog + x[:, -1:]
            return torch.concat([o_prog, o_sfc_temp, o_diag], dim=1) + 1

    class Atmos(torch.nn.Module):
        # in: [a_prog, a_sfc_temp, ocean_fraction, o_prog]
        # out: [a_prog + 2, a_sfc_temp + 2, a_diag = a_prog + o_prog + 2]
        def forward(self, x):
            a_prog = x[:, :1]
            a_sfc_temp = x[:, 1:2]
            a_diag = a_prog + x[:, -1:]
            return torch.concat([a_prog, a_sfc_temp, a_diag], dim=1) + 2

    coupler, coupled_data = get_stepper_and_batch(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_in_names,
        atmosphere_out_names=atmos_out_names,
        n_forward_times_ocean=2,
        n_forward_times_atmosphere=4,
        n_samples=3,
        sst_name_in_ocean_data="o_sfc_temp",
        sst_mask_name_in_ocean_data="o_mask",
        sfc_temp_name_in_atmosphere_data="a_sfc_temp",
        ocean_fraction_name="ocean_frac",
        ocean_builder=ModuleSelector(type="prebuilt", config={"module": Ocean()}),
        atmosphere_builder=ModuleSelector(type="prebuilt", config={"module": Atmos()}),
    )
    data = coupled_data.data

    atmos_prognostic_names = coupler.atmosphere.prognostic_names
    ocean_prognostic_names = coupler.ocean.prognostic_names
    atmos_prognostic = data.atmosphere_data.get_start(
        atmos_prognostic_names, n_ic_timesteps=1
    )
    ocean_prognostic = data.ocean_data.get_start(
        ocean_prognostic_names, n_ic_timesteps=1
    )
    ic = CoupledPrognosticState(
        atmosphere_data=atmos_prognostic, ocean_data=ocean_prognostic
    )

    paired_data, prognostic_state = coupler.predict_paired(
        initial_condition=ic,
        forcing=data,
    )
    assert isinstance(paired_data, CoupledPairedData)
    assert isinstance(prognostic_state, CoupledPrognosticState)

    # first two predicted atmos surface_temperature replaced by the ocean
    # initial condition sea surface temperature
    for i in range(2):
        mask = torch.round(
            torch.minimum(
                data.atmosphere_data.data["ocean_frac"][:, i + 1],
                data.ocean_data.data["o_mask"][:, 0],
            )
        )
        atmos_sst = paired_data.atmosphere_data.prediction["a_sfc_temp"][:, i] * mask
        ocean_sst = data.ocean_data.data["o_sfc_temp"][:, 0] * mask
        assert torch.allclose(atmos_sst, ocean_sst)

    # next two predicted atmos surface_temperature replaced by the ocean
    # predicted sea surface temperature
    for i in range(2, 4):
        mask = torch.round(
            torch.minimum(
                data.atmosphere_data.data["ocean_frac"][:, i + 1],
                data.ocean_data.data["o_mask"][:, 1],
            )
        )
        atmos_sst = paired_data.atmosphere_data.prediction["a_sfc_temp"][:, i] * mask
        ocean_sst = paired_data.ocean_data.prediction["o_sfc_temp"][:, 0] * mask
        assert torch.allclose(atmos_sst, ocean_sst)

    a_prog_ic = data.atmosphere_data.data["a_prog"].select(dim=1, index=0)
    o_prog_ic = data.ocean_data.data["o_prog"].select(dim=1, index=0)

    # check o_prog computations
    for i in range(2):
        o_prog_pred = o_prog_ic + (i + 1)
        assert torch.allclose(
            paired_data.ocean_data.prediction["o_prog"].select(dim=1, index=i),
            o_prog_pred,
        )

    # check a_prog computations
    for i in range(4):
        a_prog_pred = a_prog_ic + (i + 1) * 2
        assert torch.allclose(
            paired_data.atmosphere_data.prediction["a_prog"].select(dim=1, index=i),
            a_prog_pred,
        )

    # check first two a_diag computations
    a_diag0 = a_prog_ic + o_prog_ic + 2
    a_diag1 = (a_prog_ic + 2) + o_prog_ic + 2
    assert torch.allclose(
        paired_data.atmosphere_data.prediction["a_diag"].select(dim=1, index=0),
        a_diag0,
    )
    assert torch.allclose(
        paired_data.atmosphere_data.prediction["a_diag"].select(dim=1, index=1),
        a_diag1,
    )

    # check first o_diag computation
    o_diag0 = (a_diag0 + a_diag1) / 2 + o_prog_ic + 1
    assert torch.allclose(
        paired_data.ocean_data.prediction["o_diag"].select(dim=1, index=0),
        o_diag0,
    )

    # check next two a_diag computations
    a_diag2 = (a_prog_ic + 2 + 2) + (o_prog_ic + 1) + 2
    a_diag3 = (a_prog_ic + 2 + 2 + 2) + (o_prog_ic + 1) + 2
    assert torch.allclose(
        paired_data.atmosphere_data.prediction["a_diag"].select(dim=1, index=2),
        a_diag2,
    )
    assert torch.allclose(
        paired_data.atmosphere_data.prediction["a_diag"].select(dim=1, index=3),
        a_diag3,
    )

    # check second o_diag computation
    o_diag1 = (a_diag2 + a_diag3) / 2 + (o_prog_ic + 1) + 1
    assert torch.allclose(
        paired_data.ocean_data.prediction["o_diag"].select(dim=1, index=1),
        o_diag1,
    )


@pytest.mark.xfail(
    raises=KeyError,
    reason=(
        "CoupledStepper doesn't yet support the use of ocean diagnostic variables as "
        "forcings in the atmosphere model."
    ),
)
def test_predict_paired_with_ocean_to_atmos_diag_forcing():
    ocean_in_names = ["o_prog", "o_sfc_temp", "a_diag", "o_mask"]
    ocean_out_names = ["o_prog", "o_sfc_temp", "o_diag"]
    atmos_in_names = ["a_prog", "a_sfc_temp", "ocean_frac", "o_diag"]
    atmos_out_names = ["a_prog", "a_sfc_temp", "a_diag"]
    coupler, coupled_data = get_stepper_and_batch(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_in_names,
        atmosphere_out_names=atmos_out_names,
        n_forward_times_ocean=2,
        n_forward_times_atmosphere=4,
        n_samples=3,
        sst_name_in_ocean_data="o_sfc_temp",
        sst_mask_name_in_ocean_data="o_mask",
        sfc_temp_name_in_atmosphere_data="a_sfc_temp",
        ocean_fraction_name="ocean_frac",
    )
    data = coupled_data.data
    atmos_prognostic_names = coupler.atmosphere.prognostic_names
    ocean_prognostic_names = coupler.ocean.prognostic_names
    atmos_prognostic = data.atmosphere_data.get_start(
        atmos_prognostic_names, n_ic_timesteps=1
    )
    ocean_prognostic = data.ocean_data.get_start(
        ocean_prognostic_names, n_ic_timesteps=1
    )
    ic = CoupledPrognosticState(
        atmosphere_data=atmos_prognostic, ocean_data=ocean_prognostic
    )
    _ = coupler.predict_paired(
        initial_condition=ic,
        forcing=data,
    )


def test_predict_paired_with_derived_variables():
    ocean_in_names = (
        [f"thetao_{i}" for i in range(NZ - 1)]
        + ["sst"]
        + [f"mask_{i}" for i in range(NZ - 1)]
    )
    ocean_out_names = ocean_in_names
    atmos_prog_names = [f"specific_total_water_{i}" for i in range(NZ - 1)] + [
        "PRESsfc",
        "surface_temperature",
    ]
    atmos_out_names = atmos_prog_names + ["LHFLX"]

    coupler, coupled_data = get_stepper_and_batch(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_prog_names + ["ocean_fraction"],
        atmosphere_out_names=atmos_out_names,
        n_forward_times_ocean=1,
        n_forward_times_atmosphere=2,
        n_samples=3,
    )
    data = coupled_data.data

    atmos_prognostic_names = coupler.atmosphere.prognostic_names
    ocean_prognostic_names = coupler.ocean.prognostic_names
    atmos_prognostic = data.atmosphere_data.get_start(
        atmos_prognostic_names, n_ic_timesteps=1
    )
    ocean_prognostic = data.ocean_data.get_start(
        ocean_prognostic_names, n_ic_timesteps=1
    )
    ic = CoupledPrognosticState(
        atmosphere_data=atmos_prognostic, ocean_data=ocean_prognostic
    )

    paired_data, _ = coupler.predict_paired(
        initial_condition=ic,
        forcing=data,
        compute_derived_variables=True,
    )
    expected_atmos_derived = ["surface_pressure_due_to_dry_air", "total_water_path"]
    for name in expected_atmos_derived:
        assert name in paired_data.atmosphere_data.prediction


def test_train_on_batch_loss():
    ocean_in_names = ["o_sfc", "ext_forcing", "o_mask"]
    ocean_out_names = ["o_sfc", "o_diag"]
    atmos_in_names = ["a_prog", "a_sfc", "ocean_frac"]
    atmos_out_names = ["a_prog", "a_sfc", "a_diag"]
    coupler, coupled_data = get_stepper_and_batch(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_in_names,
        atmosphere_out_names=atmos_out_names,
        n_forward_times_ocean=1,
        n_forward_times_atmosphere=2,
        n_samples=3,
        sst_name_in_ocean_data="o_sfc",
        sst_mask_name_in_ocean_data="o_mask",
        sfc_temp_name_in_atmosphere_data="a_sfc",
        ocean_fraction_name="ocean_frac",
    )
    output = coupler.train_on_batch(
        data=coupled_data.data,
        optimization=NullOptimization(),
    )
    loss = output.metrics["loss"]
    pred = {}
    target = {}
    for i, name in enumerate(ocean_out_names):
        # NOTE: default ocean module from _get_stepper_and_batch just multiplies
        # input fields by 2 and renames "ext_forcing" to "o_diag"
        in_name = ocean_in_names[i]
        pred[name] = 2 * coupled_data.data.ocean_data.data[in_name].select(
            dim=1, index=0
        )
        target[name] = coupled_data.data.ocean_data.data[name].select(dim=1, index=1)
    expected_loss = coupler.ocean.loss_obj(pred, target)
    assert torch.allclose(loss, expected_loss)


def test_train_on_batch_with_derived_variables():
    ocean_in_names = (
        [f"thetao_{i}" for i in range(NZ - 1)]
        + ["sst"]
        + [f"mask_{i}" for i in range(NZ - 1)]
    )
    ocean_out_names = ocean_in_names
    atmos_prog_names = [f"specific_total_water_{i}" for i in range(NZ - 1)] + [
        "PRESsfc",
        "surface_temperature",
    ]
    atmos_out_names = atmos_prog_names + ["LHFLX"]
    coupler, coupled_data = get_stepper_and_batch(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_prog_names + ["ocean_fraction"],
        atmosphere_out_names=atmos_out_names,
        n_forward_times_ocean=1,
        n_forward_times_atmosphere=2,
        n_samples=3,
    )
    output = coupler.train_on_batch(
        data=coupled_data.data,
        optimization=NullOptimization(),
        compute_derived_variables=True,
    )
    expected_atmos_derived = ["surface_pressure_due_to_dry_air", "total_water_path"]
    for name in expected_atmos_derived:
        assert name in output.atmosphere_data.gen_data


def test_reloaded_stepper_gives_same_prediction():
    torch.manual_seed(0)
    config = CoupledStepperConfig(
        atmosphere=ComponentConfig(
            timedelta="1D",
            stepper=SingleModuleStepperConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
                ),
                in_names=["a", "a_sfc", "constant_mask"],
                out_names=["a", "a_sfc"],
                normalization=NormalizationConfig(
                    means={"a": 0.0, "a_sfc": 0.0, "constant_mask": 0.0},
                    stds={"a": 1.0, "a_sfc": 1.0, "constant_mask": 1.0},
                ),
                loss=WeightedMappingLossConfig(type="MSE"),
                ocean=OceanConfig(
                    surface_temperature_name="a_sfc",
                    ocean_fraction_name="constant_mask",
                ),
            ),
        ),
        ocean=ComponentConfig(
            timedelta="2D",
            stepper=SingleModuleStepperConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet", config={"scale_factor": 1}
                ),
                in_names=["o", "o_sfc", "o_mask"],
                out_names=["o", "o_sfc"],
                normalization=NormalizationConfig(
                    means={"o": 0.0, "o_sfc": 0.0, "o_mask": 0.0},
                    stds={"o": 1.0, "o_sfc": 1.0, "o_mask": 1.0},
                ),
                loss=WeightedMappingLossConfig(type="MSE"),
                corrector=CorrectorSelector("ocean_corrector", {}),
            ),
        ),
        sst_name="o_sfc",
        sst_mask_name="o_mask",
    )
    area = torch.ones((N_LAT, N_LON), device=DEVICE)
    vertical_coordinate = CoupledVerticalCoordinate(
        ocean=DepthCoordinate(torch.arange(2), torch.ones(N_LAT, N_LON, 1)),
        atmosphere=HybridSigmaPressureCoordinate(
            ak=torch.arange(7), bk=torch.arange(7)
        ),
    )
    stepper = config.get_stepper(
        img_shape=(N_LAT, N_LON),
        gridded_operations=LatLonOperations(area),
        vertical_coordinate=vertical_coordinate,
    )
    new_stepper = CoupledStepper.from_state(stepper.get_state())
    data = get_coupled_data(
        ["o", "o_sfc", "o_mask"],
        ["a", "a_sfc", "constant_mask"],
        n_forward_times_ocean=2,
        n_forward_times_atmosphere=4,
        n_samples=1,
    )

    first_result = stepper.train_on_batch(
        data=data.data,
        optimization=NullOptimization(),
    )
    second_result = new_stepper.train_on_batch(
        data=data.data,
        optimization=NullOptimization(),
    )
    assert torch.allclose(first_result.metrics["loss"], second_result.metrics["loss"])
    assert torch.allclose(
        first_result.metrics["loss/ocean"], second_result.metrics["loss/ocean"]
    )
    assert torch.allclose(
        first_result.metrics["loss/ocean_step_0"],
        second_result.metrics["loss/ocean_step_0"],
    )
    assert torch.allclose(
        first_result.metrics["loss/ocean_step_1"],
        second_result.metrics["loss/ocean_step_1"],
    )
    assert torch.allclose(
        first_result.ocean_data.gen_data["o"], second_result.ocean_data.gen_data["o"]
    )
    assert torch.allclose(
        first_result.ocean_data.gen_data["o_sfc"],
        second_result.ocean_data.gen_data["o_sfc"],
    )
    assert torch.allclose(
        first_result.atmosphere_data.gen_data["a"],
        second_result.atmosphere_data.gen_data["a"],
    )
    assert torch.allclose(
        first_result.atmosphere_data.gen_data["a_sfc"],
        second_result.atmosphere_data.gen_data["a_sfc"],
    )
    assert torch.allclose(
        first_result.ocean_data.target_data["o"],
        second_result.ocean_data.target_data["o"],
    )
    assert torch.allclose(
        first_result.ocean_data.target_data["o_sfc"],
        second_result.ocean_data.target_data["o_sfc"],
    )
    assert torch.allclose(
        first_result.atmosphere_data.target_data["a"],
        second_result.atmosphere_data.target_data["a"],
    )
    assert torch.allclose(
        first_result.atmosphere_data.target_data["a_sfc"],
        second_result.atmosphere_data.target_data["a_sfc"],
    )
