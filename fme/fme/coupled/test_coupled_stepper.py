from collections import namedtuple
from unittest.mock import MagicMock

import pytest
import torch

import fme
from fme.ace.data_loading.config import XarrayDataConfig
from fme.ace.stepper import SingleModuleStepperConfig
from fme.ace.test_stepper import get_scalar_data
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.ocean import OceanConfig
from fme.core.registry.module import ModuleSelector

from .data_loading.config import CoupledDataConfig, CoupledDataLoaderConfig
from .data_loading.getters import get_coupled_data_loader
from .stepper import CoupledComponentConfig, CoupledStepperConfig

DEVICE = fme.get_device()


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
    in_names=["a"],
    out_names=["a"],
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
    atmos_in = inputs.atmos_in + ["surface_temp", "frac"]
    atmos_out = inputs.atmos_out + ["surface_temp"]
    atmosphere = CoupledComponentConfig(
        timedelta="6h",
        stepper=SingleModuleStepperConfig(
            builder=MagicMock(),
            in_names=atmos_in,
            out_names=atmos_out,
            normalization=MagicMock(),
            loss=MagicMock(),
            ocean=OceanConfig(
                surface_temperature_name="surface_temp",
                ocean_fraction_name="frac",
            ),
        ),
    )
    ocean_in = inputs.ocean_in + ["surface_temp"]
    ocean_out = inputs.ocean_out + ["surface_temp"]
    ocean = CoupledComponentConfig(
        timedelta="12h",
        stepper=SingleModuleStepperConfig(
            builder=MagicMock(),
            in_names=ocean_in,
            out_names=ocean_out,
            normalization=MagicMock(),
            loss=MagicMock(),
        ),
    )
    config = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)
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
        expectations.ocean_to_atmos_forcings + ["surface_temp"]
    )
    assert sorted(config._all_ocean_names) == sorted(
        expectations.all_ocean + ["surface_temp"]
    )
    assert sorted(config._all_atmosphere_names) == sorted(
        expectations.all_atmos
        + [
            "frac",
            "surface_temp",
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
    atmosphere = CoupledComponentConfig(
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
    ocean = CoupledComponentConfig(
        timedelta="12h",
        surface_temperature_name="ocean_surface_temp",
        stepper=SingleModuleStepperConfig(
            builder=MagicMock(),
            in_names=ocean_in,
            out_names=ocean_out,
            normalization=MagicMock(),
            loss=MagicMock(),
        ),
    )
    config = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)
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
    atmosphere = CoupledComponentConfig(
        timedelta=timedelta_atmos,
        stepper=ATMOS_STEPPER_CONFIG,
    )
    ocean = CoupledComponentConfig(
        timedelta=timedelta_ocean,
        stepper=OCEAN_STEPPER_CONFIG,
    )
    config = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)
    assert config.n_inner_steps == expected_n_inner_steps


def test_config_init_atmos_stepper_missing_ocean_error():
    # atmosphere is required to have stepper.ocean attribute
    atmosphere = CoupledComponentConfig(
        timedelta="6h",
        stepper=OCEAN_STEPPER_CONFIG,
    )
    ocean = CoupledComponentConfig(
        timedelta="1h",
        stepper=OCEAN_STEPPER_CONFIG,
    )

    with pytest.raises(ValueError) as err:
        _ = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)
    assert "atmosphere stepper 'ocean' config is missing" in str(err.value)


def test_config_init_timedelta_comparison_error():
    # atmosphere timedelta > ocean timedelta raises error
    atmosphere = CoupledComponentConfig(
        timedelta="6h",
        stepper=ATMOS_STEPPER_CONFIG,
    )
    ocean = CoupledComponentConfig(
        timedelta="1h",
        stepper=OCEAN_STEPPER_CONFIG,
    )

    with pytest.raises(ValueError) as err:
        _ = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)
    assert "Atmosphere timedelta must not be larger" in str(err.value)


def test_config_init_incompatible_timedelta_error():
    # ocean timestep % atmosphere timestep != 0 raises error
    atmosphere = CoupledComponentConfig(
        timedelta="2h",
        stepper=ATMOS_STEPPER_CONFIG,
    )
    ocean = CoupledComponentConfig(
        timedelta="5h",
        stepper=OCEAN_STEPPER_CONFIG,
    )
    with pytest.raises(ValueError) as err:
        _ = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)
    assert "Ocean timedelta must be a multiple" in str(err.value)


def _validate_mock_coupled_data_names(mock_data, ocean_names, atmos_names):
    ocean_diff = set(ocean_names).difference(mock_data.ocean.data_vars)
    if len(ocean_diff) > 0:
        raise ValueError(
            f"Missing expected ocean variables {ocean_diff} in mock_coupled_data. "
            f"Did the fixture change?"
        )
    atmos_diff = set(atmos_names).difference(mock_data.atmosphere.data_vars)
    if len(atmos_diff) > 0:
        raise ValueError(
            f"Missing expected atmosphere variables {atmos_diff} in mock_coupled_data. "
            f"Did the fixture change?"
        )


@pytest.mark.parametrize(
    "keep_initial_condition",
    [True, False],
)
@pytest.mark.xfail(reason="to be fixed when updating coupler for inference")
def test_train_on_batch_integration(mock_coupled_data, keep_initial_condition):
    # get the dataset
    ocean_names = ["o_exog", "o_prog", "o_sfc"]
    atmos_names = ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"]
    _validate_mock_coupled_data_names(mock_coupled_data, ocean_names, atmos_names)

    # create the coupled stepper
    def atmos_transform(x, forcing):
        forcing = forcing.expand(*x.shape)
        return (x + torch.sqrt((x - 0.5 * forcing) ** 2)) / 2

    class AtmosModule(torch.nn.Module):
        def forward(self, x):
            return atmos_transform(x[:, :3], x[:, :1])

    def ocean_transform(x):
        return torch.log(torch.exp(x) + 2)

    class OceanModule(torch.nn.Module):
        def forward(self, x):
            return ocean_transform(x[:, :2] + x[:, 2:])

    config = CoupledStepperConfig(
        atmosphere=CoupledComponentConfig(
            timedelta="1D",
            stepper=SingleModuleStepperConfig(
                builder=ModuleSelector(
                    type="prebuilt", config={"module": AtmosModule()}
                ),
                in_names=["a_exog", "a_prog", "a_sfc", "constant_mask"],
                out_names=["a_diag", "a_prog", "a_sfc"],
                normalization=NormalizationConfig(
                    means=get_scalar_data(
                        ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"], 0.0
                    ),
                    stds=get_scalar_data(
                        ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"], 1.0
                    ),
                ),
                loss=WeightedMappingLossConfig(type="MSE"),
                ocean=OceanConfig(
                    surface_temperature_name="a_sfc",
                    ocean_fraction_name="constant_mask",
                ),
            ),
        ),
        ocean=CoupledComponentConfig(
            timedelta="2D",
            surface_temperature_name="o_sfc",
            stepper=SingleModuleStepperConfig(
                builder=ModuleSelector(
                    type="prebuilt", config={"module": OceanModule()}
                ),
                in_names=["o_prog", "a_diag", "o_exog", "o_sfc"],
                out_names=["o_prog", "o_sfc"],
                normalization=NormalizationConfig(
                    means=get_scalar_data(["a_diag", "o_exog", "o_prog", "o_sfc"], 0.0),
                    stds=get_scalar_data(["a_diag", "o_exog", "o_prog", "o_sfc"], 1.0),
                ),
                loss=WeightedMappingLossConfig(type="MSE"),
            ),
        ),
    )
    gridded_operations = LatLonOperations(torch.ones((5, 5), device=DEVICE))
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )

    coupler = config.get_stepper(
        img_shape=(5, 5),
        gridded_operations=gridded_operations,
        vertical_coordinate=vertical_coordinate,
    )

    # get the data loader
    data_config = CoupledDataLoaderConfig(
        dataset=[
            CoupledDataConfig(
                ocean=XarrayDataConfig(data_path=mock_coupled_data.ocean_dir),
                atmosphere=XarrayDataConfig(data_path=mock_coupled_data.atmosphere_dir),
            )
        ],
        batch_size=1,
        num_data_workers=0,
        strict_ensemble=True,
    )
    coupled_requirements = config.get_data_requirements(n_coupled_steps=2)
    assert coupled_requirements.ocean_requirements.n_timesteps == 3
    assert coupled_requirements.atmosphere_requirements.n_timesteps == 5
    assert sorted(coupled_requirements.ocean_requirements.names) == [
        "o_exog",
        "o_prog",
        "o_sfc",
    ]
    assert sorted(coupled_requirements.atmosphere_requirements.names) == [
        "a_diag",
        "a_exog",
        "a_prog",
        "a_sfc",
        "constant_mask",
    ]

    # unshuffled data loader
    data = get_coupled_data_loader(data_config, False, coupled_requirements)

    # this check is in case of an external change in Packer
    if coupler.ocean.in_packer.names != ["o_prog", "a_diag", "o_exog", "o_sfc"]:
        raise ValueError(
            "Ocean stepper input packer name ordering now differs from config order, "
            "which breaks an important assumption in testing CoupledStepper."
        )

    batch = next(iter(data.loader))
    stepped = coupler.train_on_batch(
        batch,
        optimization=MagicMock(),
        keep_initial_condition=keep_initial_condition,
    )

    assert "loss" in stepped.metrics

    gen_data_atmos = stepped.atmosphere_data.gen_data
    gen_data_ocean = stepped.ocean_data.gen_data

    assert set(gen_data_atmos.keys()) == set(["a_diag", "a_prog", "a_sfc"])
    assert set(gen_data_ocean.keys()) == set(["o_prog", "o_sfc"])
    assert len(gen_data_atmos["a_diag"].shape) == 4
    assert len(gen_data_ocean["o_prog"].shape) == 4

    data_atmos = batch.atmosphere_data.data
    data_ocean = batch.ocean_data.data

    tdim = coupler.TIME_DIM

    if keep_initial_condition:
        ic_offset = 1
        for var in gen_data_atmos:
            assert torch.allclose(
                gen_data_atmos[var].select(tdim, 0), data_atmos[var].select(tdim, 0)
            )
        for var in gen_data_ocean:
            assert torch.allclose(
                gen_data_ocean[var].select(tdim, 0), data_ocean[var].select(tdim, 0)
            )
    else:
        ic_offset = 0

    assert gen_data_atmos["a_diag"].shape[1] == 4 + ic_offset
    assert gen_data_ocean["o_prog"].shape[1] == 2 + ic_offset

    # atmosphere prognostic var is not influenced by ocean
    atmos_prog1 = atmos_transform(
        data_atmos["a_prog"].select(tdim, 0),
        forcing=data_atmos["a_exog"].select(tdim, 0),
    )
    atmos_prog2 = atmos_transform(
        atmos_prog1,
        forcing=data_atmos["a_exog"].select(tdim, 1),
    )
    atmos_prog3 = atmos_transform(
        atmos_prog2,
        forcing=data_atmos["a_exog"].select(tdim, 2),
    )
    atmos_prog4 = atmos_transform(
        atmos_prog3,
        forcing=data_atmos["a_exog"].select(tdim, 3),
    )
    assert torch.allclose(
        gen_data_atmos["a_prog"].select(tdim, 0 + ic_offset), atmos_prog1
    )
    assert torch.allclose(
        gen_data_atmos["a_prog"].select(tdim, 1 + ic_offset), atmos_prog2
    )
    assert torch.allclose(
        gen_data_atmos["a_prog"].select(tdim, 2 + ic_offset), atmos_prog3
    )
    assert torch.allclose(
        gen_data_atmos["a_prog"].select(tdim, 3 + ic_offset), atmos_prog4
    )

    # atmosphere diag is transformed version of the exog forcing
    atmos_diag1 = atmos_transform(
        data_atmos["a_exog"].select(tdim, 0),
        forcing=data_atmos["a_exog"].select(tdim, 0),
    )
    atmos_diag2 = atmos_transform(
        data_atmos["a_exog"].select(tdim, 1),
        forcing=data_atmos["a_exog"].select(tdim, 1),
    )
    atmos_diag3 = atmos_transform(
        data_atmos["a_exog"].select(tdim, 2),
        forcing=data_atmos["a_exog"].select(tdim, 2),
    )
    atmos_diag4 = atmos_transform(
        data_atmos["a_exog"].select(tdim, 3),
        forcing=data_atmos["a_exog"].select(tdim, 3),
    )
    assert torch.allclose(
        gen_data_atmos["a_diag"].select(tdim, 0 + ic_offset), atmos_diag1
    )
    assert torch.allclose(
        gen_data_atmos["a_diag"].select(tdim, 1 + ic_offset), atmos_diag2
    )
    assert torch.allclose(
        gen_data_atmos["a_diag"].select(tdim, 2 + ic_offset), atmos_diag3
    )
    assert torch.allclose(
        gen_data_atmos["a_diag"].select(tdim, 3 + ic_offset), atmos_diag4
    )

    # atmosphere surface temp is replace by ocean surface temp
    assert torch.allclose(
        gen_data_atmos["a_sfc"].select(tdim, 0 + ic_offset),
        data_ocean["o_sfc"].select(tdim, 0),
    )
    assert torch.allclose(
        gen_data_atmos["a_sfc"].select(tdim, 1 + ic_offset),
        data_ocean["o_sfc"].select(tdim, 0),
    )
    # use first ocean generated step
    assert torch.allclose(
        gen_data_atmos["a_sfc"].select(tdim, 2 + ic_offset),
        gen_data_ocean["o_sfc"].select(tdim, 0 + ic_offset),
    )
    assert torch.allclose(
        gen_data_atmos["a_sfc"].select(tdim, 3 + ic_offset),
        gen_data_ocean["o_sfc"].select(tdim, 0 + ic_offset),
    )

    ocean_prog1 = ocean_transform(
        data_ocean["o_prog"].select(tdim, 0) + data_ocean["o_exog"].select(tdim, 0)
    )
    ocean_prog2 = ocean_transform(ocean_prog1 + data_ocean["o_exog"].select(tdim, 1))
    assert torch.allclose(
        gen_data_ocean["o_prog"].select(tdim, 0 + ic_offset), ocean_prog1
    )
    assert torch.allclose(
        gen_data_ocean["o_prog"].select(tdim, 1 + ic_offset), ocean_prog2
    )

    # ocean surface temp is transformed IC ocean surface temp + time-mean atmos diag
    atmos_diag12 = 0.5 * (atmos_diag1 + atmos_diag2)
    atmos_diag34 = 0.5 * (atmos_diag3 + atmos_diag4)
    ocean_sfc1 = ocean_transform(data_ocean["o_sfc"].select(tdim, 0) + atmos_diag12)
    ocean_sfc2 = ocean_transform(ocean_sfc1 + atmos_diag34)
    assert torch.allclose(
        gen_data_ocean["o_sfc"].select(tdim, 0 + ic_offset), ocean_sfc1
    )
    assert torch.allclose(
        gen_data_ocean["o_sfc"].select(tdim, 1 + ic_offset), ocean_sfc2
    )
