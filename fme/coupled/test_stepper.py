import dataclasses
import datetime
from collections import namedtuple
from collections.abc import Iterable
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData
from fme.ace.stepper import StepperConfig
from fme.ace.stepper.parameter_init import ParameterInitializationConfig
from fme.core.coordinates import (
    DepthCoordinate,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
    NullVerticalCoordinate,
    VerticalCoordinate,
)
from fme.core.dataset_info import DatasetInfo
from fme.core.loss import StepLossConfig
from fme.core.mask_provider import MaskProvider
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.ocean import OceanConfig, SlabOceanConfig
from fme.core.optimization import NullOptimization
from fme.core.registry.corrector import CorrectorSelector
from fme.core.registry.module import ModuleSelector
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.coupled.dataset_info import CoupledDatasetInfo

from .data_loading.batch_data import (
    CoupledBatchData,
    CoupledPairedData,
    CoupledPrognosticState,
)
from .data_loading.data_typing import (
    CoupledHorizontalCoordinates,
    CoupledVerticalCoordinate,
)
from .stepper import (
    ComponentConfig,
    CoupledOceanFractionConfig,
    CoupledParameterInitConfig,
    CoupledStepper,
    CoupledStepperConfig,
)

NZ = 3  # number of vertical interface levels in mock data from get_data
N_LAT = 5
N_LON = 5
LON, LAT = torch.linspace(0, 360, N_LON), torch.linspace(-89.5, 89.5, N_LAT)
OCEAN_TIMEDELTA = "2D"
ATMOS_TIMEDELTA = "1D"
OCEAN_TIMESTEP = datetime.timedelta(days=2)
ATMOS_TIMESTEP = datetime.timedelta(days=1)


ATMOS_STEPPER_CONFIG = StepperConfig(
    step=StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                ),
                in_names=["a", "f"],
                out_names=["a"],
                normalization=NetworkAndLossNormalizationConfig(
                    network=NormalizationConfig(
                        means={"a": 0.0, "f": 0.0},
                        stds={"a": 1.0, "f": 1.0},
                    ),
                ),
                ocean=OceanConfig(
                    surface_temperature_name="a",
                    ocean_fraction_name="f",
                ),
            ),
        ),
    ),
)

OCEAN_STEPPER_CONFIG = StepperConfig(
    step=StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                ),
                in_names=["sst"],
                out_names=["sst"],
                normalization=NetworkAndLossNormalizationConfig(
                    network=NormalizationConfig(
                        means={"sst": 0.0},
                        stds={"sst": 1.0},
                    ),
                ),
            ),
        ),
    ),
)


@dataclasses.dataclass
class CoupledDatasetInfoBuilder:
    vcoord: CoupledVerticalCoordinate
    hcoord: CoupledHorizontalCoordinates | None = None
    ocean_timestep: datetime.timedelta = OCEAN_TIMESTEP
    atmos_timestep: datetime.timedelta = ATMOS_TIMESTEP
    ocean_mask_provider: MaskProvider = dataclasses.field(
        default_factory=lambda: MaskProvider()
    )
    atmos_mask_provider: MaskProvider = dataclasses.field(
        default_factory=lambda: MaskProvider()
    )

    def __post_init__(self):
        if self.hcoord is None:
            lat = torch.arange(N_LAT)
            lon = torch.arange(N_LON)
            self.hcoord = CoupledHorizontalCoordinates(
                ocean=LatLonCoordinates(lon=lon, lat=lat),
                atmosphere=LatLonCoordinates(lon=lon, lat=lat),
            )

    @property
    def dataset_info(self) -> CoupledDatasetInfo:
        assert self.hcoord is not None
        return CoupledDatasetInfo(
            ocean=DatasetInfo(
                horizontal_coordinates=self.hcoord.ocean,
                vertical_coordinate=self.vcoord.ocean,
                mask_provider=self.ocean_mask_provider,
                timestep=self.ocean_timestep,
            ),
            atmosphere=DatasetInfo(
                horizontal_coordinates=self.hcoord.atmosphere,
                vertical_coordinate=self.vcoord.atmosphere,
                mask_provider=self.atmos_mask_provider,
                timestep=self.atmos_timestep,
            ),
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
            ["a", "b", "c", "f"], ["c", "f"], ["b"], ["d"], ["c", "f"], []
        ),
    ),
    (
        ForcingInputs(["a", "c", "f"], ["a", "b"], ["b", "f"], ["d"]),
        ForcingExpectations(["a", "b", "c", "f"], ["c", "f"], ["b"], ["d"], ["f"], []),
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
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=atmos_in,
                        out_names=atmos_out,
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"a_sfc_temp": 0.0},
                                stds={"a_sfc_temp": 1.0},
                            ),
                        ),
                        ocean=OceanConfig(
                            surface_temperature_name="a_sfc_temp",
                            ocean_fraction_name="frac",
                        ),
                    ),
                ),
            ),
        ),
    )
    ocean_in = inputs.ocean_in + ["o_sfc_temp"]
    ocean_out = inputs.ocean_out + ["o_sfc_temp"]
    ocean = ComponentConfig(
        timedelta="12h",
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=ocean_in,
                        out_names=ocean_out,
                        next_step_forcing_names=expectations.atmos_to_ocean_forcings,
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"o_sfc_temp": 0.0},
                                stds={"o_sfc_temp": 1.0},
                            ),
                        ),
                    ),
                ),
            ),
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
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=atmos_in,
                        out_names=atmos_out,
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"atmos_surface_temp": 0.0},
                                stds={"atmos_surface_temp": 1.0},
                            ),
                        ),
                        ocean=OceanConfig(
                            surface_temperature_name="atmos_surface_temp",
                            ocean_fraction_name="frac",
                        ),
                    ),
                ),
            ),
        ),
    )
    ocean_in = inputs.ocean_in + ["ocean_surface_temp"]
    ocean_out = inputs.ocean_out + ["ocean_surface_temp"]
    ocean = ComponentConfig(
        timedelta="12h",
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=ocean_in,
                        out_names=ocean_out,
                        next_step_forcing_names=expectations.atmos_to_ocean_forcings,
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"ocean_surface_temp": 0.0},
                                stds={"ocean_surface_temp": 1.0},
                            ),
                        ),
                    ),
                ),
            ),
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


def test_config_init_atmos_stepper_with_slab_ocean_error():
    # atmosphere is required to have stepper.ocean attribute
    atmosphere = ComponentConfig(
        timedelta="6h",
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=["a", "f"],
                        out_names=["a"],
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"a": 0.0, "f": 0.0},
                                stds={"a": 1.0, "f": 1.0},
                            ),
                        ),
                        ocean=OceanConfig(
                            surface_temperature_name="a",
                            ocean_fraction_name="f",
                            slab=SlabOceanConfig(
                                mixed_layer_depth_name="mixed_layer_depth",
                                q_flux_name="q_flux",
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    ocean = ComponentConfig(
        timedelta="1h",
        stepper=OCEAN_STEPPER_CONFIG,
    )

    with pytest.raises(
        ValueError,
        match="atmosphere stepper 'ocean' config cannot use 'slab'",
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
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=["sst", "a", "b"],
                        out_names=["sst"],
                        next_step_forcing_names=["b"],
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"sst": 0.0, "a": 0.0, "b": 0.0},
                                stds={"sst": 1.0, "a": 1.0, "b": 1.0},
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r".* next_step_forcing_names: \['a'\]\."):
        _ = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)


def test_config_ocean_diag_to_atmos_forcing_error():
    atmosphere = ComponentConfig(
        timedelta="6h",
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=["a_sfc", "o_diag", "o_frac"],
                        out_names=["a_sfc", "a_diag"],
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"a_sfc": 0.0, "a_diag": 0.0},
                                stds={"a_sfc": 1.0, "a_diag": 1.0},
                            ),
                        ),
                        ocean=OceanConfig(
                            surface_temperature_name="a_sfc",
                            ocean_fraction_name="o_frac",
                        ),
                    ),
                ),
            ),
        ),
    )
    ocean = ComponentConfig(
        timedelta="5D",
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=["sst", "a_diag"],
                        out_names=["sst", "o_diag"],
                        next_step_forcing_names=["a_diag"],
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"sst": 0.0, "a_diag": 0.0},
                                stds={"sst": 1.0, "a_diag": 1.0},
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"CoupledStepper only supports ocean prognostic.*\['o_diag'\]\.",
    ):
        _ = CoupledStepperConfig(atmosphere=atmosphere, ocean=ocean)


def test_config_parameter_init_error():
    mock_param_init = Mock()
    mock_param_init.weights_path = "ckpt.pt"
    atmosphere = ComponentConfig(
        timedelta="6h",
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=["a", "f"],
                        out_names=["a"],
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"a": 0.0, "f": 0.0},
                                stds={"a": 1.0, "f": 1.0},
                            ),
                        ),
                        ocean=OceanConfig(
                            surface_temperature_name="a",
                            ocean_fraction_name="f",
                        ),
                    ),
                ),
            ),
            parameter_init=mock_param_init,
        ),
    )
    ocean = ComponentConfig(
        timedelta="5D",
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config={"scale_factor": 1, "embed_dim": 1, "num_layers": 1},
                        ),
                        in_names=["sst"],
                        out_names=["sst"],
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                means={"sst": 0.0},
                                stds={"sst": 1.0},
                            ),
                        ),
                    ),
                ),
            ),
            parameter_init=mock_param_init,
        ),
    )
    mock_coupled_param_init = Mock()
    mock_coupled_param_init.checkpoint_path = "ckpt.pt"
    with pytest.raises(
        ValueError,
        match="Please specify CoupledParameterInitConfig",
    ):
        _ = CoupledStepperConfig(
            atmosphere=atmosphere,
            ocean=ocean,
            parameter_init=mock_coupled_param_init,
        )


OCN_FRAC = CoupledOceanFractionConfig(
    sea_ice_fraction_name="sea_ice_fraction",
    land_fraction_name="land_fraction",
)

OCN_FRAC_OSIC = CoupledOceanFractionConfig(
    sea_ice_fraction_name="ocean_sea_ice_fraction",
    land_fraction_name="land_fraction",
    sea_ice_fraction_name_in_atmosphere="sea_ice_fraction",
)


# Shared parametrization data for testing various data requirements
DATA_REQUIREMENTS_TEST_CASES = [
    pytest.param(
        # atmosphere does not have sea ice input, ocean does not predict ocean fraction
        ForcingInputs(
            ["land_fraction", "ocean_frac", "sfc_temp"],
            ["sfc_temp", "a_diag"],
            ["deptho", "land_fraction", "sst", "a_diag"],
            ["sst"],
        ),
        None,
        {
            "atmos_forcing_exog": ["land_fraction", "ocean_frac"],
            "atmos_data_reqs": ["land_fraction", "ocean_frac", "sfc_temp", "a_diag"],
            "atmos_forcing_window": ["land_fraction", "ocean_frac"],
            "ocean_data_reqs": ["deptho", "sst"],
            "ocean_forcing_window": ["deptho"],  # no land_fraction
        },
        id="no_sea_ice-no_ocean_frac_pred",
    ),
    pytest.param(
        # atmosphere has sea ice input, ocean does not predict ocean fraction
        ForcingInputs(
            ["land_fraction", "ocean_frac", "sfc_temp", "sea_ice_fraction"],
            ["sfc_temp", "a_diag"],
            ["deptho", "land_fraction", "sst", "a_diag"],
            ["sst"],
        ),
        None,
        {
            "atmos_forcing_exog": ["sea_ice_fraction", "land_fraction", "ocean_frac"],
            "atmos_data_reqs": [
                "sea_ice_fraction",
                "land_fraction",
                "ocean_frac",
                "sfc_temp",
                "a_diag",
            ],
            "atmos_forcing_window": ["sea_ice_fraction", "land_fraction", "ocean_frac"],
            "ocean_data_reqs": ["deptho", "sst"],
            "ocean_forcing_window": ["deptho"],  # no land_fraction
        },
        id="with_sea_ice-no_ocean_frac_pred",
    ),
    pytest.param(
        # atmosphere has sea ice input, ocean predicts same sea ice name
        ForcingInputs(
            [
                "land_fraction",
                "ocean_frac",
                "sfc_temp",
                "_".join(
                    ["sea_ice", "fraction"]
                ),  # trick to catch str `is` identity comparison bug
            ],
            ["sfc_temp", "a_diag"],
            [
                "deptho",
                "land_fraction",
                "sst",
                "a_diag",
                "sea_ice_fraction",
            ],
            ["sst", "sea_ice_fraction"],
        ),
        OCN_FRAC,
        {
            "atmos_forcing_exog": ["land_fraction"],
            "atmos_data_reqs": ["land_fraction", "sfc_temp", "a_diag"],
            "atmos_forcing_window": [
                "land_fraction"
            ],  # no ocean_frac or sea_ice_fraction
            "ocean_data_reqs": [
                "deptho",
                "land_fraction",
                "sst",
                "sea_ice_fraction",
            ],
            "ocean_forcing_window": ["deptho"],  # no land_fraction
        },
        id="with_sea_ice-ocean_frac_pred_same_name",
    ),
    pytest.param(
        # atmosphere has sea ice input, ocean predicts a different sea ice name
        ForcingInputs(
            [
                "land_fraction",
                "ocean_frac",
                "sfc_temp",
                "_".join(
                    ["sea_ice", "fraction"]
                ),  # trick to catch str `is` identity comparison bug
            ],
            ["sfc_temp", "a_diag"],
            [
                "deptho",
                "land_fraction",
                "sst",
                "a_diag",
                "ocean_sea_ice_fraction",
            ],
            ["sst", "ocean_sea_ice_fraction"],
        ),
        OCN_FRAC_OSIC,
        {
            "atmos_forcing_exog": ["land_fraction"],
            "atmos_data_reqs": ["land_fraction", "sfc_temp", "a_diag"],
            "atmos_forcing_window": [
                "land_fraction"
            ],  # no ocean_frac or sea_ice_fraction
            "ocean_data_reqs": [
                "deptho",
                "land_fraction",
                "sst",
                "ocean_sea_ice_fraction",
            ],
            "ocean_forcing_window": ["deptho"],  # no land_fraction
        },
        id="with_sea_ice-ocean_frac_pred_different_name",
    ),
]


def create_config_for_data_requirements_test(in_out_names, ocean_fraction_prediction):
    """Helper to create stepper config for data requirements tests."""
    return get_stepper_config(
        ocean_in_names=in_out_names.ocean_in,
        ocean_out_names=in_out_names.ocean_out,
        atmosphere_in_names=in_out_names.atmos_in,
        atmosphere_out_names=in_out_names.atmos_out,
        sst_name_in_ocean_data="sst",
        sfc_temp_name_in_atmosphere_data="sfc_temp",
        ocean_fraction_name="ocean_frac",
        ocean_fraction_prediction=ocean_fraction_prediction,
    )


@pytest.mark.parametrize(
    "in_out_names, ocean_fraction_prediction, expectations",
    DATA_REQUIREMENTS_TEST_CASES,
)
def test_config_atmosphere_forcing_exogenous_names(
    in_out_names, ocean_fraction_prediction, expectations
):
    config = create_config_for_data_requirements_test(
        in_out_names, ocean_fraction_prediction
    )
    assert sorted(config.atmosphere_forcing_exogenous_names) == sorted(
        expectations["atmos_forcing_exog"]
    )


@pytest.mark.parametrize(
    "in_out_names, ocean_fraction_prediction, expectations",
    DATA_REQUIREMENTS_TEST_CASES,
)
def test_data_requirements_names(in_out_names, ocean_fraction_prediction, expectations):
    config = create_config_for_data_requirements_test(
        in_out_names, ocean_fraction_prediction
    )

    ocean_reqs = config._get_ocean_data_requirements(1)
    assert sorted(ocean_reqs.names) == sorted(expectations["ocean_data_reqs"])

    atmos_reqs = config._get_atmosphere_data_requirements(1)
    assert sorted(atmos_reqs.names) == sorted(expectations["atmos_data_reqs"])

    if ocean_fraction_prediction is not None:
        assert config.ocean_fraction_name not in atmos_reqs.names


@pytest.mark.parametrize(
    "in_out_names, ocean_fraction_prediction, expectations",
    DATA_REQUIREMENTS_TEST_CASES,
)
def test_forcing_window_data_requirements_names(
    in_out_names, ocean_fraction_prediction, expectations
):
    config = create_config_for_data_requirements_test(
        in_out_names, ocean_fraction_prediction
    )

    requirements = config.get_forcing_window_data_requirements(1)

    ocean_reqs = requirements.ocean_requirements
    assert sorted(ocean_reqs.names) == sorted(expectations["ocean_forcing_window"])

    atmos_reqs = requirements.atmosphere_requirements
    assert sorted(atmos_reqs.names) == sorted(expectations["atmos_forcing_window"])


SphericalData = namedtuple(
    "SphericalData",
    [
        "data",
        "horizontal_coord",
        "vertical_coord",
    ],
)


def get_data(
    names: Iterable[str], n_samples, n_time, realm: Literal["atmosphere", "ocean"]
) -> SphericalData:
    data_dict = {}
    for name in names:
        data_dict[name] = torch.rand(
            n_samples, n_time, N_LAT, N_LON, device=fme.get_device()
        )
    lats = torch.linspace(-89.5, 89.5, N_LAT)
    horizontal_coords = LatLonCoordinates(lat=lats, lon=torch.linspace(0, 360, N_LON))
    vertical_coord: VerticalCoordinate
    if realm == "atmosphere":
        ak, bk = torch.arange(NZ), torch.arange(NZ)
        vertical_coord = HybridSigmaPressureCoordinate(ak, bk).to(fme.get_device())
    elif realm == "ocean":
        vertical_coord = DepthCoordinate(
            torch.arange(NZ), torch.ones(N_LAT, N_LON, NZ - 1)
        ).to(fme.get_device())
    data = BatchData.new_on_device(
        data=data_dict,
        time=xr.DataArray(
            np.zeros((n_samples, n_time)),
            dims=["sample", "time"],
        ),
        labels=[set() for _ in range(n_time)],
        horizontal_dims=["lat", "lon"],
    )
    return SphericalData(data, horizontal_coords, vertical_coord)


def get_coupled_data(
    ocean_names: list[str],
    atmosphere_names: list[str],
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
        CoupledHorizontalCoordinates(
            ocean_data.horizontal_coord, atmos_data.horizontal_coord
        ),
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
    ocean_in_names: list[str],
    ocean_out_names: list[str],
    atmosphere_in_names: list[str],
    atmosphere_out_names: list[str],
    sst_name_in_ocean_data: str = "sst",
    sfc_temp_name_in_atmosphere_data: str = "surface_temperature",
    ocean_fraction_name: str = "ocean_fraction",
    ocean_builder: ModuleSelector | None = None,
    atmosphere_builder: ModuleSelector | None = None,
    ocean_timedelta: str = OCEAN_TIMEDELTA,
    atmosphere_timedelta: str = ATMOS_TIMEDELTA,
    ocean_fraction_prediction: CoupledOceanFractionConfig | None = None,
    ocean_parameter_init: ParameterInitializationConfig | None = None,
    atmosphere_parameter_init: ParameterInitializationConfig | None = None,
    checkpoint_path: str | None = None,
):
    if ocean_parameter_init is None:
        ocean_parameter_init = ParameterInitializationConfig()
    if atmosphere_parameter_init is None:
        atmosphere_parameter_init = ParameterInitializationConfig()
    # CoupledStepper requires that both component datasets include prognostic
    # surface temperature variables and that the atmosphere data includes an
    # ocean fraction forcing variable
    assert sst_name_in_ocean_data in ocean_in_names
    assert sst_name_in_ocean_data in ocean_out_names
    assert sfc_temp_name_in_atmosphere_data in atmosphere_in_names
    assert sfc_temp_name_in_atmosphere_data in atmosphere_out_names
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
            stepper=StepperConfig(
                step=StepSelector(
                    type="single_module",
                    config=dataclasses.asdict(
                        SingleModuleStepConfig(
                            builder=atmosphere_builder,
                            in_names=atmosphere_in_names,
                            out_names=atmosphere_out_names,
                            normalization=NetworkAndLossNormalizationConfig(
                                network=NormalizationConfig(
                                    means={name: 0.0 for name in atmos_norm_names},
                                    stds={name: 1.0 for name in atmos_norm_names},
                                ),
                            ),
                            ocean=OceanConfig(
                                surface_temperature_name=sfc_temp_name_in_atmosphere_data,
                                ocean_fraction_name=ocean_fraction_name,
                            ),
                        ),
                    ),
                ),
                parameter_init=atmosphere_parameter_init,
                loss=StepLossConfig(type="MSE"),
            ),
        ),
        ocean=ComponentConfig(
            timedelta=ocean_timedelta,
            stepper=StepperConfig(
                step=StepSelector(
                    type="single_module",
                    config=dataclasses.asdict(
                        SingleModuleStepConfig(
                            builder=ocean_builder,
                            in_names=ocean_in_names,
                            out_names=ocean_out_names,
                            next_step_forcing_names=next_step_forcing_names,
                            normalization=NetworkAndLossNormalizationConfig(
                                network=NormalizationConfig(
                                    means={name: 0.0 for name in ocean_norm_names},
                                    stds={name: 1.0 for name in ocean_norm_names},
                                ),
                            ),
                            corrector=CorrectorSelector("ocean_corrector", {}),
                        ),
                    ),
                ),
                parameter_init=ocean_parameter_init,
                loss=StepLossConfig(type="MSE"),
            ),
        ),
        sst_name=sst_name_in_ocean_data,
        ocean_fraction_prediction=ocean_fraction_prediction,
        parameter_init=CoupledParameterInitConfig(checkpoint_path=checkpoint_path),
    )
    return config


def get_stepper_and_batch(
    ocean_in_names: list[str],
    ocean_out_names: list[str],
    atmosphere_in_names: list[str],
    atmosphere_out_names: list[str],
    n_forward_times_ocean: int,
    n_forward_times_atmosphere: int,
    n_samples: int,
    sst_name_in_ocean_data: str = "sst",
    sfc_temp_name_in_atmosphere_data: str = "surface_temperature",
    ocean_fraction_name: str = "ocean_fraction",
    ocean_builder: ModuleSelector | None = None,
    atmosphere_builder: ModuleSelector | None = None,
    ocean_timedelta: str = OCEAN_TIMEDELTA,
    atmosphere_timedelta: str = ATMOS_TIMEDELTA,
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
        sfc_temp_name_in_atmosphere_data=sfc_temp_name_in_atmosphere_data,
        ocean_fraction_name=ocean_fraction_name,
        ocean_builder=ocean_builder,
        atmosphere_builder=atmosphere_builder,
        # NOTE: these values don't actually matter here because they aren't used
        # when stepping the batch forward... if you need consistency between the
        # timedeltas and batch time dims then you should use
        # n_forward_times_atmosphere = 2 * n_forward_times_ocean
        ocean_timedelta=ocean_timedelta,
        atmosphere_timedelta=atmosphere_timedelta,
    )
    dataset_info = CoupledDatasetInfoBuilder(
        vcoord=coupled_data.vertical_coord
    ).dataset_info
    coupler = config.get_stepper(dataset_info)
    return coupler, coupled_data


@pytest.mark.parametrize(
    "ocean_fraction_prediction, sea_ice_frac_is_ocean_prog",
    [
        (None, True),
        (None, False),
        (
            CoupledOceanFractionConfig(
                sea_ice_fraction_name="sea_ice_fraction",
                land_fraction_name="land_fraction",
            ),
            True,  # NOTE: required
        ),
        (
            CoupledOceanFractionConfig(
                sea_ice_fraction_name="ocean_sea_ice_fraction",
                land_fraction_name="land_fraction",
                sea_ice_fraction_name_in_atmosphere="sea_ice_fraction",
            ),
            True,  # NOTE: required
        ),
    ],
)
@pytest.mark.parametrize("sea_ice_frac_is_input_to_atmos", [False, True])
def test__get_atmosphere_forcings(
    ocean_fraction_prediction,
    sea_ice_frac_is_input_to_atmos,
    sea_ice_frac_is_ocean_prog,
):
    torch.manual_seed(0)
    ocean_in_names = ["land_fraction", "sst", "a_diag"]
    ocean_out_names = ["sst"]
    sea_ice_frac_name = "sea_ice_frac"
    sea_ice_frac_name_in_atmos = sea_ice_frac_name
    if ocean_fraction_prediction:
        sea_ice_frac_name = ocean_fraction_prediction.sea_ice_fraction_name
        sea_ice_frac_name_in_atmos = (
            ocean_fraction_prediction.sea_ice_fraction_name_in_atmosphere
            or sea_ice_frac_name
        )
    if sea_ice_frac_is_ocean_prog:
        ocean_in_names.append(sea_ice_frac_name)
        ocean_out_names.append(sea_ice_frac_name)
    atmos_in_names = ["land_fraction", "ocean_frac", "sfc_temp"]
    atmos_out_names = ["sfc_temp", "a_diag"]
    if sea_ice_frac_is_input_to_atmos:
        atmos_in_names.append(sea_ice_frac_name)
    config = get_stepper_config(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_in_names,
        atmosphere_out_names=atmos_out_names,
        sst_name_in_ocean_data="sst",
        sfc_temp_name_in_atmosphere_data="sfc_temp",
        ocean_fraction_name="ocean_frac",
        ocean_fraction_prediction=ocean_fraction_prediction,
    )
    vertical_coord = Mock(spec=CoupledVerticalCoordinate)
    vertical_coord.atmosphere = NullVerticalCoordinate()
    vertical_coord.ocean = NullVerticalCoordinate()
    sst_mask = torch.ones(N_LAT, N_LON).to(fme.get_device())
    sst_mask[0, 0] = 0
    dataset_info = CoupledDatasetInfoBuilder(
        vcoord=vertical_coord,
        ocean_mask_provider=MaskProvider({"mask_2d": sst_mask}),
    ).dataset_info
    coupler = config.get_stepper(dataset_info)
    shape_ocean = (1, 1, N_LAT, N_LON)
    shape_atmos = (1, coupler.n_inner_steps + 1, N_LAT, N_LON)
    forcings_from_ocean = {
        sea_ice_frac_name: torch.rand(*shape_ocean, device=fme.get_device()),
        "sst": torch.rand(*shape_ocean, device=fme.get_device()),
    }
    for tensor in forcings_from_ocean.values():
        # apply mask to ocean data
        tensor[..., 0, 0] = float("nan")
    atmos_forcing_data = {
        "land_fraction": torch.rand(*shape_atmos, device=fme.get_device()),
        "ocean_frac": torch.rand(*shape_atmos, device=fme.get_device()),
    }
    expected_forcings_from_ocean = {
        k: v.clone() for k, v in forcings_from_ocean.items()
    }
    if ocean_fraction_prediction is None:
        expected_forcings_from_ocean["ocean_frac"] = atmos_forcing_data[
            "ocean_frac"
        ].clone()
    elif ocean_fraction_prediction.sea_ice_fraction_name == "sea_ice_fraction":
        expected_forcings_from_ocean["ocean_frac"] = torch.clip(
            1
            - (
                atmos_forcing_data["land_fraction"]
                + forcings_from_ocean[sea_ice_frac_name]
            ),
            min=0.0,
        )
    elif ocean_fraction_prediction.sea_ice_fraction_name == "ocean_sea_ice_fraction":
        # back sea_ice_fraction out of ocean_sea_ice_fraction
        sic = forcings_from_ocean[sea_ice_frac_name] * (
            1 - atmos_forcing_data["land_fraction"]
        )
        expected_forcings_from_ocean[sea_ice_frac_name_in_atmos] = sic
        expected_forcings_from_ocean["ocean_frac"] = torch.clip(
            1 - (atmos_forcing_data["land_fraction"] + sic),
            min=0.0,
        )
    else:
        sea_ice_fraction_name = ocean_fraction_prediction.sea_ice_fraction_name
        raise ValueError(
            "test__get_atmosphere_forcings has CoupledOceanFractionConfig with "
            f"incompatible value {sea_ice_fraction_name=}"
        )

    expected_forcings_from_ocean["ocean_frac"][:, :, 0, 0] = 0.0
    expected_atmos_forcings = {
        "land_fraction": atmos_forcing_data["land_fraction"].clone(),
        "ocean_frac": expected_forcings_from_ocean["ocean_frac"].clone(),
        "sfc_temp": forcings_from_ocean["sst"].clone().expand(*shape_atmos),
    }
    if sea_ice_frac_is_input_to_atmos:
        if ocean_fraction_prediction is None and not sea_ice_frac_is_ocean_prog:
            # sea ice frac comes from atmosphere
            atmos_forcing_data[sea_ice_frac_name_in_atmos] = torch.rand(
                *shape_atmos, device=fme.get_device()
            )
            expected_atmos_forcings[sea_ice_frac_name_in_atmos] = atmos_forcing_data[
                sea_ice_frac_name_in_atmos
            ].clone()
        else:
            # sea ice frac comes from the ocean
            expected_atmos_forcings[sea_ice_frac_name_in_atmos] = (
                expected_forcings_from_ocean[sea_ice_frac_name_in_atmos]
                .clone()
                .expand(*shape_atmos)
            )
    new_atmos_forcings = coupler._get_atmosphere_forcings(
        atmos_forcing_data, forcings_from_ocean
    )
    for name in expected_atmos_forcings:
        torch.testing.assert_close(
            new_atmos_forcings[name], torch.nan_to_num(expected_atmos_forcings[name])
        )


def test__get_ocean_forcings():
    torch.manual_seed(0)
    ocean_in_names = ["o_exog", "exog", "sst", "a_diag"]
    ocean_out_names = ["sst"]
    atmos_in_names = ["exog", "ocean_frac", "sfc_temp"]
    atmos_out_names = ["a_diag", "sfc_temp"]
    config = get_stepper_config(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_in_names,
        atmosphere_out_names=atmos_out_names,
        sst_name_in_ocean_data="sst",
        sfc_temp_name_in_atmosphere_data="sfc_temp",
        ocean_fraction_name="ocean_frac",
    )
    vertical_coord = Mock(spec=CoupledVerticalCoordinate)
    vertical_coord.atmosphere = NullVerticalCoordinate()
    vertical_coord.ocean = NullVerticalCoordinate()
    dataset_info = CoupledDatasetInfoBuilder(vcoord=vertical_coord).dataset_info
    coupler = config.get_stepper(dataset_info)
    ocean_shape = (1, 2, N_LAT, N_LON)
    atmos_shape = (1, 2, N_LAT, N_LON)
    ocean_data = {
        "o_exog": torch.rand(*ocean_shape, device=fme.get_device()),
        "sst": torch.rand(*ocean_shape, device=fme.get_device()),
    }
    atmos_gen = {"a_diag": torch.rand(*atmos_shape, device=fme.get_device())}
    atmos_forcings = {"exog": torch.rand(*atmos_shape, device=fme.get_device())}
    expected_ocean_forcings = {
        "o_exog": ocean_data["o_exog"].clone(),
        "exog": atmos_forcings["exog"].mean(dim=1),
        "a_diag": atmos_gen["a_diag"].mean(dim=1),
    }
    new_ocean_forcings = coupler._get_ocean_forcings(
        ocean_data, atmos_gen, atmos_forcings
    )
    assert new_ocean_forcings.keys() == expected_ocean_forcings.keys()
    # next step forcing
    torch.testing.assert_close(
        new_ocean_forcings["a_diag"][:, 1], expected_ocean_forcings["a_diag"]
    )
    assert torch.all(new_ocean_forcings["a_diag"][:, 0].isnan())
    # shared exogenous forcing
    torch.testing.assert_close(
        new_ocean_forcings["exog"][:, 0], expected_ocean_forcings["exog"]
    )
    assert torch.all(new_ocean_forcings["exog"][:, 1].isnan())
    # ocean-specific exogenous forcing
    torch.testing.assert_close(
        new_ocean_forcings["o_exog"], expected_ocean_forcings["o_exog"]
    )


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
        torch.testing.assert_close(atmos_sst, ocean_sst)

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
        torch.testing.assert_close(atmos_sst, ocean_sst)

    a_prog_ic = data.atmosphere_data.data["a_prog"].select(dim=1, index=0)
    o_prog_ic = data.ocean_data.data["o_prog"].select(dim=1, index=0)

    # check o_prog computations
    for i in range(2):
        o_prog_pred = o_prog_ic + (i + 1)
        torch.testing.assert_close(
            paired_data.ocean_data.prediction["o_prog"].select(dim=1, index=i),
            o_prog_pred,
        )

    # check a_prog computations
    for i in range(4):
        a_prog_pred = a_prog_ic + (i + 1) * 2
        torch.testing.assert_close(
            paired_data.atmosphere_data.prediction["a_prog"].select(dim=1, index=i),
            a_prog_pred,
        )

    # check first two a_diag computations
    a_diag0 = a_prog_ic + o_prog_ic + 2
    a_diag1 = (a_prog_ic + 2) + o_prog_ic + 2
    torch.testing.assert_close(
        paired_data.atmosphere_data.prediction["a_diag"].select(dim=1, index=0),
        a_diag0,
    )
    torch.testing.assert_close(
        paired_data.atmosphere_data.prediction["a_diag"].select(dim=1, index=1),
        a_diag1,
    )

    # check first o_diag computation
    o_diag0 = (a_diag0 + a_diag1) / 2 + o_prog_ic + 1
    torch.testing.assert_close(
        paired_data.ocean_data.prediction["o_diag"].select(dim=1, index=0),
        o_diag0,
    )

    # check next two a_diag computations
    a_diag2 = (a_prog_ic + 2 + 2) + (o_prog_ic + 1) + 2
    a_diag3 = (a_prog_ic + 2 + 2 + 2) + (o_prog_ic + 1) + 2
    torch.testing.assert_close(
        paired_data.atmosphere_data.prediction["a_diag"].select(dim=1, index=2),
        a_diag2,
    )
    torch.testing.assert_close(
        paired_data.atmosphere_data.prediction["a_diag"].select(dim=1, index=3),
        a_diag3,
    )

    # check second o_diag computation
    o_diag1 = (a_diag2 + a_diag3) / 2 + (o_prog_ic + 1) + 1
    torch.testing.assert_close(
        paired_data.ocean_data.prediction["o_diag"].select(dim=1, index=1),
        o_diag1,
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
        assert name in output.atmosphere.gen_data


def test_reloaded_stepper_gives_same_prediction():
    torch.manual_seed(0)
    config = CoupledStepperConfig(
        atmosphere=ComponentConfig(
            timedelta="1D",
            stepper=StepperConfig(
                step=StepSelector(
                    type="single_module",
                    config=dataclasses.asdict(
                        SingleModuleStepConfig(
                            builder=ModuleSelector(
                                type="SphericalFourierNeuralOperatorNet",
                                config={"scale_factor": 1},
                            ),
                            in_names=["a", "a_sfc", "constant_mask"],
                            out_names=["a", "a_sfc"],
                            normalization=NetworkAndLossNormalizationConfig(
                                network=NormalizationConfig(
                                    means={
                                        "a": 0.0,
                                        "a_sfc": 0.0,
                                        "constant_mask": 0.0,
                                    },
                                    stds={"a": 1.0, "a_sfc": 1.0, "constant_mask": 1.0},
                                ),
                            ),
                            ocean=OceanConfig(
                                surface_temperature_name="a_sfc",
                                ocean_fraction_name="constant_mask",
                            ),
                        ),
                    ),
                ),
                loss=StepLossConfig(type="MSE"),
            ),
        ),
        ocean=ComponentConfig(
            timedelta="2D",
            stepper=StepperConfig(
                step=StepSelector(
                    type="single_module",
                    config=dataclasses.asdict(
                        SingleModuleStepConfig(
                            builder=ModuleSelector(
                                type="SphericalFourierNeuralOperatorNet",
                                config={"scale_factor": 1},
                            ),
                            in_names=["o", "o_sfc", "o_mask"],
                            out_names=["o", "o_sfc"],
                            normalization=NetworkAndLossNormalizationConfig(
                                network=NormalizationConfig(
                                    means={"o": 0.0, "o_sfc": 0.0, "o_mask": 0.0},
                                    stds={"o": 1.0, "o_sfc": 1.0, "o_mask": 1.0},
                                ),
                            ),
                            corrector=CorrectorSelector("ocean_corrector", {}),
                        ),
                    ),
                ),
                loss=StepLossConfig(type="MSE"),
            ),
        ),
        sst_name="o_sfc",
    )
    vertical_coord = Mock(spec=CoupledVerticalCoordinate)
    vertical_coord.atmosphere = NullVerticalCoordinate()
    vertical_coord.ocean = NullVerticalCoordinate()
    dataset_info = CoupledDatasetInfoBuilder(vcoord=vertical_coord).dataset_info
    stepper = config.get_stepper(dataset_info)
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
    torch.testing.assert_close(
        first_result.total_metrics["loss"], second_result.total_metrics["loss"]
    )
    for metric in ["loss/ocean", "loss/ocean_step_0", "loss/ocean_step_1"]:
        torch.testing.assert_close(
            first_result.ocean.metrics[metric], second_result.ocean.metrics[metric]
        )
    for name in ["o", "o_sfc"]:
        torch.testing.assert_close(
            first_result.ocean.gen_data[name], second_result.ocean.gen_data[name]
        )
        torch.testing.assert_close(
            first_result.ocean.target_data[name],
            second_result.ocean.target_data[name],
        )
    for metric in [
        "loss/atmosphere",
        "loss/atmosphere_step_0",
        "loss/atmosphere_step_1",
        "loss/atmosphere_step_2",
        "loss/atmosphere_step_3",
    ]:
        torch.testing.assert_close(
            first_result.atmosphere.metrics[metric],
            second_result.atmosphere.metrics[metric],
        )
    for name in ["a", "a_sfc"]:
        torch.testing.assert_close(
            first_result.atmosphere.gen_data[name],
            second_result.atmosphere.gen_data[name],
        )
        torch.testing.assert_close(
            first_result.atmosphere.target_data[name],
            second_result.atmosphere.target_data[name],
        )


def test_set_train_eval():
    stepper, _ = get_stepper_and_batch(
        ["sst", "mask_0"],
        ["sst"],
        ["surface_temperature", "ocean_fraction"],
        ["surface_temperature"],
        1,
        1,
        1,
    )
    for module in stepper.modules:
        assert module.training
    stepper.set_eval()
    for module in stepper.modules:
        assert not module.training
    stepper.set_train()
    for module in stepper.modules:
        assert module.training
