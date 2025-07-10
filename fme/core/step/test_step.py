import dataclasses
import datetime
import pathlib
import tempfile
import unittest
import unittest.mock
from collections.abc import Callable

import dacite
import pytest
import torch
import torch._dynamo.exc
from torch import nn

import fme
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.ace.testing.fv3gfs_data import get_scalar_dataset
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig, EnergyBudgetConfig
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import DummyWrapper
from fme.core.multi_call import MultiCallConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.multi_call import MultiCallStepConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepABC, StepSelector
from fme.core.typing_ import TensorDict

from .radiation import SeparateRadiationStepConfig

DEFAULT_IMG_SHAPE = (45, 90)


def get_network_and_loss_normalization_config(
    names: list[str],
    dir: pathlib.Path | None = None,
) -> NetworkAndLossNormalizationConfig:
    if dir is None:
        return NetworkAndLossNormalizationConfig(
            network=NormalizationConfig(
                means={name: 0.0 for name in names},
                stds={name: 1.0 for name in names},
            ),
        )
    else:
        return NetworkAndLossNormalizationConfig(
            network=NormalizationConfig(
                global_means_path=dir / "means.nc",
                global_stds_path=dir / "stds.nc",
            ),
        )


def get_separate_radiation_config(
    dir: pathlib.Path | None = None,
) -> SeparateRadiationStepConfig:
    normalization = get_network_and_loss_normalization_config(
        names=[
            "prog_a",
            "prog_b",
            "forcing_shared",
            "forcing_rad",
            "diagnostic_rad",
            "diagnostic_main",
        ],
        dir=dir,
    )

    return SeparateRadiationStepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={
                "scale_factor": 1,
                "embed_dim": 4,
                "num_layers": 2,
            },
        ),
        radiation_builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={
                "scale_factor": 1,
                "embed_dim": 4,
                "num_layers": 2,
            },
        ),
        main_prognostic_names=["prog_a", "prog_b"],
        shared_forcing_names=["forcing_shared"],
        radiation_only_forcing_names=["forcing_rad"],
        radiation_diagnostic_names=["diagnostic_rad"],
        main_diagnostic_names=["diagnostic_main"],
        normalization=normalization,
    )


def get_separate_radiation_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    return StepSelector(
        type="separate_radiation",
        config=dataclasses.asdict(get_separate_radiation_config(dir)),
    )


def get_single_module_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    normalization = get_network_and_loss_normalization_config(
        names=[
            "forcing_shared",
            "forcing_rad",
            "diagnostic_main",
            "diagnostic_rad",
        ],
        dir=dir,
    )
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={
                        "scale_factor": 1,
                        "embed_dim": 4,
                        "num_layers": 2,
                    },
                ),
                in_names=["forcing_shared", "forcing_rad"],
                out_names=["diagnostic_main", "diagnostic_rad"],
                normalization=normalization,
            ),
        ),
    )


def get_single_module_noise_conditioned_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    normalization = get_network_and_loss_normalization_config(
        names=[
            "forcing_shared",
            "forcing_rad",
            "diagnostic_main",
            "diagnostic_rad",
        ],
        dir=dir,
    )
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="NoiseConditionedSFNO",
                    config=dataclasses.asdict(
                        NoiseConditionedSFNOBuilder(
                            embed_dim=4,
                            noise_embed_dim=4,
                            noise_type="isotropic",
                            num_layers=2,
                            local_blocks=[0],
                        )
                    ),
                ),
                in_names=["forcing_shared", "forcing_rad"],
                out_names=["diagnostic_main", "diagnostic_rad"],
                normalization=normalization,
            ),
        ),
    )


def get_single_module_with_atmosphere_corrector_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    in_names = [
        "DSWRFtoa",
        "HGTsfc",
        "air_temperature_0",
        "air_temperature_1",
        "air_temperature_2",
        "air_temperature_3",
        "air_temperature_4",
        "air_temperature_5",
        "specific_total_water_0",
        "specific_total_water_1",
        "specific_total_water_2",
        "specific_total_water_3",
        "specific_total_water_4",
        "specific_total_water_5",
        "PRESsfc",
        "PRATEsfc",
    ]
    out_names = [
        "air_temperature_0",
        "air_temperature_1",
        "air_temperature_2",
        "air_temperature_3",
        "air_temperature_4",
        "air_temperature_5",
        "specific_total_water_0",
        "specific_total_water_1",
        "specific_total_water_2",
        "specific_total_water_3",
        "specific_total_water_4",
        "specific_total_water_5",
        "PRESsfc",
        "PRATEsfc",
        "LHTFLsfc",
        "SHTFLsfc",
        "ULWRFsfc",
        "ULWRFtoa",
        "DLWRFsfc",
        "DSWRFsfc",
        "USWRFtoa",
        "USWRFsfc",
        "tendency_of_total_water_path_due_to_advection",
    ]
    normalization = get_network_and_loss_normalization_config(
        names=list(set(in_names).union(out_names)),
        dir=dir,
    )
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={
                        "scale_factor": 1,
                        "embed_dim": 4,
                        "num_layers": 2,
                    },
                ),
                in_names=in_names,
                out_names=out_names,
                normalization=normalization,
                corrector=AtmosphereCorrectorConfig(
                    conserve_dry_air=True,
                    zero_global_mean_moisture_advection=True,
                    moisture_budget_correction="advection_and_precipitation",
                    force_positive_names=["PRATEsfc"],
                    total_energy_budget_correction=EnergyBudgetConfig(
                        "constant_temperature"
                    ),
                ),
            ),
        ),
    )


def get_multi_call_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    return StepSelector(
        type="multi_call",
        config=dataclasses.asdict(
            MultiCallStepConfig(
                wrapped_step=StepSelector(
                    type="separate_radiation",
                    config=dataclasses.asdict(get_separate_radiation_config(dir)),
                ),
                config=MultiCallConfig(
                    forcing_name="forcing_rad",
                    forcing_multipliers={"double": 2.0},
                    output_names=["diagnostic_rad"],
                ),
            ),
        ),
    )


SEPARATE_RADIATION_CONFIG = get_separate_radiation_config()

SELECTOR_GETTERS = [
    get_single_module_with_atmosphere_corrector_selector,
    get_separate_radiation_selector,
    get_single_module_selector,
    get_single_module_noise_conditioned_selector,
    get_multi_call_selector,
]

SELECTOR_CONFIG_CASES = [
    pytest.param(
        getter(),
        id=getter.__name__,
    )
    for getter in SELECTOR_GETTERS
]

EXPORTABLE_SELECTOR_CONFIG_CASES = [
    # Some configs use th.DiscreteContinuousConvS2 layers
    # which currently do not support export, see
    # https://github.com/NVIDIA/torch-harmonics/issues/73
    pytest.param(
        getter(),
        id=getter.__name__,
    )
    for getter in [
        get_single_module_with_atmosphere_corrector_selector,
        get_separate_radiation_selector,
        get_single_module_selector,
        get_multi_call_selector,
    ]
]

HAS_NEXT_STEP_FORCING_NAME_CASES = [
    pytest.param(
        StepSelector(
            type="separate_radiation",
            config=dataclasses.asdict(SEPARATE_RADIATION_CONFIG),
        ),
        id="multi_call_separate_radiation",
    ),
]

HAS_NEXT_STEP_FORCING_NAME_CASES = [
    pytest.param(
        StepSelector(
            type="separate_radiation",
            config=dataclasses.asdict(SEPARATE_RADIATION_CONFIG),
        ),
        id="separate_radiation",
    ),
]
TIMESTEP = datetime.timedelta(hours=6)


def get_tensor_dict(
    names: list[str], img_shape: tuple[int, int], n_samples: int
) -> TensorDict:
    data_dict = {}
    device = fme.get_device()
    for name in names:
        data_dict[name] = torch.rand(
            n_samples,
            *img_shape,
            device=device,
        )
    return data_dict


def get_step(
    selector: StepSelector,
    img_shape: tuple[int, int],
    init_weights: Callable[[list[nn.Module]], None] = lambda _: None,
) -> StepABC:
    device = fme.get_device()
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(img_shape[0], device=device),
        lon=torch.zeros(img_shape[1], device=device),
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=TIMESTEP,
    )
    return selector.get_step(dataset_info, init_weights)


@pytest.mark.parametrize("config", HAS_NEXT_STEP_FORCING_NAME_CASES)
def test_next_step_forcing_names_is_forcing(config: StepSelector):
    data = dataclasses.asdict(config)
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(config, img_shape)
    forcing_names = set(step.input_names).difference(step.output_names)
    data["config"]["next_step_forcing_names"] = [list(forcing_names)[0]]
    dacite.from_dict(config.__class__, data, config=dacite.Config(strict=True))


@pytest.mark.parametrize("config", HAS_NEXT_STEP_FORCING_NAME_CASES)
def test_next_step_forcing_names_is_prognostic(config: StepSelector):
    data = dataclasses.asdict(config)
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(config, img_shape)
    prognostic_names = set(step.output_names).intersection(step.input_names)
    name = list(prognostic_names)[0]
    data["config"]["next_step_forcing_names"] = [name]
    with pytest.raises(ValueError) as err:
        dacite.from_dict(config.__class__, data, config=dacite.Config(strict=True))
    assert "next_step_forcing_name" in str(err.value)
    assert name in str(err.value)


@pytest.mark.parametrize("config", HAS_NEXT_STEP_FORCING_NAME_CASES)
def test_next_step_forcing_names_is_diagnostic(config: StepSelector):
    data = dataclasses.asdict(config)
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(config, img_shape)
    diagnostic_names = set(step.output_names).difference(step.input_names)
    name = list(diagnostic_names)[0]
    data["config"]["next_step_forcing_names"] = [name]
    with pytest.raises(ValueError) as err:
        dacite.from_dict(config.__class__, data, config=dacite.Config(strict=True))
    assert "next_step_forcing_name" in str(err.value)
    assert name in str(err.value)


@pytest.mark.parametrize("config", SELECTOR_CONFIG_CASES)
def test_step_applies_wrapper(config: StepSelector):
    torch.manual_seed(0)
    img_shape = DEFAULT_IMG_SHAPE
    n_samples = 5
    step = get_step(config, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    multi_calls = 1
    if isinstance(config._step_config_instance, MultiCallStepConfig):
        if config._step_config_instance.config is not None:
            multi_calls += len(config._step_config_instance.config.forcing_multipliers)

    wrapper = unittest.mock.MagicMock(side_effect=lambda x: x)
    step.step(input_data, next_step_input_data, wrapper=wrapper)
    assert wrapper.call_count == multi_calls * len(step.modules)
    for module in step.modules:
        wrapper.assert_any_call(module)


@pytest.mark.parametrize("config", SELECTOR_CONFIG_CASES)
def test_step_initializes_weights(config: StepSelector):
    torch.manual_seed(0)
    img_shape = DEFAULT_IMG_SHAPE
    init_weights = unittest.mock.MagicMock(side_effect=lambda x: x)
    step = get_step(config, img_shape, init_weights)
    assert init_weights.called
    call_args, call_kwargs = init_weights.call_args
    assert len(call_args) == 1
    assert len(call_kwargs) == 0
    assert isinstance(call_args[0], list | nn.ModuleList)
    assert len(call_args[0]) == len(step.modules)
    for i, module in enumerate(step.modules):
        assert isinstance(module, DummyWrapper)
        assert call_args[0][i] is module.module


@pytest.mark.parametrize("config", EXPORTABLE_SELECTOR_CONFIG_CASES)
def test_export_step(config: StepSelector, very_fast_only: bool):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    torch.manual_seed(0)
    img_shape = DEFAULT_IMG_SHAPE
    n_samples = 5
    step = get_step(config, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    unscripted_output = step.step(input_data, next_step_input_data)
    for name in step.output_names:
        # informative check for nan values
        torch.testing.assert_close(unscripted_output[name], unscripted_output[name])
    exported = step.export(input_data, next_step_input_data)
    output = exported.module()(input_data, next_step_input_data)
    for name in step.output_names:
        torch.testing.assert_close(output[name], unscripted_output[name])


@pytest.mark.parametrize(
    "get_config",
    SELECTOR_GETTERS,
)
def test_load_config(
    get_config: Callable[[pathlib.Path | None], StepSelector],
):
    non_path_config: StepSelector = get_config(
        None
    )  # doesn't depend on files, use to get names
    all_names = set(non_path_config.input_names).union(non_path_config.output_names)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        get_scalar_dataset(all_names, fill_value=0.1).to_netcdf(temp_path / "means.nc")
        get_scalar_dataset(all_names, fill_value=1.1).to_netcdf(temp_path / "stds.nc")
        config = get_config(temp_path)
        config.load()
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(config, img_shape)
    normalizer = step.normalizer
    assert normalizer.means.keys() == all_names
    assert normalizer.stds.keys() == all_names
    assert all(normalizer.means[name] == 0.1 for name in all_names)
    assert all(normalizer.stds[name] == 1.1 for name in all_names)


@pytest.mark.parametrize(
    "get_config",
    SELECTOR_GETTERS,
)
def test_load_is_required_for_path_config(
    get_config: Callable[[pathlib.Path | None], StepSelector],
):
    non_path_config: StepSelector = get_config(
        None
    )  # doesn't depend on files, use to get names
    all_names = set(non_path_config.input_names).union(non_path_config.output_names)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        get_scalar_dataset(all_names, fill_value=0.1).to_netcdf(temp_path / "means.nc")
        get_scalar_dataset(all_names, fill_value=1.1).to_netcdf(temp_path / "stds.nc")
        config = get_config(temp_path)
    img_shape = DEFAULT_IMG_SHAPE
    with pytest.raises(FileNotFoundError):
        get_step(config, img_shape)
