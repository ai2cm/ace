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
from torch import nn

import fme
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.ace.step.fcn3 import FCN3Config, FCN3Selector, FCN3StepConfig
from fme.ace.testing.fv3gfs_data import get_scalar_dataset
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig, EnergyBudgetConfig
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed.distributed import Distributed
from fme.core.distributed.non_distributed import DummyWrapper
from fme.core.labels import BatchLabels
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.global_mean_removal import (
    PerChannelGlobalMeanRemovalConfig,
    SharedGlobalMeanRemovalConfig,
)
from fme.core.step.multi_call import MultiCallConfig, MultiCallStepConfig
from fme.core.step.secondary_decoder import SecondaryDecoderConfig
from fme.core.step.secondary_module import SecondaryModuleStepConfig
from fme.core.step.single_module import (
    SingleModuleStepConfig,
    _apply_extra_channel_mask,
    _apply_input_mask,
    _build_channel_mask_dict,
    _build_effective_input_mask,
)
from fme.core.step.step import StepABC, StepSelector
from fme.core.typing_ import TensorDict
from fme.core.var_masking import UniformMaskingConfig, VariableMaskingConfig

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
                            filter_type="linear",
                            filter_num_groups=2,
                            context_pos_embed_dim=2,
                            pos_embed=False,
                            num_layers=2,
                            local_blocks=[0],
                            affine_norms=True,
                        )
                    ),
                ),
                in_names=["forcing_shared", "forcing_rad"],
                out_names=["diagnostic_main"],
                secondary_decoder=SecondaryDecoderConfig(
                    secondary_diagnostic_names=["diagnostic_rad"],
                    network=ModuleSelector(type="MLP", config={}),
                ),
                normalization=normalization,
            ),
        ),
    )


def get_label_conditioned_selector(
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
                    conditional=True,
                    config=dataclasses.asdict(
                        NoiseConditionedSFNOBuilder(
                            embed_dim=4,
                            noise_embed_dim=4,
                            noise_type="isotropic",
                            num_layers=2,
                            local_blocks=[0],
                            label_embed_dim=3,
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


def get_fcn3_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    forcing_names = [
        "DSWRFtoa",
        "HGTsfc",
    ]
    atmosphere_prognostic_names = [
        "air_temperature",
        "specific_total_water",
    ]
    atmosphere_diagnostic_names = [
        "radiative_heating",
    ]
    atmosphere_levels = 6
    surface_prognostic_names = [
        "PRESsfc",
    ]
    surface_diagnostic_names = [
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
    atmosphere_packed_names = []
    for i in range(atmosphere_levels):
        atmosphere_packed_names.append(f"air_temperature_{i}")
        atmosphere_packed_names.append(f"specific_total_water_{i}")
        atmosphere_packed_names.append(f"radiative_heating_{i}")
    normalization = get_network_and_loss_normalization_config(
        names=list(
            set(forcing_names)
            .union(atmosphere_packed_names)
            .union(surface_prognostic_names)
            .union(surface_diagnostic_names)
        ),
        dir=dir,
    )
    step_config = FCN3StepConfig(
        builder=FCN3Selector(
            type="FCN3",
            config=FCN3Config(
                scale_factor=1,
                atmo_embed_dim=2,
                surf_embed_dim=2,
                aux_embed_dim=2,
                num_layers=2,
            ),
        ),
        forcing_names=forcing_names,
        atmosphere_prognostic_names=atmosphere_prognostic_names,
        atmosphere_diagnostic_names=atmosphere_diagnostic_names,
        atmosphere_levels=atmosphere_levels,
        surface_prognostic_names=surface_prognostic_names,
        surface_diagnostic_names=surface_diagnostic_names,
        normalization=normalization,
        corrector=AtmosphereCorrectorConfig(
            conserve_dry_air=True,
            zero_global_mean_moisture_advection=True,
            moisture_budget_correction="advection_and_precipitation",
            force_positive_names=["PRATEsfc"],
            total_energy_budget_correction=EnergyBudgetConfig("constant_temperature"),
        ),
    )
    return StepSelector(
        type="FCN3",
        config=dataclasses.asdict(step_config),
    )


def get_secondary_module_selector(
    dir: pathlib.Path | None = None,
) -> StepSelector:
    normalization = get_network_and_loss_normalization_config(
        names=[
            "forcing_a",
            "prog_a",
            "prog_b",
            "diag_a",
        ],
        dir=dir,
    )
    return StepSelector(
        type="secondary_module",
        config=dataclasses.asdict(
            SecondaryModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={
                        "scale_factor": 1,
                        "embed_dim": 4,
                        "num_layers": 2,
                    },
                ),
                in_names=["forcing_a", "prog_a", "prog_b"],
                out_names=["prog_a", "prog_b"],
                normalization=normalization,
                secondary_builder=ModuleSelector(type="MLP", config={}),
                secondary_out_names=["diag_a"],
                secondary_residual_out_names=["prog_a"],
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
    get_fcn3_selector,
    get_single_module_with_atmosphere_corrector_selector,
    get_separate_radiation_selector,
    get_single_module_selector,
    get_single_module_noise_conditioned_selector,
    get_secondary_module_selector,
    get_multi_call_selector,
]

SELECTOR_CONFIG_CASES = [
    pytest.param(
        getter(),
        id=getter.__name__,
    )
    for getter in SELECTOR_GETTERS
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
    all_labels: set[str] | None = None,
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
        all_labels=all_labels,
    )
    return selector.get_step(dataset_info, init_weights)


@pytest.mark.parallel
def test_label_conditioned_step():
    dist = Distributed.get_instance()
    selector = get_label_conditioned_selector()
    step = get_step(selector, DEFAULT_IMG_SHAPE, all_labels={"a", "b"})
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples=1)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples=1
    )
    input_data = dist.scatter_spatial(input_data, DEFAULT_IMG_SHAPE)
    next_step_input_data = dist.scatter_spatial(next_step_input_data, DEFAULT_IMG_SHAPE)
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=BatchLabels.new_from_set(
                {"a", "b"}, n_samples=1, device=fme.get_device()
            ),
        ),
        wrapper=lambda x: x,
    )
    h_sl, w_sl = dist.get_local_slices(DEFAULT_IMG_SHAPE)
    local_h = DEFAULT_IMG_SHAPE[0] if h_sl == slice(None) else h_sl.stop - h_sl.start
    local_w = DEFAULT_IMG_SHAPE[1] if w_sl == slice(None) else w_sl.stop - w_sl.start
    assert output["diagnostic_main"].shape == (1, local_h, local_w)
    assert output["diagnostic_rad"].shape == (1, local_h, local_w)


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
    step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
        ),
        wrapper=wrapper,
    )
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


@pytest.mark.parametrize(
    ["conflict"],
    [
        pytest.param(
            "output",
            id="conflict_with_output",
        ),
        pytest.param(
            "input",
            id="conflict_with_input",
        ),
    ],
)
@pytest.mark.parallel
def test_input_output_names_secondary_decoder_conflict(conflict: str):
    input_names = ["input"]
    output_names = ["output"]
    secondary_decoder_names = [conflict]
    normalization = get_network_and_loss_normalization_config(
        names=input_names + output_names + secondary_decoder_names,
        dir=None,
    )
    with pytest.raises(ValueError) as err:
        SingleModuleStepConfig(
            normalization=normalization,
            in_names=input_names,
            out_names=output_names,
            builder=ModuleSelector(type="MLP", config={}),
            secondary_decoder=SecondaryDecoderConfig(
                secondary_diagnostic_names=secondary_decoder_names,
                network=ModuleSelector(type="MLP", config={}),
            ),
        )
    assert f"secondary_diagnostic_name is an {conflict} variable:" in str(err.value)


def test_step_with_prescribed_prognostic_overwrites_output():
    normalization = get_network_and_loss_normalization_config(
        names=["forcing_shared", "forcing_rad", "diagnostic_main", "diagnostic_rad"],
    )
    config = StepSelector(
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
                prescribed_prognostic_names=["diagnostic_main"],
            ),
        ),
    )
    img_shape = DEFAULT_IMG_SHAPE
    n_samples = 2
    step = get_step(config, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    prescribed_value = torch.full(
        (n_samples,) + img_shape, 42.0, device=fme.get_device()
    )
    next_step_input_data["diagnostic_main"] = prescribed_value
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
        ),
        wrapper=lambda x: x,
    )
    torch.testing.assert_close(output["diagnostic_main"], prescribed_value)


def test_secondary_module_empty_names_raises():
    normalization = get_network_and_loss_normalization_config(
        names=["a", "b"],
    )
    with pytest.raises(ValueError, match="at least one of"):
        SecondaryModuleStepConfig(
            builder=ModuleSelector(type="MLP", config={}),
            in_names=["a"],
            out_names=["b"],
            normalization=normalization,
            secondary_builder=ModuleSelector(type="MLP", config={}),
        )


def test_secondary_module_out_name_overlaps_out_names_raises():
    normalization = get_network_and_loss_normalization_config(
        names=["a", "b"],
    )
    with pytest.raises(ValueError, match="secondary_out_names must not overlap"):
        SecondaryModuleStepConfig(
            builder=ModuleSelector(type="MLP", config={}),
            in_names=["a"],
            out_names=["b"],
            normalization=normalization,
            secondary_builder=ModuleSelector(type="MLP", config={}),
            secondary_out_names=["b"],
        )


def test_secondary_module_out_name_overlaps_residual_out_names_raises():
    normalization = get_network_and_loss_normalization_config(
        names=["a", "b", "c"],
    )
    with pytest.raises(
        ValueError, match="secondary_out_names must not overlap.*residual"
    ):
        SecondaryModuleStepConfig(
            builder=ModuleSelector(type="MLP", config={}),
            in_names=["a"],
            out_names=["b"],
            normalization=normalization,
            secondary_builder=ModuleSelector(type="MLP", config={}),
            secondary_out_names=["c"],
            secondary_residual_out_names=["c"],
        )


def test_secondary_module_residual_out_name_not_in_out_or_in_names_raises():
    normalization = get_network_and_loss_normalization_config(
        names=["a", "b", "c"],
    )
    with pytest.raises(ValueError, match="secondary_residual_out_name 'c'"):
        SecondaryModuleStepConfig(
            builder=ModuleSelector(type="MLP", config={}),
            in_names=["a"],
            out_names=["b"],
            normalization=normalization,
            secondary_builder=ModuleSelector(type="MLP", config={}),
            secondary_residual_out_names=["c"],
        )


def test_secondary_module_output_names_full_field_only():
    """secondary_out_names appear in output_names."""
    normalization = get_network_and_loss_normalization_config(
        names=["a", "b", "c"],
    )
    config = SecondaryModuleStepConfig(
        builder=ModuleSelector(type="MLP", config={}),
        in_names=["a"],
        out_names=["b"],
        normalization=normalization,
        secondary_builder=ModuleSelector(type="MLP", config={}),
        secondary_out_names=["c"],
    )
    assert "c" in config.output_names
    assert "b" in config.output_names


def test_secondary_module_output_names_residual_only():
    """secondary_residual_out_names that are in out_names appear in output_names."""
    normalization = get_network_and_loss_normalization_config(
        names=["a", "b"],
    )
    config = SecondaryModuleStepConfig(
        builder=ModuleSelector(type="MLP", config={}),
        in_names=["a"],
        out_names=["b"],
        normalization=normalization,
        secondary_builder=ModuleSelector(type="MLP", config={}),
        secondary_residual_out_names=["b"],
    )
    assert "b" in config.output_names


def test_secondary_module_output_names_residual_on_input_only():
    """secondary_residual_out_names on input-only name adds to output_names."""
    normalization = get_network_and_loss_normalization_config(
        names=["a", "b"],
    )
    config = SecondaryModuleStepConfig(
        builder=ModuleSelector(type="MLP", config={}),
        in_names=["a", "b"],
        out_names=["b"],
        normalization=normalization,
        secondary_builder=ModuleSelector(type="MLP", config={}),
        secondary_residual_out_names=["a"],
    )
    assert "a" in config.output_names
    assert "b" in config.output_names


@pytest.mark.parallel
def test_secondary_module_full_field_and_residual():
    """Test secondary_out_names and secondary_residual_out_names together."""
    torch.manual_seed(0)
    normalization = get_network_and_loss_normalization_config(
        names=["forcing", "prog", "diag"],
    )
    config = StepSelector(
        type="secondary_module",
        config=dataclasses.asdict(
            SecondaryModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={
                        "scale_factor": 1,
                        "embed_dim": 4,
                        "num_layers": 2,
                    },
                ),
                in_names=["forcing", "prog"],
                out_names=["prog"],
                normalization=normalization,
                secondary_builder=ModuleSelector(type="MLP", config={}),
                secondary_out_names=["diag"],
                secondary_residual_out_names=["prog"],
            ),
        ),
    )
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(config, img_shape)
    assert "prog" in step.output_names
    assert "diag" in step.output_names
    assert "prog" in step.prognostic_names
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples=2)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples=2
    )
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
        ),
    )
    assert "prog" in output
    assert "diag" in output
    assert output["prog"].shape == (2, *img_shape)
    assert output["diag"].shape == (2, *img_shape)


@pytest.mark.parallel
def test_secondary_module_state_round_trip():
    """Test get_state/load_state with secondary module."""
    torch.manual_seed(0)
    normalization = get_network_and_loss_normalization_config(
        names=["forcing", "prog", "diag"],
    )
    config = SecondaryModuleStepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={
                "scale_factor": 1,
                "embed_dim": 4,
                "num_layers": 2,
            },
        ),
        in_names=["forcing", "prog"],
        out_names=["prog"],
        normalization=normalization,
        secondary_builder=ModuleSelector(type="MLP", config={}),
        secondary_out_names=["diag"],
        secondary_residual_out_names=["prog"],
    )
    img_shape = DEFAULT_IMG_SHAPE
    step1 = get_step(
        StepSelector(type="secondary_module", config=dataclasses.asdict(config)),
        img_shape,
    )
    state = step1.get_state()
    assert "secondary_module" in state

    step2 = get_step(
        StepSelector(type="secondary_module", config=dataclasses.asdict(config)),
        img_shape,
    )
    step2.load_state(state)

    input_data = get_tensor_dict(step1.input_names, img_shape, n_samples=1)
    next_step_input_data = get_tensor_dict(
        step1.next_step_input_names, img_shape, n_samples=1
    )
    args = StepArgs(
        input=input_data,
        next_step_input_data=next_step_input_data,
        n_ensemble=1,
        labels=None,
    )
    out1, _ = step1.step(args=args)
    out2, _ = step2.step(args=args)
    for name in out1:
        torch.testing.assert_close(out1[name], out2[name])


@pytest.mark.parallel
def test_secondary_module_residual_on_input_only_with_residual_prediction():
    """When residual_prediction=True and secondary_residual_out_name is in in_names
    but not out_names, the input should not be added twice."""
    torch.manual_seed(0)
    normalization = get_network_and_loss_normalization_config(
        names=["forcing", "prog_a", "prog_b"],
    )
    config = StepSelector(
        type="secondary_module",
        config=dataclasses.asdict(
            SecondaryModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={
                        "scale_factor": 1,
                        "embed_dim": 4,
                        "num_layers": 2,
                    },
                ),
                in_names=["forcing", "prog_a", "prog_b"],
                out_names=["prog_a", "prog_b"],
                normalization=normalization,
                residual_prediction=True,
                secondary_builder=ModuleSelector(type="MLP", config={}),
                secondary_residual_out_names=["prog_a"],
            ),
        ),
    )
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(config, img_shape)
    assert "prog_a" in step.output_names
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples=2)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples=2
    )
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
        ),
    )
    assert output["prog_a"].shape == (2, *img_shape)
    assert output["prog_b"].shape == (2, *img_shape)


def test_apply_input_mask_zeros_masked_variables():
    input_norm = {
        "a": torch.ones(4, 8, 16),
        "b": torch.ones(4, 8, 16) * 2.0,
    }
    mask = {
        "a": torch.tensor([True, True, False, False]),
        "b": torch.tensor([True, False, True, False]),
    }
    result = _apply_input_mask(input_norm, mask)
    expected_a = torch.tensor([1.0, 1.0, 0.0, 0.0]).view(4, 1, 1).expand(4, 8, 16)
    expected_b = torch.tensor([2.0, 0.0, 2.0, 0.0]).view(4, 1, 1).expand(4, 8, 16)
    torch.testing.assert_close(result["a"], expected_a)
    torch.testing.assert_close(result["b"], expected_b)


def test_apply_input_mask_ignores_unknown_names():
    input_norm = {"a": torch.ones(2, 4, 8)}
    mask = {"not_a_variable": torch.tensor([False, False])}
    result = _apply_input_mask(input_norm, mask)
    torch.testing.assert_close(result["a"], input_norm["a"])


def test_apply_input_mask_does_not_mutate_original():
    original = torch.ones(2, 4, 8)
    input_norm = {"a": original}
    mask = {"a": torch.tensor([False, False])}
    result = _apply_input_mask(input_norm, mask)
    torch.testing.assert_close(original, torch.ones(2, 4, 8))
    assert result["a"] is not original


def test_step_with_data_mask():
    normalization = get_network_and_loss_normalization_config(
        names=["forcing_shared", "forcing_rad", "diagnostic_main", "diagnostic_rad"],
    )
    config = StepSelector(
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
    img_shape = DEFAULT_IMG_SHAPE
    n_samples = 4
    step = get_step(config, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    data_mask = {
        "forcing_shared": torch.tensor(
            [True, True, False, False], device=fme.get_device()
        ),
    }
    output_no_mask, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
            data_mask=None,
        ),
    )
    output_with_mask, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
            data_mask=data_mask,
        ),
    )
    for name in ["diagnostic_main", "diagnostic_rad"]:
        assert output_with_mask[name].shape == (n_samples, *img_shape)
        torch.testing.assert_close(output_with_mask[name][:2], output_no_mask[name][:2])
        assert not torch.allclose(output_with_mask[name][2:], output_no_mask[name][2:])


def test_build_channel_mask_dict_with_data_mask():
    in_names = ["a", "b"]
    packed = torch.zeros(3, 2, 8, 16)
    data_mask = {
        "a": torch.tensor([True, False, True]),
        "b": torch.tensor([False, True, True]),
    }
    result = _build_channel_mask_dict(in_names, data_mask, packed)
    assert set(result) == {"a", "b"}
    assert result["a"].shape == (3, 8, 16)
    assert result["a"][0, 0, 0] == 1.0
    assert result["a"][1, 0, 0] == 0.0
    assert result["a"][2, 0, 0] == 1.0
    assert result["b"][0, 0, 0] == 0.0
    assert result["b"][1, 0, 0] == 1.0
    assert (result["b"][2] == 1.0).all()


def test_build_channel_mask_dict_no_data_mask():
    packed = torch.zeros(2, 3, 4, 8)
    result = _build_channel_mask_dict(["x", "y", "z"], None, packed)
    assert set(result) == {"x", "y", "z"}
    for name in result:
        assert result[name].shape == (2, 4, 8)
        assert (result[name] == 1.0).all()


def test_build_channel_mask_dict_partial_mask():
    packed = torch.zeros(2, 2, 4, 8)
    data_mask = {"a": torch.tensor([True, False])}
    result = _build_channel_mask_dict(["a", "b"], data_mask, packed)
    assert set(result) == {"a", "b"}
    assert result["a"][0, 0, 0] == 1.0
    assert result["a"][1, 0, 0] == 0.0
    assert (result["b"] == 1.0).all()


def test_step_with_include_channel_mask_inputs():
    normalization = get_network_and_loss_normalization_config(
        names=["forcing_shared", "forcing_rad", "diagnostic_main", "diagnostic_rad"],
    )
    config = StepSelector(
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
                include_channel_mask_inputs=True,
            ),
        ),
    )
    img_shape = DEFAULT_IMG_SHAPE
    n_samples = 4
    step = get_step(config, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    data_mask = {
        "forcing_shared": torch.tensor(
            [True, True, False, False], device=fme.get_device()
        ),
    }
    output_no_mask, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
            data_mask=None,
        ),
    )
    output_with_mask, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
            data_mask=data_mask,
        ),
    )
    for name in ["diagnostic_main", "diagnostic_rad"]:
        assert output_with_mask[name].shape == (n_samples, *img_shape)
        torch.testing.assert_close(output_with_mask[name][:2], output_no_mask[name][:2])
        assert not torch.allclose(output_with_mask[name][2:], output_no_mask[name][2:])


def test_step_with_include_channel_mask_inputs_no_data_mask():
    normalization = get_network_and_loss_normalization_config(
        names=["forcing_shared", "forcing_rad", "diagnostic_main", "diagnostic_rad"],
    )
    config = StepSelector(
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
                include_channel_mask_inputs=True,
            ),
        ),
    )
    img_shape = DEFAULT_IMG_SHAPE
    n_samples = 2
    step = get_step(config, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    output_no_mask, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
            data_mask=None,
        ),
    )
    all_unmasked = {
        name: torch.ones(n_samples, dtype=torch.bool, device=fme.get_device())
        for name in step.input_names
    }
    output_all_unmasked, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            labels=None,
            data_mask=all_unmasked,
        ),
    )
    for name in ["diagnostic_main", "diagnostic_rad"]:
        assert output_no_mask[name].shape == (n_samples, *img_shape)
        torch.testing.assert_close(output_no_mask[name], output_all_unmasked[name])


def test_step_masked_nan_input_is_zeroed_before_network_with_corrector():
    """Masked NaNs should be replaced in normalized space before module input."""
    in_names = ["masked_input", "forcing"]
    out_names = ["masked_input"]
    normalization = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={name: 0.0 for name in set(in_names + out_names)},
            stds={name: 1.0 for name in set(in_names + out_names)},
        ),
    )

    class AssertFiniteInputModule(nn.Module):
        def __init__(self):
            super().__init__()
            self._p = nn.Parameter(torch.zeros(1))

        def forward(self, inp):
            assert not torch.isnan(inp).any()
            return self._p.new_full(
                (inp.shape[0], len(out_names), *inp.shape[-2:]), -1.0
            )

    config = StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="prebuilt", config={"module": AssertFiniteInputModule()}
                ),
                in_names=in_names,
                out_names=out_names,
                normalization=normalization,
                corrector=AtmosphereCorrectorConfig(force_positive_names=out_names),
            )
        ),
    )
    img_shape = (4, 8)
    n_samples = 4
    step = get_step(config, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples)
    input_data["masked_input"][2:] = torch.nan
    next_step_input_data = get_tensor_dict(
        step.next_step_input_names, img_shape, n_samples
    )
    data_mask = {
        "masked_input": torch.tensor(
            [True, True, False, False], device=fme.get_device()
        ),
    }

    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
            data_mask=data_mask,
        ),
    )
    assert not output["masked_input"].isnan().any()
    assert (output["masked_input"] == 0.0).all()


def test_input_dropout_applied_in_train_mode_not_eval():
    """input_dropout must zero channels when the module is training and skip
    when it is in eval mode.

    Two calls are made with the same non-zero input: one in train mode (module
    in training state) and one in eval mode.  With rate=1.0 the dropout always
    fires during training, so the indicator channel for the dropped variable
    must be 0.0 in train output and 1.0 in eval output.
    """
    in_names = ["x", "y"]
    out_names = ["x"]
    img_shape = (4, 8)
    n_samples = 2

    normalization = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={n: 0.0 for n in in_names + out_names},
            stds={n: 1.0 for n in in_names + out_names},
        )
    )

    # Module records the indicator channels (second half of input) for inspection.
    class RecordMaskModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_x_indicator: torch.Tensor | None = None
            self._p = nn.Parameter(torch.zeros(1))

        def forward(self, inp):
            n_in = inp.shape[1] // 2
            # indicator channels are the second half
            indicators = inp[:, n_in:]
            # "x" is first in in_names
            self.last_x_indicator = indicators[:, 0].detach().clone()
            return self._p.new_zeros(n_samples, len(out_names), *img_shape)

    recorder = RecordMaskModule()
    from fme.core.step.single_module import SingleModuleStep

    step_config = SingleModuleStepConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": recorder}),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
        include_channel_mask_inputs=True,
        input_dropout=VariableMaskingConfig(per_variable={"x": 1.0}),
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(img_shape[0]),
            lon=torch.zeros(img_shape[1]),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7), bk=torch.arange(7)
        ),
        timestep=TIMESTEP,
    )
    step_obj = step_config.get_step(dataset_info, init_weights=lambda _: None)
    assert isinstance(step_obj, SingleModuleStep)
    input_data = get_tensor_dict(in_names, img_shape, n_samples)
    next_step_input_data = get_tensor_dict(
        step_obj.next_step_input_names, img_shape, n_samples
    )

    # Training mode: dropout fires, indicator for "x" should be 0.0
    step_obj.module.torch_module.train()
    step_obj.step(
        StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
        )
    )
    assert recorder.last_x_indicator is not None
    assert (recorder.last_x_indicator == 0.0).all(), "Expected x masked in train mode"

    # Eval mode: dropout disabled, indicator for "x" should be 1.0
    step_obj.module.torch_module.eval()
    step_obj.step(
        StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            n_ensemble=1,
        )
    )
    assert (recorder.last_x_indicator == 1.0).all(), "Expected x present in eval mode"


def _make_input_dropout_step_config(
    input_dropout: VariableMaskingConfig | None = None,
    in_names: list[str] | None = None,
    out_names: list[str] | None = None,
    global_mean_removal: (
        PerChannelGlobalMeanRemovalConfig | SharedGlobalMeanRemovalConfig | None
    ) = None,
) -> SingleModuleStepConfig:
    if in_names is None:
        in_names = ["x", "y"]
    if out_names is None:
        out_names = ["x"]
    all_names = set(in_names + out_names)
    normalization = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={name: 0.0 for name in all_names},
            stds={name: 1.0 for name in all_names},
        )
    )
    return SingleModuleStepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
        ),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
        global_mean_removal=global_mean_removal,
        input_dropout=input_dropout,
    )


def test_input_dropout_config_round_trips_through_dacite():
    variable_config = _make_input_dropout_step_config(
        VariableMaskingConfig(per_variable={"x": 0.25})
    )
    variable_restored = dacite.from_dict(
        SingleModuleStepConfig,
        dataclasses.asdict(variable_config),
        config=dacite.Config(strict=True),
    )
    assert isinstance(variable_restored.input_dropout, VariableMaskingConfig)
    assert variable_restored.input_dropout.per_variable == {"x": 0.25}

    uniform_config = _make_input_dropout_step_config(
        VariableMaskingConfig(uniform=UniformMaskingConfig(min_vars=1, max_vars=1))
    )
    uniform_restored = dacite.from_dict(
        SingleModuleStepConfig,
        dataclasses.asdict(uniform_config),
        config=dacite.Config(strict=True),
    )
    assert isinstance(uniform_restored.input_dropout, VariableMaskingConfig)
    assert isinstance(uniform_restored.input_dropout.uniform, UniformMaskingConfig)
    assert uniform_restored.input_dropout.uniform.min_vars == 1
    assert uniform_restored.input_dropout.uniform.max_vars == 1


def test_build_effective_input_mask_merges_data_mask_and_dropout():
    data_mask = {
        "a": torch.tensor([True, False, True]),
        "b": torch.tensor([True, True, False]),
    }
    input_data = {"a": torch.ones(3, 1, 1)}
    result = _build_effective_input_mask(
        data_mask=data_mask,
        input_dropout=VariableMaskingConfig(per_variable={"a": 1.0}),
        training=True,
        in_names=["a"],
        extra_channel_names=[],
        input_data=input_data,
        n_ensemble=1,
    )

    torch.testing.assert_close(result["a"], torch.tensor([False, False, False]))
    torch.testing.assert_close(result["b"], data_mask["b"])


def test_build_effective_input_mask_repeats_dropout_for_ensemble_members():
    torch.manual_seed(0)
    n_samples = 4
    n_ensemble = 3
    data_mask = {"b": torch.ones(n_samples * n_ensemble, dtype=torch.bool)}
    input_data = {"a": torch.ones(n_samples * n_ensemble, 1, 1)}
    result = _build_effective_input_mask(
        data_mask=data_mask,
        input_dropout=VariableMaskingConfig(per_variable={"a": 0.5}),
        training=True,
        in_names=["a"],
        extra_channel_names=[],
        input_data=input_data,
        n_ensemble=n_ensemble,
    )

    grouped = result["a"].view(n_samples, n_ensemble)
    assert (grouped == grouped[:, :1]).all()
    assert grouped[:, 0].any()
    assert (~grouped[:, 0]).any()
    torch.testing.assert_close(result["b"], data_mask["b"])


def test_input_dropout_uniform_bounds_validated_against_in_names():
    with pytest.raises(ValueError, match="min_vars"):
        _make_input_dropout_step_config(
            in_names=["x", "y"],
            out_names=["x"],
            input_dropout=VariableMaskingConfig(
                uniform=UniformMaskingConfig(
                    min_vars=2, max_vars="max", ignore_vars=["y"]
                )
            ),
        )


@pytest.mark.parametrize(
    "input_dropout",
    [
        pytest.param(
            VariableMaskingConfig(per_variable={"surface_temperature": 1.0}),
            id="variable",
        ),
        pytest.param(
            VariableMaskingConfig(uniform=UniformMaskingConfig(min_vars=1, max_vars=1)),
            id="uniform",
        ),
    ],
)
def test_input_dropout_can_mask_shared_global_mean_reference(input_dropout):
    config = _make_input_dropout_step_config(
        in_names=["surface_temperature", "x"],
        out_names=["surface_temperature"],
        global_mean_removal=SharedGlobalMeanRemovalConfig(
            reference_field="surface_temperature",
            field_names=["surface_temperature", "x"],
        ),
        input_dropout=input_dropout,
    )
    assert isinstance(config.input_dropout, VariableMaskingConfig)


def test_input_dropout_allows_shared_global_mean_reference_when_ignored():
    config = _make_input_dropout_step_config(
        in_names=["surface_temperature", "x"],
        out_names=["surface_temperature"],
        global_mean_removal=SharedGlobalMeanRemovalConfig(
            reference_field="surface_temperature",
            field_names=["surface_temperature", "x"],
        ),
        input_dropout=VariableMaskingConfig(
            uniform=UniformMaskingConfig(
                min_vars=1, max_vars=1, ignore_vars=["surface_temperature"]
            )
        ),
    )
    assert isinstance(config.input_dropout, VariableMaskingConfig)


def _make_global_mean_removal_step(
    global_mean_removal, in_names=None, out_names=None, means=None, stds=None
):
    if in_names is None:
        in_names = ["forcing_shared", "forcing_rad"]
    if out_names is None:
        out_names = ["diagnostic_main", "diagnostic_rad"]
    all_names = list(set(in_names + out_names))
    if means is None:
        means = {name: 0.0 for name in all_names}
    if stds is None:
        stds = {name: 1.0 for name in all_names}
    normalization = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(means=means, stds=stds),
    )
    config = StepSelector(
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
                global_mean_removal=global_mean_removal,
            ),
        ),
    )
    return get_step(config, DEFAULT_IMG_SHAPE)


def test_step_shared_global_mean_removal():
    in_names = ["surface_temperature", "air_temperature_0"]
    out_names = ["surface_temperature", "air_temperature_0"]
    means = {n: 280.0 for n in in_names}
    stds = {n: 5.0 for n in in_names}
    removal = SharedGlobalMeanRemovalConfig(
        reference_field="surface_temperature",
        field_names=in_names,
    )
    step = _make_global_mean_removal_step(
        removal, in_names, out_names, means=means, stds=stds
    )
    n_samples = 2
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step,
            n_ensemble=1,
            labels=None,
        ),
    )
    for name in out_names:
        assert output[name].shape == (n_samples, *DEFAULT_IMG_SHAPE)


def test_step_shared_global_mean_removal_with_extra_channels():
    in_names = ["surface_temperature", "air_temperature_0"]
    out_names = ["surface_temperature", "air_temperature_0"]
    means = {n: 280.0 for n in in_names}
    stds = {n: 5.0 for n in in_names}
    removal = SharedGlobalMeanRemovalConfig(
        reference_field="surface_temperature",
        field_names=in_names,
        append_as_input=True,
    )
    step = _make_global_mean_removal_step(
        removal, in_names, out_names, means=means, stds=stds
    )
    n_samples = 2
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step,
            n_ensemble=1,
            labels=None,
        ),
    )
    for name in out_names:
        assert output[name].shape == (n_samples, *DEFAULT_IMG_SHAPE)


def test_step_per_channel_global_mean_removal():
    removal = PerChannelGlobalMeanRemovalConfig(field_names=None)
    step = _make_global_mean_removal_step(removal)
    n_samples = 2
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step,
            n_ensemble=1,
            labels=None,
        ),
    )
    for name in step.output_names:
        assert output[name].shape == (n_samples, *DEFAULT_IMG_SHAPE)


def test_step_per_channel_global_mean_removal_with_extra_channels():
    in_names = ["forcing_shared", "forcing_rad"]
    out_names = ["diagnostic_main", "diagnostic_rad"]
    removal = PerChannelGlobalMeanRemovalConfig(
        field_names=in_names,
        append_as_input=True,
    )
    step = _make_global_mean_removal_step(removal, in_names, out_names)
    n_samples = 2
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step,
            n_ensemble=1,
            labels=None,
        ),
    )
    for name in step.output_names:
        assert output[name].shape == (n_samples, *DEFAULT_IMG_SHAPE)


def _assert_global_mean_removal_affects_output(removal, in_names, out_names, means):
    stds = {n: 1.0 for n in set(in_names + out_names)}
    n_samples = 2
    torch.manual_seed(0)
    step_baseline = _make_global_mean_removal_step(
        None, in_names, out_names, means=means, stds=stds
    )
    torch.manual_seed(0)
    step_with_removal = _make_global_mean_removal_step(
        removal, in_names, out_names, means=means, stds=stds
    )
    input_data = get_tensor_dict(
        step_baseline.input_names, DEFAULT_IMG_SHAPE, n_samples
    )
    # Shift inputs so they have a non-zero spatial mean per channel; otherwise
    # forward_transform would be a no-op and the outputs would coincide.
    input_data = {k: v + 5.0 for k, v in input_data.items()}
    next_step = get_tensor_dict(
        step_baseline.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )
    baseline_output, _ = step_baseline.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step,
            n_ensemble=1,
            labels=None,
        ),
    )
    removal_output, _ = step_with_removal.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step,
            n_ensemble=1,
            labels=None,
        ),
    )
    differs = any(
        not torch.allclose(baseline_output[name], removal_output[name])
        for name in out_names
    )
    assert differs, "global_mean_removal had no effect on step outputs"


def test_step_per_channel_global_mean_removal_affects_output():
    """Verify PerChannelGlobalMeanRemoval is actually invoked during step.

    The per-field unit tests in ``test_global_mean_removal.py`` cover the
    forward/inverse value behavior. This test ties that unit coverage to the
    full step by confirming that enabling ``global_mean_removal`` produces a
    different output than disabling it, with all other state (seed, weights,
    inputs) held fixed.
    """
    in_names = ["forcing_shared", "forcing_rad"]
    out_names = ["diagnostic_main", "diagnostic_rad"]
    means = {n: 0.0 for n in set(in_names + out_names)}
    _assert_global_mean_removal_affects_output(
        PerChannelGlobalMeanRemovalConfig(field_names=in_names),
        in_names,
        out_names,
        means,
    )


def test_step_shared_global_mean_removal_affects_output():
    """Verify SharedGlobalMeanRemoval is actually invoked during step.

    Companion to the PerChannel version above; same rationale.
    """
    in_names = ["surface_temperature", "air_temperature_0"]
    out_names = ["surface_temperature", "air_temperature_0"]
    means = {n: 280.0 for n in set(in_names + out_names)}
    _assert_global_mean_removal_affects_output(
        SharedGlobalMeanRemovalConfig(
            reference_field="surface_temperature",
            field_names=in_names,
        ),
        in_names,
        out_names,
        means,
    )


def test_step_shared_global_mean_removal_raises_on_masked_reference():
    in_names = ["surface_temperature", "air_temperature_0"]
    out_names = ["surface_temperature", "air_temperature_0"]
    means = {n: 280.0 for n in in_names}
    stds = {n: 5.0 for n in in_names}
    removal = SharedGlobalMeanRemovalConfig(
        reference_field="surface_temperature",
        field_names=in_names,
    )
    step = _make_global_mean_removal_step(
        removal, in_names, out_names, means=means, stds=stds
    )
    n_samples = 2
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )
    data_mask = {
        "surface_temperature": torch.tensor([True, False], device=fme.get_device()),
    }
    with pytest.raises(ValueError, match="masked"):
        step.step(
            args=StepArgs(
                input=input_data,
                next_step_input_data=next_step,
                n_ensemble=1,
                labels=None,
                data_mask=data_mask,
            ),
        )


def _make_shared_gmr_recorder_step(img_shape, n_samples, input_dropout):
    """SharedGMR+append_as_input step; records named and extra channels."""
    in_names = ["surface_temperature", "x"]
    out_names = ["surface_temperature"]
    means = {n: 280.0 for n in in_names + out_names}
    stds = {n: 5.0 for n in in_names + out_names}
    normalization = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(means=means, stds=stds),
    )
    device = fme.get_device()

    class RecordModule(nn.Module):
        def __init__(self):
            super().__init__()
            # named channels + 1 extra (surface_temperature_global_mean)
            self.last_named: torch.Tensor | None = None
            self.last_extra: torch.Tensor | None = None
            self._p = nn.Parameter(torch.zeros(1))

        def forward(self, inp):
            n_named = len(in_names)
            self.last_named = inp[:, :n_named].detach().clone()
            self.last_extra = inp[:, n_named:].detach().clone()
            return self._p.new_zeros(n_samples, len(out_names), *img_shape)

    recorder = RecordModule()
    step_config = SingleModuleStepConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": recorder}),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
        global_mean_removal=SharedGlobalMeanRemovalConfig(
            reference_field="surface_temperature",
            field_names=list(in_names),
            append_as_input=True,
        ),
        input_dropout=input_dropout,
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(img_shape[0], device=device),
            lon=torch.zeros(img_shape[1], device=device),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
        ),
        timestep=TIMESTEP,
    )
    from fme.core.step.single_module import SingleModuleStep

    step_obj = step_config.get_step(dataset_info, init_weights=lambda _: None)
    assert isinstance(step_obj, SingleModuleStep)
    return step_obj, recorder


def test_input_dropout_shared_gmr_reference_dropout_does_not_raise_at_step_time():
    """input_dropout targeting the shared reference field must not raise at step time.

    forward_transform receives args.data_mask (no dropout), so it sees the
    unmasked reference field and computes the correct offset.
    """
    img_shape = (4, 8)
    n_samples = 2
    device = fme.get_device()

    step_obj, _ = _make_shared_gmr_recorder_step(
        img_shape,
        n_samples,
        input_dropout=VariableMaskingConfig(per_variable={"surface_temperature": 1.0}),
    )
    input_data = {
        "surface_temperature": torch.full(
            (n_samples, *img_shape), 285.0, device=device
        ),
        "x": torch.full((n_samples, *img_shape), 275.0, device=device),
    }
    next_step_data = get_tensor_dict(
        step_obj.next_step_input_names, img_shape, n_samples
    )
    step_obj.module.torch_module.train()
    # Must not raise, even though surface_temperature is the shared reference.
    step_obj.step(
        StepArgs(input=input_data, next_step_input_data=next_step_data, n_ensemble=1)
    )


def test_input_dropout_shared_gmr_reference_zeroes_named_not_extra():
    """Masking the reference field zeroes only the named channel, not the extra."""
    img_shape = (4, 8)
    n_samples = 2
    device = fme.get_device()

    step_obj, recorder = _make_shared_gmr_recorder_step(
        img_shape,
        n_samples,
        input_dropout=VariableMaskingConfig(per_variable={"surface_temperature": 1.0}),
    )
    input_data = {
        "surface_temperature": torch.full(
            (n_samples, *img_shape), 285.0, device=device
        ),
        "x": torch.full((n_samples, *img_shape), 275.0, device=device),
    }
    next_step_data = get_tensor_dict(
        step_obj.next_step_input_names, img_shape, n_samples
    )
    step_obj.module.torch_module.train()
    step_obj.step(
        StepArgs(input=input_data, next_step_input_data=next_step_data, n_ensemble=1)
    )
    assert recorder.last_named is not None
    assert recorder.last_extra is not None
    # surface_temperature is the first named channel → must be zeroed
    assert (
        recorder.last_named[:, 0] == 0.0
    ).all(), "named 'surface_temperature' not zeroed when masked"
    # extra channel (surface_temperature_global_mean) must NOT be zeroed
    assert (
        recorder.last_extra[:, 0] != 0.0
    ).all(), "extra 'surface_temperature_global_mean' wrongly zeroed"


def test_input_dropout_shared_gmr_global_mean_name_zeroes_only_extra():
    """Masking 'surface_temperature_global_mean' zeroes only the extra channel."""
    img_shape = (4, 8)
    n_samples = 2
    device = fme.get_device()

    step_obj, recorder = _make_shared_gmr_recorder_step(
        img_shape,
        n_samples,
        input_dropout=VariableMaskingConfig(
            per_variable={"surface_temperature_global_mean": 1.0}
        ),
    )
    input_data = {
        "surface_temperature": torch.full(
            (n_samples, *img_shape), 285.0, device=device
        ),
        "x": torch.full((n_samples, *img_shape), 275.0, device=device),
    }
    next_step_data = get_tensor_dict(
        step_obj.next_step_input_names, img_shape, n_samples
    )
    step_obj.module.torch_module.train()
    step_obj.step(
        StepArgs(input=input_data, next_step_input_data=next_step_data, n_ensemble=1)
    )
    assert recorder.last_named is not None
    assert recorder.last_extra is not None
    # We verify the extra channel is zeroed (named channel check is omitted because
    # SharedGMR shifts surface_temperature to clim_mean → normalizes to 0 regardless).
    assert (
        recorder.last_extra[:, 0] == 0.0
    ).all(), "extra 'surface_temperature_global_mean' not zeroed when explicitly masked"


def test_extra_channel_names_have_global_mean_suffix():
    """extra_channel_names returns _global_mean-suffixed strings."""
    in_names = ["surface_temperature", "x"]
    out_names = ["surface_temperature"]
    means = {n: 280.0 for n in in_names + out_names}
    stds = {n: 5.0 for n in in_names + out_names}

    shared_config = SharedGlobalMeanRemovalConfig(
        reference_field="surface_temperature",
        field_names=list(in_names),
        append_as_input=True,
    )
    assert shared_config.extra_channel_names(in_names) == [
        "surface_temperature_global_mean"
    ]

    per_ch_config = PerChannelGlobalMeanRemovalConfig(
        field_names=list(in_names), append_as_input=True
    )
    assert per_ch_config.extra_channel_names(in_names) == [
        "surface_temperature_global_mean",
        "x_global_mean",
    ]

    # Verify the runtime objects match.
    from fme.core.normalizer import NormalizationConfig

    normalizer = NormalizationConfig(means=means, stds=stds).build(list(means))
    shared_rt = shared_config.build(normalizer, in_names)
    assert shared_rt.extra_channel_names == ["surface_temperature_global_mean"]

    per_ch_rt = per_ch_config.build(normalizer, in_names)
    assert per_ch_rt.extra_channel_names == [
        "surface_temperature_global_mean",
        "x_global_mean",
    ]


def _make_recorder_step(
    in_names, out_names, img_shape, n_samples, means, stds, input_dropout
):
    """Helper: build a step with PerChannelGMR+extra channels and a recording module."""
    normalization = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(means=means, stds=stds),
    )

    class RecordExtraModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_extra: torch.Tensor | None = None
            self._p = nn.Parameter(torch.zeros(1))

        def forward(self, inp):
            self.last_extra = inp[:, -len(in_names) :].detach().clone()
            return self._p.new_zeros(n_samples, len(out_names), *img_shape)

    recorder = RecordExtraModule()
    device = fme.get_device()
    step_config = SingleModuleStepConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": recorder}),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
        global_mean_removal=PerChannelGlobalMeanRemovalConfig(
            field_names=in_names,
            append_as_input=True,
        ),
        input_dropout=input_dropout,
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(img_shape[0], device=device),
            lon=torch.zeros(img_shape[1], device=device),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
        ),
        timestep=TIMESTEP,
    )
    from fme.core.step.single_module import SingleModuleStep

    step_obj = step_config.get_step(dataset_info, init_weights=lambda _: None)
    assert isinstance(step_obj, SingleModuleStep)
    return step_obj, recorder


def test_dropout_masking_named_channel_does_not_zero_extra_channel():
    """Masking 'a' via per_variable does NOT zero the 'a_global_mean' extra channel.

    The extra channel is independently named; to zero it, target 'a_global_mean'.
    """
    in_names = ["a", "b"]
    out_names = ["a"]
    img_shape = (4, 8)
    n_samples = 2
    device = fme.get_device()
    means = {n: 10.0 for n in in_names + out_names}
    stds = {n: 1.0 for n in in_names + out_names}

    step_obj, recorder = _make_recorder_step(
        in_names,
        out_names,
        img_shape,
        n_samples,
        means,
        stds,
        input_dropout=VariableMaskingConfig(per_variable={"a": 1.0}),
    )
    input_data = {
        n: torch.full((n_samples, *img_shape), 15.0, device=device) for n in in_names
    }
    next_step_data = get_tensor_dict(
        step_obj.next_step_input_names, img_shape, n_samples
    )

    step_obj.module.torch_module.train()
    step_obj.step(
        StepArgs(input=input_data, next_step_input_data=next_step_data, n_ensemble=1)
    )
    assert recorder.last_extra is not None
    # channel 0 = extra for "a_global_mean": NOT zeroed because dropout targets "a"
    assert (
        recorder.last_extra[:, 0] != 0.0
    ).all(), "extra channel for 'a_global_mean' wrongly zeroed when masking 'a'"
    # channel 1 = extra for "b_global_mean": also non-zero
    assert (
        recorder.last_extra[:, 1] != 0.0
    ).all(), "extra channel for 'b_global_mean' wrongly zeroed"


def test_dropout_masking_extra_channel_name_zeroes_only_extra_channel():
    """Masking 'a_global_mean' zeroes only the extra channel, not the named input."""
    in_names = ["a", "b"]
    out_names = ["a"]
    img_shape = (4, 8)
    n_samples = 2
    device = fme.get_device()
    means = {n: 10.0 for n in in_names + out_names}
    stds = {n: 1.0 for n in in_names + out_names}

    step_obj, recorder = _make_recorder_step(
        in_names,
        out_names,
        img_shape,
        n_samples,
        means,
        stds,
        input_dropout=VariableMaskingConfig(per_variable={"a_global_mean": 1.0}),
    )
    input_data = {
        n: torch.full((n_samples, *img_shape), 15.0, device=device) for n in in_names
    }
    next_step_data = get_tensor_dict(
        step_obj.next_step_input_names, img_shape, n_samples
    )

    step_obj.module.torch_module.train()
    step_obj.step(
        StepArgs(input=input_data, next_step_input_data=next_step_data, n_ensemble=1)
    )
    assert recorder.last_extra is not None
    # extra channel 0 = "a_global_mean": must be zeroed (rate=1.0)
    assert (
        recorder.last_extra[:, 0] == 0.0
    ).all(), "extra channel 'a_global_mean' not zeroed when explicitly masked"
    # extra channel 1 = "b_global_mean": NOT zeroed
    assert (
        recorder.last_extra[:, 1] != 0.0
    ).all(), "extra channel 'b_global_mean' wrongly zeroed"


def test_dropout_global_mean_removal_inverse_restores_sample_mean():
    """When dropout masks a field with PerChannelGMR, the inverse transform
    restores the per-sample global mean rather than zeroing it out.

    forward_transform receives args.data_mask (no dropout), so it computes
    the real shift from the unmasked input.  The network sees 0 for the
    masked field, outputs 0, and inverse_transform restores the shift, giving
    the original sample mean as the output.
    """
    in_names = ["a"]
    out_names = ["a"]
    img_shape = (4, 8)
    n_samples = 2
    input_value = 15.0
    clim_mean = 10.0

    class ReturnZerosModule(nn.Module):
        def __init__(self):
            super().__init__()
            self._p = nn.Parameter(torch.zeros(1))

        def forward(self, inp):
            return self._p.new_zeros(n_samples, len(out_names), *img_shape)

    normalization = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={n: clim_mean for n in in_names + out_names},
            stds={n: 1.0 for n in in_names + out_names},
        ),
    )
    step_config = SingleModuleStepConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": ReturnZerosModule()}),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
        global_mean_removal=PerChannelGlobalMeanRemovalConfig(field_names=in_names),
        input_dropout=VariableMaskingConfig(per_variable={"a": 1.0}),
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(img_shape[0], device=fme.get_device()),
            lon=torch.zeros(img_shape[1], device=fme.get_device()),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=fme.get_device()),
            bk=torch.arange(7, device=fme.get_device()),
        ),
        timestep=TIMESTEP,
    )
    step_obj = step_config.get_step(dataset_info, init_weights=lambda _: None)
    input_data = {
        "a": torch.full((n_samples, *img_shape), input_value, device=fme.get_device())
    }

    step_obj.module.torch_module.train()
    output, _ = step_obj.step(
        StepArgs(input=input_data, next_step_input_data={}, n_ensemble=1),
    )

    # forward_transform shifts "a" by (clim_mean - input_value) so the field is
    # at clim_mean → normalizes to 0.  Network outputs 0 → denormalizes to
    # clim_mean.  inverse_transform subtracts the cached shift, restoring the
    # original sample mean (input_value when input is spatially constant).
    torch.testing.assert_close(
        output["a"],
        torch.full_like(output["a"], input_value),
    )


def test_apply_extra_channel_mask_no_op_when_no_names_match():
    """Returns the original tensor object when no channel names appear in the mask."""
    extra = torch.ones(2, 3, 4, 4)
    data_mask = {"c": torch.tensor([True, False])}
    result = _apply_extra_channel_mask(extra, ["a", "b"], data_mask)
    assert result is extra


def test_apply_extra_channel_mask_zeros_masked_channels_per_sample():
    """Channels whose name is in data_mask are zeroed for samples where
    mask is False."""
    # batch=2, 2 channels, 1x1 spatial
    extra = torch.ones(2, 2, 1, 1)
    # sample 0 present=True (keep), sample 1 present=False (zero)
    data_mask = {"a": torch.tensor([True, False])}
    result = _apply_extra_channel_mask(extra, ["a", "b"], data_mask)
    assert result is not extra  # must be a clone
    assert result[0, 0, 0, 0].item() == 1.0  # sample 0, channel "a": kept
    assert result[1, 0, 0, 0].item() == 0.0  # sample 1, channel "a": zeroed
    assert result[0, 1, 0, 0].item() == 1.0  # channel "b": untouched
    assert result[1, 1, 0, 0].item() == 1.0  # channel "b": untouched


def test_apply_extra_channel_mask_multiple_channels_masked_independently():
    """Each channel uses its own per-sample mask independently."""
    extra = torch.ones(2, 2, 1, 1) * 5.0
    data_mask = {
        "x": torch.tensor([True, False]),  # sample 1 zeroed for channel 0
        "y": torch.tensor([False, True]),  # sample 0 zeroed for channel 1
    }
    result = _apply_extra_channel_mask(extra, ["x", "y"], data_mask)
    assert result[0, 0, 0, 0].item() == 5.0  # x, sample 0: present
    assert result[1, 0, 0, 0].item() == 0.0  # x, sample 1: absent
    assert result[0, 1, 0, 0].item() == 0.0  # y, sample 0: absent
    assert result[1, 1, 0, 0].item() == 5.0  # y, sample 1: present
