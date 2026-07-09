import dataclasses
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
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig, EnergyBudgetConfig
from fme.core.distributed.distributed import Distributed
from fme.core.distributed.non_distributed import DummyWrapper
from fme.core.labels import BatchLabels
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.global_mean_removal import (
    PerChannelGlobalMeanRemovalConfig,
    SharedGlobalMeanRemovalConfig,
    extra_channel_source_field,
)
from fme.core.step.multi_call import MultiCallConfig, MultiCallStep, MultiCallStepConfig
from fme.core.step.output import StepOutput
from fme.core.step.secondary_decoder import SecondaryDecoderConfig
from fme.core.step.secondary_module import SecondaryModuleStepConfig
from fme.core.step.single_module import (
    SingleModuleStep,
    SingleModuleStepConfig,
    _apply_input_mask,
    _build_channel_mask_dict,
)
from fme.core.step.step import StepABC, StepSelector
from fme.core.testing import get_dataset_info, trivial_network_and_loss_normalization
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.var_masking import (
    BernoulliMaskingConfig,
    MaskingGroupConfig,
    UniformMaskingConfig,
    VariableMaskingConfig,
)

from .radiation import SeparateRadiationStepConfig

DEFAULT_IMG_SHAPE = (45, 90)


def get_network_and_loss_normalization_config(
    names: list[str],
    dir: pathlib.Path | None = None,
) -> NetworkAndLossNormalizationConfig:
    if dir is None:
        return trivial_network_and_loss_normalization(names)
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
    dataset_info = get_dataset_info(
        img_shape=img_shape,
        all_labels=all_labels,
        device=fme.get_device(),
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
    output = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=BatchLabels.new_from_set(
                {"a", "b"}, n_samples=1, device=fme.get_device()
            ),
        ),
        wrapper=lambda x: x,
    ).output
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
            input=input_data, next_step_input_data=next_step_input_data, labels=None
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
    output = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
        ),
        wrapper=lambda x: x,
    ).output
    torch.testing.assert_close(output["diagnostic_main"], prescribed_value)


def test_step_returns_step_output_with_populated_detached_delta():
    selector = get_single_module_with_atmosphere_corrector_selector()
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(selector, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples=2)
    next_step_input_data = get_tensor_dict(step.next_step_input_names, img_shape, 2)
    result = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
        ),
    )
    assert isinstance(result, StepOutput)
    delta = result.corrector_diagnostics.delta
    assert delta  # the atmosphere corrector modifies fields
    for name, tensor in delta.items():
        assert name in result.output
        assert not tensor.requires_grad  # detached at the step boundary


def test_from_state_strips_deprecated_clip_frozen_precipitation():
    # Checkpoints serialized before clip_frozen_precipitation was folded into the
    # moisture corrector (always-on) carry a stale corrector key. from_state must
    # drop it so those older checkpoints still load.
    normalization = get_network_and_loss_normalization_config(
        names=["forcing_shared", "forcing_rad", "diagnostic_main", "diagnostic_rad"],
    )
    config = SingleModuleStepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
        ),
        in_names=["forcing_shared", "forcing_rad"],
        out_names=["diagnostic_main", "diagnostic_rad"],
        normalization=normalization,
        corrector=AtmosphereCorrectorConfig(conserve_dry_air=True),
    )
    state = dataclasses.asdict(config)
    state["corrector"]["clip_frozen_precipitation"] = True
    restored = SingleModuleStepConfig.from_state(state)
    assert isinstance(restored.corrector, AtmosphereCorrectorConfig)
    assert restored.corrector.conserve_dry_air is True
    assert not hasattr(restored.corrector, "clip_frozen_precipitation")


def test_step_empty_delta_when_no_corrector():
    selector = get_single_module_selector()
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(selector, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples=2)
    next_step_input_data = get_tensor_dict(step.next_step_input_names, img_shape, 2)
    result = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
        ),
    )
    assert isinstance(result, StepOutput)
    assert dict(result.corrector_diagnostics.delta) == {}


def _single_module_corrector_prescribed_selector(
    force_positive_names: list[str],
    prescribed_prognostic_names: list[str],
) -> StepSelector:
    normalization = get_network_and_loss_normalization_config(
        names=["forcing_shared", "forcing_rad", "diagnostic_main", "diagnostic_rad"],
    )
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
                ),
                in_names=["forcing_shared", "forcing_rad"],
                out_names=["diagnostic_main", "diagnostic_rad"],
                normalization=normalization,
                corrector=AtmosphereCorrectorConfig(
                    force_positive_names=force_positive_names
                ),
                prescribed_prognostic_names=prescribed_prognostic_names,
            ),
        ),
    )


def test_step_boundary_disjointness_passes_when_disjoint():
    # corrector modifies diagnostic_rad; the post-corrector prescription
    # overwrites diagnostic_main -> disjoint, so the step succeeds.
    selector = _single_module_corrector_prescribed_selector(
        force_positive_names=["diagnostic_rad"],
        prescribed_prognostic_names=["diagnostic_main"],
    )
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(selector, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples=2)
    next_step_input_data = get_tensor_dict(step.next_step_input_names, img_shape, 2)
    next_step_input_data["diagnostic_main"] = torch.zeros(
        2, *img_shape, device=fme.get_device()
    )
    result = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
        ),
    )
    assert "diagnostic_rad" in result.corrector_diagnostics.delta
    assert "diagnostic_main" not in result.corrector_diagnostics.delta


def test_step_boundary_overlap_prescribes_and_drops_stale_delta():
    # corrector modifies diagnostic_main and the post-corrector prescription also
    # writes diagnostic_main -> the prescribed value wins and its now-stale delta
    # is dropped so ``output - delta = network_output`` holds for reported deltas.
    selector = _single_module_corrector_prescribed_selector(
        force_positive_names=["diagnostic_main"],
        prescribed_prognostic_names=["diagnostic_main"],
    )
    img_shape = DEFAULT_IMG_SHAPE
    step = get_step(selector, img_shape)
    input_data = get_tensor_dict(step.input_names, img_shape, n_samples=2)
    next_step_input_data = get_tensor_dict(step.next_step_input_names, img_shape, 2)
    prescribed_value = torch.full((2, *img_shape), 3.5, device=fme.get_device())
    next_step_input_data["diagnostic_main"] = prescribed_value
    result = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
        ),
    )
    # prescribed value wins over both the network output and the corrector
    torch.testing.assert_close(result.output["diagnostic_main"], prescribed_value)
    # the overwritten name's stale delta is dropped from the diagnostics
    assert "diagnostic_main" not in result.corrector_diagnostics.delta


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
    output = step.step(
        args=StepArgs(
            input=input_data, next_step_input_data=next_step_input_data, labels=None
        ),
    ).output
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
        input=input_data, next_step_input_data=next_step_input_data, labels=None
    )
    out1 = step1.step(args=args).output
    out2 = step2.step(args=args).output
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
    output = step.step(
        args=StepArgs(
            input=input_data, next_step_input_data=next_step_input_data, labels=None
        ),
    ).output
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
    output_no_mask = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
            data_mask=None,
        ),
    ).output
    output_with_mask = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
            data_mask=data_mask,
        ),
    ).output
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


def test_build_channel_mask_dict_gmr_extra_inherits_source_mask():
    # The GMR sentinel channel `__gmr_extra__a` must inherit `a`'s mask
    # because its value is zeroed in forward_transform when `a` is masked;
    # otherwise the network sees a 0-valued extra with a 1-valued mask
    # (i.e. contradictory "present" signal on a masked sample).
    packed = torch.zeros(2, 4, 4, 8)
    data_mask = {"a": torch.tensor([True, False])}
    result = _build_channel_mask_dict(
        ["a", "b", "__gmr_extra__a", "__gmr_extra__b"], data_mask, packed
    )
    assert result["a"][0, 0, 0] == 1.0
    assert result["a"][1, 0, 0] == 0.0
    assert result["__gmr_extra__a"][0, 0, 0] == 1.0
    assert result["__gmr_extra__a"][1, 0, 0] == 0.0
    assert (result["b"] == 1.0).all()
    assert (result["__gmr_extra__b"] == 1.0).all()


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
    output_no_mask = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
            data_mask=None,
        ),
    ).output
    output_with_mask = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
            data_mask=data_mask,
        ),
    ).output
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
    output_no_mask = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
            data_mask=None,
        ),
    ).output
    all_unmasked = {
        name: torch.ones(n_samples, dtype=torch.bool, device=fme.get_device())
        for name in step.input_names
    }
    output_all_unmasked = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_input_data,
            labels=None,
            data_mask=all_unmasked,
        ),
    ).output
    for name in ["diagnostic_main", "diagnostic_rad"]:
        assert output_no_mask[name].shape == (n_samples, *img_shape)
        torch.testing.assert_close(output_no_mask[name], output_all_unmasked[name])


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
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None),
    ).output
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
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None),
    ).output
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
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None),
    ).output
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
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None),
    ).output
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
    baseline_output = step_baseline.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None),
    ).output
    removal_output = step_with_removal.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None),
    ).output
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


def test_step_per_channel_global_mean_removal_with_channel_masks():
    """When both ``include_channel_mask_inputs`` and per-channel GMR extras
    are enabled, the GMR extras should be packed as ordinary input
    channels and receive their own mask channels.  Total network input
    channels = ``2 * (n_in_names + n_extras)``.
    """
    in_names = ["forcing_shared", "forcing_rad"]
    out_names = ["diagnostic_main", "diagnostic_rad"]
    all_names = list(set(in_names + out_names))
    normalization = trivial_network_and_loss_normalization(all_names)
    removal = PerChannelGlobalMeanRemovalConfig(
        field_names=in_names, append_as_input=True
    )
    config = StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
                ),
                in_names=in_names,
                out_names=out_names,
                normalization=normalization,
                include_channel_mask_inputs=True,
                global_mean_removal=removal,
            ),
        ),
    )
    step = get_step(config, DEFAULT_IMG_SHAPE)
    assert isinstance(step, SingleModuleStep)
    # 2 real inputs + 2 GMR extras → 4 packed input channels (doubled to 8
    # by the channel-mask append in network_call).
    assert len(step.in_packer.names) == 4

    n_samples = 2
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None),
    ).output
    for name in out_names:
        assert output[name].shape == (n_samples, *DEFAULT_IMG_SHAPE)


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
                labels=None,
                data_mask=data_mask,
            ),
        )


def _make_single_module_step(
    input_dropout: VariableMaskingConfig | None,
    include_channel_mask_inputs: bool = False,
) -> SingleModuleStep:
    in_names = ["forcing_shared", "forcing_rad"]
    out_names = ["diagnostic_main", "diagnostic_rad"]
    normalization = get_network_and_loss_normalization_config(
        names=list(set(in_names + out_names))
    )
    config = StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
                ),
                in_names=in_names,
                out_names=out_names,
                normalization=normalization,
                include_channel_mask_inputs=include_channel_mask_inputs,
                input_dropout=input_dropout,
            )
        ),
    )
    step = get_step(config, DEFAULT_IMG_SHAPE)
    assert isinstance(step, SingleModuleStep)
    return step


def _make_gmr_input_dropout_step(
    input_dropout: VariableMaskingConfig, include_channel_mask_inputs: bool
):
    in_names = ["forcing_shared", "forcing_rad"]
    out_names = ["diagnostic_main", "diagnostic_rad"]
    all_names = list(set(in_names + out_names))
    normalization = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={name: 0.0 for name in all_names},
            stds={name: 1.0 for name in all_names},
        ),
    )
    config = StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet",
                    config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
                ),
                in_names=in_names,
                out_names=out_names,
                normalization=normalization,
                global_mean_removal=PerChannelGlobalMeanRemovalConfig(
                    field_names=in_names, append_as_input=True
                ),
                include_channel_mask_inputs=include_channel_mask_inputs,
                input_dropout=input_dropout,
            )
        ),
    )
    step = get_step(config, DEFAULT_IMG_SHAPE)
    assert isinstance(step, SingleModuleStep)
    return step


def _presence_mask(step: SingleModuleStep, present: bool) -> TensorDict:
    """An input_dropout mask (broadcast over batch) that keeps or drops all."""
    value = torch.full((1,), present, dtype=torch.bool, device=fme.get_device())
    return {name: value.clone() for name in step.in_packer.names}


def _inject_input_dropout_mask(
    step: SingleModuleStep, mask: TensorMapping | None
) -> None:
    """Force the mask drawn by the next ``step`` call.

    The Step samples its own mask internally on every forward step; tests
    override ``_draw_input_dropout_mask`` with a deterministic mask (or None
    for no dropout) so the application path is exercised without relying on
    the random sampler.
    """
    step._draw_input_dropout_mask = lambda: mask  # type: ignore[method-assign]


def test_input_dropout_unknown_group_variable_raises_at_build():
    """A typo'd group variable fails loudly when the Step is built."""
    bad = VariableMaskingConfig(
        override_groups=[
            MaskingGroupConfig(
                variables=["typo"], masking=BernoulliMaskingConfig(rate=0.5)
            )
        ]
    )
    with pytest.raises(ValueError, match="not in packed input channels"):
        _make_single_module_step(bad)


def test_input_dropout_group_variable_accepts_gmr_extra_sentinel():
    """A group may target a GMR extra sentinel; build must not reject it."""
    from fme.core.step.global_mean_removal import _extra_channel_name

    sentinel = _extra_channel_name("forcing_shared")
    config = VariableMaskingConfig(
        override_groups=[
            MaskingGroupConfig(
                variables=[sentinel], masking=BernoulliMaskingConfig(rate=0.5)
            )
        ]
    )
    _make_gmr_input_dropout_step(config, include_channel_mask_inputs=False)


def test_input_dropout_mask_zeros_inputs():
    """A supplied input_dropout_mask deterministically zeros masked inputs.

    A full-drop mask must change outputs versus no mask; an all-present mask
    must reproduce the no-mask output exactly.
    """
    step = _make_single_module_step(
        VariableMaskingConfig(default=UniformMaskingConfig(1))
    )
    step.module.torch_module.eval()  # module determinism; late dropout still applies
    n_samples = 4
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )

    def _run(mask):
        _inject_input_dropout_mask(step, mask)
        return step.step(
            args=StepArgs(
                input=input_data,
                next_step_input_data=next_step,
                labels=None,
            )
        ).output

    out_none = _run(None)
    out_full_drop = _run(_presence_mask(step, present=False))
    out_all_present = _run(_presence_mask(step, present=True))
    assert any(
        not torch.allclose(out_full_drop[name], out_none[name])
        for name in step.out_names
    ), "full-drop mask should change outputs"
    for name in step.out_names:
        torch.testing.assert_close(out_all_present[name], out_none[name])


def test_input_dropout_mask_indicator_reflects_combined_presence():
    """With include_channel_mask_inputs, the indicator equals real & synthetic.

    Inspect the indicator half of the packed network input: a channel dropped
    by input_dropout_mask must have indicator 0 and a zeroed data channel; an
    undropped channel must have indicator 1.
    """
    step = _make_single_module_step(
        VariableMaskingConfig(default=UniformMaskingConfig(1)),
        include_channel_mask_inputs=True,
    )
    step.module.torch_module.eval()
    n_samples = 2
    in_names = step.in_packer.names
    n_channels = len(in_names)
    # Drop the first channel only (mask broadcast over batch).
    dropout_mask = {
        name: torch.full((1,), i != 0, dtype=torch.bool, device=fme.get_device())
        for i, name in enumerate(in_names)
    }
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )

    captured: list[torch.Tensor] = []

    def _pre_hook(module, args):
        captured.append(args[0].detach().cpu())

    handle = step.module.torch_module.register_forward_pre_hook(_pre_hook)
    try:
        _inject_input_dropout_mask(step, dropout_mask)
        step.step(
            args=StepArgs(
                input=input_data,
                next_step_input_data=next_step,
                labels=None,
            )
        )
    finally:
        handle.remove()

    packed = captured[0]
    data_half = packed[:, :n_channels]
    indicator_half = packed[:, n_channels:]
    assert (indicator_half[:, 0] == 0.0).all()
    assert (data_half[:, 0] == 0.0).all()
    assert (indicator_half[:, 1:] == 1.0).all()


def test_input_dropout_mask_and_combine_with_data_mask():
    """AND-combine: data_mask=0 wins even when dropout leaves the channel present.

    Guards against fallback-priority resurrecting a genuinely-missing variable.
    """
    step = _make_single_module_step(
        VariableMaskingConfig(default=UniformMaskingConfig(1)),
        include_channel_mask_inputs=True,
    )
    step.module.torch_module.eval()
    n_samples = 2
    in_names = step.in_packer.names
    n_channels = len(in_names)
    # Channel 0 is genuinely missing (data_mask=0) but not synthetically dropped.
    data_mask = {
        in_names[0]: torch.zeros(n_samples, dtype=torch.bool, device=fme.get_device()),
    }
    dropout_mask = _presence_mask(step, present=True)
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )

    captured: list[torch.Tensor] = []

    def _pre_hook(module, args):
        captured.append(args[0].detach().cpu())

    handle = step.module.torch_module.register_forward_pre_hook(_pre_hook)
    try:
        _inject_input_dropout_mask(step, dropout_mask)
        step.step(
            args=StepArgs(
                input=input_data,
                next_step_input_data=next_step,
                labels=None,
                data_mask=data_mask,
            )
        )
    finally:
        handle.remove()

    packed = captured[0]
    # combined present = real(0) & synthetic(1) = 0 -> indicator 0, input zeroed
    assert (packed[:, n_channels] == 0.0).all()  # indicator for channel 0
    assert (packed[:, 0] == 0.0).all()  # data channel 0 zeroed


def test_input_dropout_mask_gmr_extras_independently_maskable():
    """GMR extra channels are dropped independently; the indicator agrees."""
    step = _make_gmr_input_dropout_step(
        VariableMaskingConfig(default=UniformMaskingConfig(1)),
        include_channel_mask_inputs=True,
    )
    step.module.torch_module.eval()
    names = step.in_packer.names
    assert len(names) == 4  # 2 named + 2 GMR extras
    n_channels = len(names)
    gmr_index = next(
        i
        for i, name in enumerate(names)
        if extra_channel_source_field(name) is not None
    )
    n_samples = 2
    dropout_mask = {
        name: torch.full(
            (1,), i != gmr_index, dtype=torch.bool, device=fme.get_device()
        )
        for i, name in enumerate(names)
    }
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )

    captured: list[torch.Tensor] = []

    def _pre_hook(module, args):
        captured.append(args[0].detach().cpu())

    handle = step.module.torch_module.register_forward_pre_hook(_pre_hook)
    try:
        _inject_input_dropout_mask(step, dropout_mask)
        step.step(
            args=StepArgs(
                input=input_data,
                next_step_input_data=next_step,
                labels=None,
            )
        )
    finally:
        handle.remove()

    packed = captured[0]
    data_half = packed[:, :n_channels]
    indicator_half = packed[:, n_channels:]
    assert (data_half[:, gmr_index] == 0.0).all()
    assert (indicator_half[:, gmr_index] == 0.0).all()
    other = [i for i in range(n_channels) if i != gmr_index]
    assert (indicator_half[:, other] == 1.0).all()


def test_draw_input_dropout_mask_shape_and_dtype():
    # rate-1 drops every channel, so emitted keys are exactly the packed input channels.
    step = _make_single_module_step(
        VariableMaskingConfig(default=BernoulliMaskingConfig(rate=1.0))
    )
    step.module.torch_module.train()
    mask = step._draw_input_dropout_mask()
    assert mask is not None
    # keyed by dropped channels, one [1] bool tensor each; present channels omitted.
    assert set(mask.keys()) == set(step.in_packer.names)
    for name in step.in_packer.names:
        assert mask[name].shape == (1,)
        assert mask[name].dtype == torch.bool
        assert not bool(mask[name].item())  # emitted channels are dropped


def test_draw_input_dropout_mask_omits_present_channels():
    # rate-0 default drops nothing, so no channels are emitted.
    step = _make_single_module_step(
        VariableMaskingConfig(default=BernoulliMaskingConfig(rate=0.0))
    )
    step.module.torch_module.train()
    mask = step._draw_input_dropout_mask()
    assert mask == {}


def test_draw_input_dropout_mask_none_when_unset():
    step = _make_single_module_step(None)
    step.module.torch_module.train()
    assert step._draw_input_dropout_mask() is None


def test_draw_input_dropout_mask_none_in_eval_mode():
    step = _make_single_module_step(
        VariableMaskingConfig(default=UniformMaskingConfig(1))
    )
    step.module.torch_module.eval()
    # configured, but eval mode disables dropout sampling
    assert step._draw_input_dropout_mask() is None


def test_draw_input_dropout_mask_includes_gmr_extras():
    # rate-1 default drops every channel so GMR extra sentinels appear too.
    step = _make_gmr_input_dropout_step(
        VariableMaskingConfig(default=BernoulliMaskingConfig(rate=1.0)),
        include_channel_mask_inputs=False,
    )
    step.module.torch_module.train()
    mask = step._draw_input_dropout_mask()
    assert mask is not None
    # GMR extra sentinel channels are independently maskable
    assert set(mask.keys()) == set(step.in_packer.names)
    assert len(step.in_packer.names) == 4


def test_input_dropout_mask_not_passed_to_global_mean_removal():
    """Synthetic dropout is applied late (after GMR); GMR sees only data_mask."""
    step = _make_gmr_input_dropout_step(
        VariableMaskingConfig(default=UniformMaskingConfig(2)),
        include_channel_mask_inputs=False,
    )
    step.module.torch_module.eval()  # disable any in-step sampling
    n_samples = 2
    input_data = get_tensor_dict(step.input_names, DEFAULT_IMG_SHAPE, n_samples)
    next_step = get_tensor_dict(
        step.next_step_input_names, DEFAULT_IMG_SHAPE, n_samples
    )
    dropout_mask = _presence_mask(step, present=False)
    _inject_input_dropout_mask(step, dropout_mask)
    out_with_mask = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step,
            labels=None,
        )
    ).output
    _inject_input_dropout_mask(step, None)
    out_without_mask = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step,
            labels=None,
        )
    ).output
    # Step runs without GMR raising on a "missing" reference field; GMR never sees mask.
    assert set(out_with_mask) == set(out_without_mask)


def test_step_train_eval_toggle_propagates_to_modules():
    step = get_step(get_single_module_selector(), DEFAULT_IMG_SHAPE)

    assert all(module.training for module in step.modules)
    assert step._training

    step.eval()
    assert all(not module.training for module in step.modules)
    assert not step._training

    step.train()
    assert all(module.training for module in step.modules)
    assert step._training

    step.train(False)
    assert all(not module.training for module in step.modules)
    assert not step._training


def test_corrector_state_not_in_state_by_default():
    step = get_step(get_single_module_selector(), DEFAULT_IMG_SHAPE)
    assert "corrector" not in step.get_state()


def test_load_state_calls_corrector_with_empty_state_when_missing():
    selector = get_single_module_selector()
    config = dict(selector.config)
    config["corrector"] = {
        **config["corrector"],
        "corrector_disabled_epochs": 1,
    }
    selector = StepSelector(type=selector.type, config=config)
    step = get_step(selector, DEFAULT_IMG_SHAPE)
    state = step.get_state()
    del state["corrector"]

    with pytest.raises(ValueError, match="corrector_disabled"):
        step.load_state(state)


def test_multi_call_step_forwards_set_epoch():
    wrapped_step = unittest.mock.MagicMock(spec=StepABC)
    config = MultiCallStepConfig(
        wrapped_step=get_single_module_selector(),
        config=None,
        include_multi_call_in_loss=False,
    )
    step = MultiCallStep(wrapped_step=wrapped_step, config=config)
    step.set_epoch(3)
    wrapped_step.set_epoch.assert_called_once_with(3)


def test_multi_call_step_forwards_train_eval():
    wrapped_step = unittest.mock.MagicMock(spec=StepABC)
    wrapped_step.modules = nn.ModuleList()
    config = MultiCallStepConfig(
        wrapped_step=get_single_module_selector(),
        config=None,
        include_multi_call_in_loss=False,
    )
    step = MultiCallStep(wrapped_step=wrapped_step, config=config)

    step.eval()
    wrapped_step.train.assert_called_once_with(False)

    wrapped_step.train.reset_mock()
    step.train()
    wrapped_step.train.assert_called_once_with(True)
