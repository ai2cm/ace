import dataclasses
import datetime

import pytest
import torch

import fme
from fme.ace.step.camulator import (
    CrossFormerConfig,
    CrossFormerStepConfig,
    NoiseConditionedCrossFormerConfig,
)
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.step.args import StepArgs
from fme.core.step.global_mean_removal import PerChannelGlobalMeanRemovalConfig
from fme.core.step.step import StepSelector

IMG_SHAPE = (48, 96)
TIMESTEP = datetime.timedelta(hours=6)

FORCING_NAMES = ["forcing_a"]
ATM_PROGNOSTIC_NAMES = ["atm_var"]
ATM_LEVELS = 2
SURF_PROGNOSTIC_NAMES = ["surf_var"]
SURF_DIAGNOSTIC_NAMES = ["surf_diag"]
ATM_DIAGNOSTIC_NAMES: list[str] = []


def make_normalization(names: list[str]) -> NetworkAndLossNormalizationConfig:
    return NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={name: 0.0 for name in names},
            stds={name: 1.0 for name in names},
        ),
    )


def make_crossformer_config() -> CrossFormerConfig:
    return CrossFormerConfig(
        frames=1,
        dim=[16, 32, 64, 128],
        depth=[1, 1, 1, 1],
        dim_head=8,
        global_window_size=[4, 4, 2, 1],
        local_window_size=3,
        cross_embed_kernel_sizes=[[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]],
        cross_embed_strides=[2, 2, 2, 2],
        use_spectral_norm=False,
        interp=True,
    )


def make_nc_crossformer_config() -> NoiseConditionedCrossFormerConfig:
    return NoiseConditionedCrossFormerConfig(
        frames=1,
        dim=[16, 32, 64, 128],
        depth=[1, 1, 1, 1],
        dim_head=8,
        global_window_size=[4, 4, 2, 1],
        local_window_size=3,
        cross_embed_kernel_sizes=[[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]],
        cross_embed_strides=[2, 2, 2, 2],
        use_spectral_norm=False,
        interp=True,
        noise_embed_dim=8,
    )


def make_config(builder) -> CrossFormerStepConfig:
    all_names = (
        FORCING_NAMES
        + SURF_PROGNOSTIC_NAMES
        + [f"{v}_{i}" for v in ATM_PROGNOSTIC_NAMES for i in range(ATM_LEVELS)]
        + SURF_DIAGNOSTIC_NAMES
    )
    return CrossFormerStepConfig(
        builder=builder,
        forcing_names=FORCING_NAMES,
        atmosphere_prognostic_names=ATM_PROGNOSTIC_NAMES,
        atmosphere_levels=ATM_LEVELS,
        surface_prognostic_names=SURF_PROGNOSTIC_NAMES,
        surface_diagnostic_names=SURF_DIAGNOSTIC_NAMES,
        atmosphere_diagnostic_names=ATM_DIAGNOSTIC_NAMES,
        normalization=make_normalization(all_names),
    )


def make_dataset_info() -> DatasetInfo:
    device = fme.get_device()
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(IMG_SHAPE[0], device=device),
            lon=torch.zeros(IMG_SHAPE[1], device=device),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
        ),
        timestep=TIMESTEP,
    )


def get_tensor_dict(names, n_samples=2):
    device = fme.get_device()
    return {name: torch.rand(n_samples, *IMG_SHAPE, device=device) for name in names}


def test_crossformer_step_output_names():
    config = make_config(make_crossformer_config())
    dataset_info = make_dataset_info()
    step = config.get_step(dataset_info, init_weights=lambda _: None)
    input_data = get_tensor_dict(step.input_names)
    next_step_data = get_tensor_dict(step.next_step_input_names)
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_data,
            labels=None,
            data_mask=None,
        )
    )
    assert set(output.keys()) == set(config.out_names)


def test_crossformer_step_selector_loads_yaml_style_config():
    config = make_config(make_crossformer_config())
    selector = StepSelector(type="CrossFormer", config=dataclasses.asdict(config))
    dataset_info = make_dataset_info()

    step = selector.get_step(dataset_info, init_weights=lambda _: None)
    input_data = get_tensor_dict(step.input_names)
    next_step_data = get_tensor_dict(step.next_step_input_names)
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_data,
            labels=None,
            data_mask=None,
        )
    )

    assert set(output.keys()) == set(config.out_names)


@pytest.mark.parametrize(
    "builder_factory", [make_crossformer_config, make_nc_crossformer_config]
)
def test_crossformer_step_config_rejects_multi_frame_builder(builder_factory):
    builder = dataclasses.replace(builder_factory(), frames=2)

    with pytest.raises(ValueError, match="builder.frames == 1"):
        make_config(builder)


def test_noise_conditioned_crossformer_step_output_names():
    config = make_config(make_nc_crossformer_config())
    dataset_info = make_dataset_info()
    step = config.get_step(dataset_info, init_weights=lambda _: None)
    input_data = get_tensor_dict(step.input_names)
    next_step_data = get_tensor_dict(step.next_step_input_names)
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_data,
            labels=None,
            data_mask=None,
        )
    )
    assert set(output.keys()) == set(config.out_names)


def test_crossformer_step_global_mean_removal_with_extra_channels():
    config = make_config(make_crossformer_config())
    config.global_mean_removal = PerChannelGlobalMeanRemovalConfig(
        field_names=[SURF_PROGNOSTIC_NAMES[0]],
        append_as_input=True,
    )
    dataset_info = make_dataset_info()

    step = config.get_step(dataset_info, init_weights=lambda _: None)
    input_data = get_tensor_dict(step.input_names)
    next_step_data = get_tensor_dict(step.next_step_input_names)
    output, _ = step.step(
        args=StepArgs(
            input=input_data,
            next_step_input_data=next_step_data,
            labels=None,
            data_mask=None,
        )
    )

    assert set(output.keys()) == set(config.out_names)
    assert step.module.module.input_only_channels == len(FORCING_NAMES) + 1


def test_noise_conditioned_crossformer_noise_divergence():
    """
    CLN zero-init makes outputs noise-independent at init; diverge after a step.

    This mirrors test_nc_swin_transformer_noise_divergence from the swin tests.
    """
    config = make_nc_crossformer_config()
    module = config.build(
        n_atmo_channels=len(ATM_PROGNOSTIC_NAMES),
        n_atmo_groups=ATM_LEVELS,
        n_surf_channels=len(SURF_PROGNOSTIC_NAMES),
        n_aux_channels=len(FORCING_NAMES),
        n_atmo_diagnostic_channels=len(ATM_DIAGNOSTIC_NAMES),
        n_surf_diagnostic_channels=len(SURF_DIAGNOSTIC_NAMES),
        img_shape=IMG_SHAPE,
    ).to(fme.get_device())
    module.train()
    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)

    n_in = (
        len(ATM_PROGNOSTIC_NAMES) * ATM_LEVELS
        + len(SURF_PROGNOSTIC_NAMES)
        + len(FORCING_NAMES)
    )
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())

    # At init: CLN noise convs are zero → noise-independent.
    with torch.no_grad():
        out1 = module(x)
        out2 = module(x)
    assert torch.allclose(out1, out2), "Expected noise-independence at init"

    # One optimizer step pushes CLN noise convs off zero.
    out = module(x)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    # After step: two independent forward passes should differ.
    with torch.no_grad():
        out1 = module(x)
        out2 = module(x)
    assert not torch.allclose(
        out1, out2
    ), "Expected noise-dependence after optimizer step"


def test_crossformer_diagnostic_names():
    config = make_config(make_crossformer_config())
    assert set(config.diagnostic_names) == set(SURF_DIAGNOSTIC_NAMES)


def test_crossformer_step_config_from_state_roundtrip():
    config = make_config(make_crossformer_config())
    state = config.get_state()
    restored = CrossFormerStepConfig.from_state(state)
    assert restored.get_state() == state


def test_nc_crossformer_step_config_from_state_roundtrip():
    config = make_config(make_nc_crossformer_config())
    state = config.get_state()
    restored = CrossFormerStepConfig.from_state(state)
    assert restored.get_state() == state


def test_crossformer_step_config_in_names_ordering():
    """in_names should be: forcing -> surface -> atmosphere."""
    config = make_config(make_crossformer_config())
    forcing_end = len(FORCING_NAMES)
    surf_end = forcing_end + len(SURF_PROGNOSTIC_NAMES)
    assert config.in_names[:forcing_end] == FORCING_NAMES
    assert config.in_names[forcing_end:surf_end] == SURF_PROGNOSTIC_NAMES
    # atmosphere names follow
    expected_atm = [f"{v}_{i}" for v in ATM_PROGNOSTIC_NAMES for i in range(ATM_LEVELS)]
    assert config.in_names[surf_end:] == expected_atm
