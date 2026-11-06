import dataclasses

import dacite
import pytest
import torch

from fme.ace.registry.field_sfno import NoiseConditionedFieldSFNOBuilder
from fme.ace.registry.registry import ModuleSelector
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device

IMG_SHAPE = (9, 18)


def _dataset_info() -> DatasetInfo:
    return DatasetInfo(img_shape=IMG_SHAPE)


@pytest.mark.parametrize("filter_type", ["linear", "makani-linear"])
def test_build_and_forward(filter_type):
    builder = NoiseConditionedFieldSFNOBuilder(
        n_prognostic_channels=5,
        filter_type=filter_type,
        num_layers=2,
        noise_embed_dim=4,
        context_pos_embed_dim=3,
        forcing_embed_dim=8,
    )
    model = builder.build(
        n_in_channels=8, n_out_channels=7, dataset_info=_dataset_info()
    ).to(get_device())
    x = torch.randn(2, 8, *IMG_SHAPE, device=get_device())
    out = model(x)
    assert out.shape == (2, 7, *IMG_SHAPE)


def test_isotropic_noise_builds():
    builder = NoiseConditionedFieldSFNOBuilder(
        n_prognostic_channels=2,
        num_layers=1,
        noise_embed_dim=4,
        noise_type="isotropic",
    )
    model = builder.build(
        n_in_channels=3, n_out_channels=2, dataset_info=_dataset_info()
    ).to(get_device())
    out = model(torch.randn(2, 3, *IMG_SHAPE, device=get_device()))
    assert out.shape == (2, 2, *IMG_SHAPE)


@pytest.mark.parametrize("n_in, n_out", [(3, 5), (5, 3)])
def test_too_few_channels_raises(n_in, n_out):
    builder = NoiseConditionedFieldSFNOBuilder(n_prognostic_channels=4, num_layers=1)
    with pytest.raises(ValueError, match="n_prognostic_channels"):
        builder.build(
            n_in_channels=n_in, n_out_channels=n_out, dataset_info=_dataset_info()
        )


def test_groups_must_divide_output_channels():
    builder = NoiseConditionedFieldSFNOBuilder(
        n_prognostic_channels=4, num_layers=1, filter_num_groups=4
    )
    with pytest.raises(ValueError, match="filter_num_groups"):
        builder.build(n_in_channels=5, n_out_channels=7, dataset_info=_dataset_info())


def test_invalid_network_option_raises_at_config():
    with pytest.raises(ValueError, match="activation function"):
        NoiseConditionedFieldSFNOBuilder(
            n_prognostic_channels=1, activation_function="tanh"
        )


def test_module_selector_builds():
    selector = ModuleSelector(
        type="NoiseConditionedFieldSFNO",
        config={
            "n_prognostic_channels": 2,
            "num_layers": 1,
            "noise_embed_dim": 4,
        },
    )
    model = selector.build(
        n_in_channels=3, n_out_channels=4, dataset_info=_dataset_info()
    ).to(get_device())
    out = model(torch.randn(1, 3, *IMG_SHAPE, device=get_device()))
    assert out.shape == (1, 4, *IMG_SHAPE)


def test_from_state_round_trip():
    builder = NoiseConditionedFieldSFNOBuilder(
        n_prognostic_channels=2, num_layers=1, noise_embed_dim=4, local_blocks=[0]
    )
    state = dataclasses.asdict(builder)
    assert NoiseConditionedFieldSFNOBuilder.from_state(state) == builder
    with pytest.raises(dacite.UnexpectedDataError):
        NoiseConditionedFieldSFNOBuilder.from_state({**state, "embed_dim": 4})


def test_module_state_round_trip():
    torch.manual_seed(0)
    builder = NoiseConditionedFieldSFNOBuilder(
        n_prognostic_channels=2, num_layers=1, noise_embed_dim=4, gamma_init=0.5
    )
    model = builder.build(
        n_in_channels=3, n_out_channels=4, dataset_info=_dataset_info()
    ).to(get_device())
    model2 = builder.build(
        n_in_channels=3, n_out_channels=4, dataset_info=_dataset_info()
    ).to(get_device())
    model2.load_state_dict(model.state_dict())
    x = torch.randn(1, 3, *IMG_SHAPE, device=get_device())
    with torch.no_grad():
        torch.manual_seed(1)
        out = model(x)
        torch.manual_seed(1)
        out2 = model2(x)
    torch.testing.assert_close(out, out2)
