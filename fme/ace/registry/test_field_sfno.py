import pytest
import torch

from fme.ace.registry.field_sfno import NoiseConditionedFieldSFNOBuilder
from fme.ace.registry.registry import ModuleSelector
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device

IMG_SHAPE = (9, 18)


def _dataset_info() -> DatasetInfo:
    return DatasetInfo(img_shape=IMG_SHAPE)


def test_build_and_forward():
    builder = NoiseConditionedFieldSFNOBuilder(
        n_prognostic_channels=5,
        filter_type="makani-linear",
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
        filter_type="makani-linear",
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


def test_registered_and_from_state():
    selector = ModuleSelector(
        type="NoiseConditionedFieldSFNO",
        config={
            "n_prognostic_channels": 2,
            "filter_type": "makani-linear",
            "num_layers": 1,
            "noise_embed_dim": 4,
        },
    )
    model = selector.build(
        n_in_channels=3, n_out_channels=4, dataset_info=_dataset_info()
    ).to(get_device())
    out = model(torch.randn(1, 3, *IMG_SHAPE, device=get_device()))
    assert out.shape == (1, 4, *IMG_SHAPE)
