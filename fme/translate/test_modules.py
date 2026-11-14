import pytest
import torch

from fme.core.testing import get_dataset_info
from fme.translate.modules import TransformSelector


def _sfno_same_grid_selector() -> TransformSelector:
    return TransformSelector(
        type="same_grid",
        config={
            "module": {
                "type": "SphericalFourierNeuralOperatorNet",
                "config": {"scale_factor": 1, "embed_dim": 2, "num_layers": 1},
            }
        },
    )


def test_same_grid_builds_on_matching_grids():
    info = get_dataset_info(img_shape=(5, 5))
    module = _sfno_same_grid_selector().build(
        n_in_channels=2, n_out_channels=3, in_dataset_info=info, out_dataset_info=info
    )
    out = module(torch.zeros(1, 2, 5, 5))
    assert out.shape == (1, 3, 5, 5)


def test_same_grid_rejects_mismatched_grids():
    with pytest.raises(ValueError, match="matching input and output"):
        _sfno_same_grid_selector().build(
            n_in_channels=1,
            n_out_channels=1,
            in_dataset_info=get_dataset_info(img_shape=(8, 16)),
            out_dataset_info=get_dataset_info(img_shape=(4, 8)),
        )


@pytest.mark.parametrize(
    "in_shape, out_shape",
    [((8, 16), (4, 8)), ((4, 8), (8, 16))],
    ids=["downsample", "upsample"],
)
def test_interpolate_maps_channels_and_resizes(in_shape, out_shape):
    selector = TransformSelector(type="interpolate", config={})
    module = selector.build(
        n_in_channels=3,
        n_out_channels=2,
        in_dataset_info=get_dataset_info(img_shape=in_shape),
        out_dataset_info=get_dataset_info(img_shape=out_shape),
    )
    out = module(torch.randn(2, 3, *in_shape))
    assert out.shape == (2, 2, *out_shape)


def test_unregistered_type_raises():
    with pytest.raises(KeyError):
        TransformSelector(type="not_a_transform", config={})
