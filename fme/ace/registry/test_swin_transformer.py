import dataclasses
from typing import Any

import torch

import fme
from fme.ace.registry.swin_transformer import SwinTransformerBuilder
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.labels import BatchLabels
from fme.core.registry import ModuleSelector

IMG_SHAPE = (16, 32)


def _get_dataset_info(all_labels: set[str] | None = None) -> DatasetInfo:
    device = fme.get_device()
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(IMG_SHAPE[0], device=device),
            lon=torch.zeros(IMG_SHAPE[1], device=device),
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=device),
            bk=torch.arange(7, device=device),
        ),
        all_labels=all_labels,
    )


def _builder(**kwargs: Any) -> SwinTransformerBuilder:
    defaults: dict[str, Any] = dict(
        embed_dim=32,
        num_heads=(2, 4, 4, 2),
        window_size=(4, 4),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
    )
    defaults.update(kwargs)
    return SwinTransformerBuilder(**defaults)


def test_swin_transformer_is_registered():
    assert "SwinTransformer" in ModuleSelector.get_available_types()


def test_swin_transformer_build_and_forward():
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    module = _builder().build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_swin_transformer_via_selector():
    selector = ModuleSelector(
        type="SwinTransformer",
        config=dataclasses.asdict(_builder()),
    )
    dataset_info = _get_dataset_info()
    module = selector.build(
        n_in_channels=5, n_out_channels=3, dataset_info=dataset_info
    ).to(fme.get_device())
    x = torch.randn(2, 5, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, 3, *IMG_SHAPE)


def test_swin_transformer_conditional_with_labels():
    n_in, n_out = 5, 3
    all_labels = {"label_a", "label_b"}
    dataset_info = _get_dataset_info(all_labels=all_labels)
    selector = ModuleSelector(
        type="SwinTransformer",
        conditional=True,
        config=dataclasses.asdict(_builder()),
    )
    module = selector.build(
        n_in_channels=n_in, n_out_channels=n_out, dataset_info=dataset_info
    ).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    labels = BatchLabels.new_from_set(all_labels, n_samples=2, device=fme.get_device())
    out = module(x, labels=labels)
    assert out.shape == (2, n_out, *IMG_SHAPE)
