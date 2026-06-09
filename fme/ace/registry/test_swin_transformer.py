import dataclasses
from typing import Any

import pytest
import torch

import fme
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.ace.registry.swin_transformer import (
    NoiseConditionedSwinTransformerBuilder,
    SwinTransformerBuilder,
)
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


def _nc_builder(**kwargs: Any) -> NoiseConditionedSwinTransformerBuilder:
    defaults: dict[str, Any] = dict(
        embed_dim=32,
        num_heads=[2, 4, 4, 2],
        window_size=[4, 4],
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        noise_embed_dim=8,
    )
    defaults.update(kwargs)
    return NoiseConditionedSwinTransformerBuilder(**defaults)


def _builder(**kwargs: Any) -> SwinTransformerBuilder:
    defaults: dict[str, Any] = dict(
        embed_dim=32,
        num_heads=[2, 4, 4, 2],
        window_size=[4, 4],
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


def test_swin_transformer_builds_with_img_shape_only_dataset_info():
    n_in, n_out = 5, 3
    dataset_info = DatasetInfo(img_shape=IMG_SHAPE)
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
    net = getattr(module.torch_module, "module")
    assert net.embed_dim_labels == len(all_labels)


def test_swin_transformer_unconditional_ignores_dataset_labels():
    n_in, n_out = 5, 3
    all_labels = {"label_a", "label_b"}
    dataset_info = _get_dataset_info(all_labels=all_labels)
    selector = ModuleSelector(
        type="SwinTransformer",
        config=dataclasses.asdict(_builder()),
    )
    module = selector.build(
        n_in_channels=n_in, n_out_channels=n_out, dataset_info=dataset_info
    ).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)
    net = getattr(module.torch_module, "module")
    assert net.embed_dim_labels == 0


def test_swin_transformer_unconditional_rejects_label_embed_dim():
    all_labels = {"label_a", "label_b"}
    dataset_info = _get_dataset_info(all_labels=all_labels)
    selector = ModuleSelector(
        type="SwinTransformer",
        config=dataclasses.asdict(_builder(embed_dim_labels=2)),
    )
    with pytest.raises(ValueError, match="conditional=True"):
        selector.build(n_in_channels=5, n_out_channels=3, dataset_info=dataset_info)


def test_nc_swin_transformer_is_registered():
    assert "NoiseConditionedSwinTransformer" in ModuleSelector.get_available_types()


def test_nc_swin_transformer_returns_noise_conditioned_model():
    """Builder returns a NoiseConditionedModel wrapping the Swin net."""
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    module = _nc_builder().build(n_in, n_out, dataset_info)
    assert isinstance(module, NoiseConditionedModel)


def test_nc_swin_transformer_via_selector():
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    selector = ModuleSelector(
        type="NoiseConditionedSwinTransformer",
        config=dataclasses.asdict(_nc_builder()),
    )
    module = selector.build(
        n_in_channels=n_in, n_out_channels=n_out, dataset_info=dataset_info
    ).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_nc_swin_transformer_builds_with_img_shape_only_dataset_info():
    n_in, n_out = 5, 3
    dataset_info = DatasetInfo(img_shape=IMG_SHAPE)
    module = _nc_builder().build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_nc_swin_transformer_unconditional_ignores_dataset_labels():
    n_in, n_out = 5, 3
    all_labels = {"label_a", "label_b"}
    dataset_info = _get_dataset_info(all_labels=all_labels)
    selector = ModuleSelector(
        type="NoiseConditionedSwinTransformer",
        config=dataclasses.asdict(_nc_builder()),
    )
    module = selector.build(
        n_in_channels=n_in, n_out_channels=n_out, dataset_info=dataset_info
    ).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)
    net = getattr(module.torch_module, "conditional_model")
    assert net.embed_dim_labels == 0


def test_nc_swin_transformer_unconditional_rejects_label_embed_dim():
    all_labels = {"label_a", "label_b"}
    dataset_info = _get_dataset_info(all_labels=all_labels)
    selector = ModuleSelector(
        type="NoiseConditionedSwinTransformer",
        config=dataclasses.asdict(_nc_builder(label_embed_dim=2)),
    )
    with pytest.raises(ValueError, match="conditional=True"):
        selector.build(n_in_channels=5, n_out_channels=3, dataset_info=dataset_info)


def test_swin_transformer_cpb_mlp_exists():
    """Built model has cpb_mlp and no relative_position_bias_table."""
    n_in, n_out = 5, 3
    dataset_info = _get_dataset_info()
    module = _builder().build(n_in, n_out, dataset_info)
    # _ContextWrappedModule wraps the SwinTransformerNet
    net = module.module  # type: ignore[attr-defined]
    block = net.layer1.blocks[0]
    assert hasattr(block.attn, "cpb_mlp"), "cpb_mlp missing"
    assert not hasattr(
        block.attn, "relative_position_bias_table"
    ), "relative_position_bias_table should not exist"


def test_nc_swin_transformer_noise_divergence():
    """Two forwards on identical input diverge after one optimizer step.

    CLN's zero-init noise convs make the freshly-built model noise-independent.
    After one optimizer step they move off zero and different resampled noise
    yields distinct outputs.
    """
    n_in, n_out = 4, 2
    dataset_info = _get_dataset_info()
    module = _nc_builder().build(n_in, n_out, dataset_info).to(fme.get_device())
    module.train()
    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)

    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())

    # At init: noise-independent (zero-init CLN convs → scale=1, bias=0).
    with torch.no_grad():
        out1 = module(x)
        out2 = module(x)
    assert torch.allclose(out1, out2), "Expected noise-independence at init"

    # One optimizer step pushes noise convs off zero.
    out = module(x)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    # After step: two independent forward passes now differ.
    with torch.no_grad():
        out1 = module(x)
        out2 = module(x)
    assert not torch.allclose(out1, out2), "Expected noise-dependence after step"


_PAD_CONF = {"activate": True, "mode": "earth", "pad_lat": [2, 1], "pad_lon": [2, 2]}


def test_swin_transformer_earth_padding():
    module = (
        _builder(padding_conf=_PAD_CONF)
        .build(5, 3, _get_dataset_info())
        .to(fme.get_device())
    )
    x = torch.randn(2, 5, *IMG_SHAPE, device=fme.get_device())
    assert module(x).shape == (2, 3, *IMG_SHAPE)


def test_nc_swin_transformer_earth_padding():
    module = (
        _nc_builder(padding_conf=_PAD_CONF)
        .build(5, 3, _get_dataset_info())
        .to(fme.get_device())
    )
    x = torch.randn(2, 5, *IMG_SHAPE, device=fme.get_device())
    assert module(x).shape == (2, 3, *IMG_SHAPE)
