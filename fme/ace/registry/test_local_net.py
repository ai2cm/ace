import dataclasses

import pytest
import torch

import fme
from fme.ace.registry.local_net import AnkurLocalNetBuilder, LocalNetBuilder
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.labels import BatchLabels
from fme.core.models.conditional_sfno.localnet import BlockType
from fme.core.registry import ModuleSelector

IMG_SHAPE = (9, 18)


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


def test_ankur_local_net_is_registered():
    assert "AnkurLocalNet" in ModuleSelector.get_available_types()


def test_local_net_is_registered():
    assert "LocalNet" in ModuleSelector.get_available_types()


@pytest.mark.parametrize("use_disco_encoder", [True, False])
@pytest.mark.parametrize("pos_embed", [True, False])
def test_ankur_local_net_build_and_forward(use_disco_encoder: bool, pos_embed: bool):
    n_in, n_out = 3, 2
    dataset_info = _get_dataset_info()
    builder = AnkurLocalNetBuilder(
        embed_dim=16,
        use_disco_encoder=use_disco_encoder,
        pos_embed=pos_embed,
    )
    module = builder.build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_ankur_local_net_via_selector():
    selector = ModuleSelector(
        type="AnkurLocalNet",
        config=dataclasses.asdict(AnkurLocalNetBuilder(embed_dim=16)),
    )
    dataset_info = _get_dataset_info()
    module = selector.build(
        n_in_channels=3, n_out_channels=2, dataset_info=dataset_info
    )
    module = module.to(fme.get_device())
    x = torch.randn(2, 3, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, 2, *IMG_SHAPE)


@pytest.mark.parametrize(
    "block_types",
    [
        ["disco", "disco"],
        ["conv1x1", "conv1x1"],
        ["disco", "conv1x1"],
    ],
)
def test_local_net_build_and_forward(block_types: list[BlockType]):
    n_in, n_out = 3, 2
    dataset_info = _get_dataset_info()
    builder = LocalNetBuilder(
        embed_dim=16,
        block_types=block_types,
    )
    module = builder.build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_local_net_via_selector():
    selector = ModuleSelector(
        type="LocalNet",
        config=dataclasses.asdict(
            LocalNetBuilder(embed_dim=16, block_types=["disco", "disco"])
        ),
    )
    dataset_info = _get_dataset_info()
    module = selector.build(
        n_in_channels=3, n_out_channels=2, dataset_info=dataset_info
    )
    module = module.to(fme.get_device())
    x = torch.randn(2, 3, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, 2, *IMG_SHAPE)


def test_ankur_local_net_backward():
    n_in, n_out = 3, 2
    dataset_info = _get_dataset_info()
    builder = AnkurLocalNetBuilder(embed_dim=16)
    module = builder.build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    out.sum().backward()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_local_net_backward():
    n_in, n_out = 3, 2
    dataset_info = _get_dataset_info()
    builder = LocalNetBuilder(embed_dim=16, block_types=["disco", "disco"])
    module = builder.build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    out.sum().backward()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_local_net_noise_produces_stochastic_output():
    """Noise conditioning should produce different outputs after training."""
    torch.manual_seed(0)
    n_in, n_out = 3, 2
    dataset_info = _get_dataset_info()
    builder = LocalNetBuilder(
        embed_dim=16,
        noise_embed_dim=8,
        block_types=["disco", "disco"],
        affine_norms=True,
    )
    module = builder.build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    # At init, noise scale/bias weights are zero so noise has no effect.
    # A training step makes them nonzero, enabling stochastic output.
    loss = module(x).sum()
    loss.backward()
    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
    optimizer.step()
    with torch.no_grad():
        out1 = module(x)
        out2 = module(x)
    assert not torch.allclose(out1, out2)


def test_local_net_isotropic_noise():
    n_in, n_out = 3, 2
    dataset_info = _get_dataset_info()
    builder = LocalNetBuilder(
        embed_dim=16,
        noise_embed_dim=8,
        noise_type="isotropic",
        block_types=["disco", "disco"],
    )
    module = builder.build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_local_net_with_context_pos_embed():
    n_in, n_out = 3, 2
    dataset_info = _get_dataset_info()
    builder = LocalNetBuilder(
        embed_dim=16,
        noise_embed_dim=8,
        context_pos_embed_dim=4,
        block_types=["disco", "disco"],
    )
    module = builder.build(n_in, n_out, dataset_info).to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    out = module(x)
    assert out.shape == (2, n_out, *IMG_SHAPE)


def test_local_net_conditional_with_labels():
    """LocalNet with conditional=True should accept and use labels."""
    n_in, n_out = 3, 2
    all_labels = {"label_a", "label_b"}
    dataset_info = _get_dataset_info(all_labels=all_labels)
    selector = ModuleSelector(
        type="LocalNet",
        conditional=True,
        config=dataclasses.asdict(
            LocalNetBuilder(
                embed_dim=16,
                noise_embed_dim=8,
                block_types=["disco", "disco"],
                affine_norms=True,
            )
        ),
    )
    module = selector.build(
        n_in_channels=n_in, n_out_channels=n_out, dataset_info=dataset_info
    )
    module = module.to(fme.get_device())
    x = torch.randn(2, n_in, *IMG_SHAPE, device=fme.get_device())
    labels = BatchLabels.new_from_set(all_labels, n_samples=2, device=fme.get_device())
    out = module(x, labels=labels)
    assert out.shape == (2, n_out, *IMG_SHAPE)
