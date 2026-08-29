import pytest
import torch

from fme.ace.models.ocean.m2lines.layers import ConvNeXtBlock, MultiResolutionFiLM
from fme.ace.models.ocean.m2lines.samudra import Samudra
from fme.ace.registry.m2lines import SamudraBuilder
from fme.ace.registry.registry import ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo


def test_samudra_builder():
    builder = SamudraBuilder()
    # assuming 5 input (3 prognostic + 2 forcing) and 3 output vars (prognostic)
    dataset_info = DatasetInfo(img_shape=(16, 32))
    model = builder.build(5, 3, dataset_info)
    assert model.layers[0].convblock[0].in_channels == 5
    assert model.layers[-1].out_channels == 3

    with pytest.raises(ValueError, match="norm_kwargs should not have num_features"):
        _ = SamudraBuilder(norm_kwargs={"num_features": 10})

    with pytest.raises(
        ValueError, match="norm_kwargs should not have normalized_shape"
    ):
        _ = SamudraBuilder(norm_kwargs={"normalized_shape": (3, 3)})


@pytest.mark.parametrize("conditioned_blocks", ["bottleneck", "all_blocks"])
def test_noise_conditioned_samudra_builder(conditioned_blocks):
    builder = SamudraBuilder(
        ch_width=[8, 12],
        dilation=[1, 2],
        n_layers=[1, 1],
        noise_embed_dim=6,
        conditioned_blocks=conditioned_blocks,
    )
    img_shape = (16, 32)
    model = builder.build(5, 3, DatasetInfo(img_shape=img_shape))
    assert isinstance(model, NoiseConditionedModel)
    assert isinstance(model.conditional_model, Samudra)
    assert model.embed_dim == 6

    blocks = [
        layer
        for layer in model.conditional_model.layers
        if isinstance(layer, ConvNeXtBlock)
    ]
    n_conditioned = sum(len(block.noise_conditioning) > 0 for block in blocks)
    assert n_conditioned == (1 if conditioned_blocks == "bottleneck" else len(blocks))
    for block in blocks:
        for module in block.noise_conditioning.values():
            assert module.W_scale.in_channels == 6

    # the module is called the way the stepper calls it: input tensor only
    output = model(torch.randn(2, 5, *img_shape))
    assert output.shape == (2, 3, *img_shape)


def test_samudra_registry_entry_covers_both_variants():
    """One registry entry builds the deterministic and the noise-conditioned
    network, so the two configurations cannot drift apart."""
    plain = ModuleSelector(
        type="Samudra",
        config={"ch_width": [8], "dilation": [1], "n_layers": [1]},
    )
    module = plain.build(4, 2, DatasetInfo(img_shape=(16, 32)))
    assert isinstance(module.torch_module, Samudra)
    assert module(torch.randn(2, 4, 16, 32)).shape == (2, 2, 16, 32)
    # defaults are captured in the serialized config
    assert plain.config["noise_embed_dim"] == 0
    assert plain.config["conditioned_blocks"] is None

    conditioned = ModuleSelector(
        type="Samudra",
        config={
            "ch_width": [8],
            "dilation": [1],
            "n_layers": [1],
            "noise_embed_dim": 6,
            "conditioned_blocks": "bottleneck",
        },
    )
    module = conditioned.build(4, 2, DatasetInfo(img_shape=(16, 32)))
    assert isinstance(module.torch_module, NoiseConditionedModel)
    assert module(torch.randn(2, 4, 16, 32)).shape == (2, 2, 16, 32)


def test_samudra_builder_validation():
    with pytest.raises(ValueError, match="requires conditioned_blocks to be set"):
        SamudraBuilder(noise_embed_dim=6)
    with pytest.raises(ValueError, match="requires a non-zero noise_embed_dim"):
        SamudraBuilder(conditioned_blocks="bottleneck")
    with pytest.raises(ValueError, match="must not be negative"):
        SamudraBuilder(noise_embed_dim=-1)
    with pytest.raises(ValueError, match="norm_kwargs should not have num_features"):
        SamudraBuilder(norm_kwargs={"num_features": 10})


def test_noise_conditioned_samudra_rejects_labels():
    builder = SamudraBuilder(
        ch_width=[8],
        dilation=[1],
        n_layers=[1],
        noise_embed_dim=6,
        conditioned_blocks="bottleneck",
    )
    dataset_info = DatasetInfo(img_shape=(16, 32), all_labels={"a", "b"})
    with pytest.raises(ValueError, match="does not support labels"):
        builder.build(4, 2, dataset_info)


def test_noise_conditioned_samudra_draws_independent_noise_per_sample():
    """Ensemble training folds the members into the batch dimension, so
    identical inputs stacked along the batch must come out as distinct members.
    """
    torch.manual_seed(0)
    img_shape = (16, 32)
    builder = SamudraBuilder(
        ch_width=[8, 12],
        dilation=[1, 2],
        n_layers=[1, 1],
        noise_embed_dim=6,
        conditioned_blocks="all_blocks",
    )
    model = builder.build(4, 2, DatasetInfo(img_shape=img_shape))
    for module in model.modules():
        if isinstance(module, MultiResolutionFiLM):
            torch.nn.init.normal_(module.W_scale.weight, std=0.1)
            torch.nn.init.normal_(module.W_bias.weight, std=0.1)
    x = torch.randn(1, 4, *img_shape).expand(3, 4, *img_shape)
    with torch.no_grad():
        output = model(x)
    assert not torch.allclose(output[0], output[1])
    assert not torch.allclose(output[1], output[2])
