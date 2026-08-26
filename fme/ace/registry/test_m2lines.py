import pytest
import torch

from fme.ace.models.ocean.m2lines.layers import ConvNeXtBlock, NoiseConditioning
from fme.ace.models.ocean.m2lines.samudra import Samudra
from fme.ace.registry.m2lines import NoiseConditionedSamudraBuilder, SamudraBuilder
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


@pytest.mark.parametrize("noise_injection", ["bottleneck", "all_blocks"])
def test_noise_conditioned_samudra_builder(noise_injection):
    builder = NoiseConditionedSamudraBuilder(
        ch_width=[8, 12],
        dilation=[1, 2],
        n_layers=[1, 1],
        noise_embed_dim=6,
        noise_injection=noise_injection,
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
    assert n_conditioned == (1 if noise_injection == "bottleneck" else len(blocks))
    for block in blocks:
        for module in block.noise_conditioning.values():
            assert module.W_scale.in_channels == 6

    # the module is called the way the stepper calls it: input tensor only
    output = model(torch.randn(2, 5, *img_shape))
    assert output.shape == (2, 3, *img_shape)


def test_noise_conditioned_samudra_is_registered():
    selector = ModuleSelector(
        type="NoiseConditionedSamudra",
        config={"ch_width": [8], "dilation": [1], "n_layers": [1]},
    )
    module = selector.build(4, 2, DatasetInfo(img_shape=(16, 32)))
    output = module(torch.randn(2, 4, 16, 32))
    assert output.shape == (2, 2, 16, 32)
    # defaults are captured in the serialized config
    assert selector.config["noise_embed_dim"] == 32
    assert selector.config["noise_injection"] == "bottleneck"


def test_noise_conditioned_samudra_builder_validation():
    with pytest.raises(ValueError, match="noise_embed_dim must be positive"):
        NoiseConditionedSamudraBuilder(noise_embed_dim=0)
    with pytest.raises(ValueError, match="norm_kwargs should not have num_features"):
        NoiseConditionedSamudraBuilder(norm_kwargs={"num_features": 10})


def test_noise_conditioned_samudra_rejects_labels():
    builder = NoiseConditionedSamudraBuilder(ch_width=[8], dilation=[1], n_layers=[1])
    dataset_info = DatasetInfo(img_shape=(16, 32), all_labels={"a", "b"})
    with pytest.raises(ValueError, match="does not support labels"):
        builder.build(4, 2, dataset_info)


def test_noise_conditioned_samudra_draws_independent_noise_per_sample():
    """Ensemble training folds the members into the batch dimension, so
    identical inputs stacked along the batch must come out as distinct members.
    """
    torch.manual_seed(0)
    img_shape = (16, 32)
    builder = NoiseConditionedSamudraBuilder(
        ch_width=[8, 12],
        dilation=[1, 2],
        n_layers=[1, 1],
        noise_embed_dim=6,
        noise_injection="all_blocks",
    )
    model = builder.build(4, 2, DatasetInfo(img_shape=img_shape))
    for module in model.modules():
        if isinstance(module, NoiseConditioning):
            torch.nn.init.normal_(module.W_scale.weight, std=0.1)
            torch.nn.init.normal_(module.W_bias.weight, std=0.1)
    x = torch.randn(1, 4, *img_shape).expand(3, 4, *img_shape)
    with torch.no_grad():
        output = model(x)
    assert not torch.allclose(output[0], output[1])
    assert not torch.allclose(output[1], output[2])
