"""Every SFNO implementation tags its dense spectral filter weights as
SpectralFilterParameter with the axes its contraction uses, so Muon
orthogonalizes the right matrices."""

import importlib

import pytest
import torch
import torch.nn as nn

from fme.ace.models.makani.spectral_convolution import SpectralConv as MakaniConv
from fme.ace.models.modulus.s2convolutions import (
    SpectralAttentionS2 as ModulusAttention,
)
from fme.ace.models.modulus.s2convolutions import SpectralConvS2 as ModulusConv
from fme.core.gridded_ops import LatLonOperations
from fme.core.models.conditional_sfno.makani.spectral_convolution import (
    SpectralConv as ConditionalMakaniConv,
)
from fme.core.models.conditional_sfno.s2convolutions import (
    SpectralAttentionS2 as ConditionalAttention,
)
from fme.core.models.conditional_sfno.s2convolutions import (
    SpectralConvS2 as ConditionalConv,
)
from fme.core.spectral_filter_parameter import SpectralFilterParameter

# imported dynamically because mypy excludes the vendored makani_fcn3 package
FCN3Conv = importlib.import_module(
    "fme.ace.models.makani_fcn3.models.common.spectral_convolution"
).SpectralConv

IN, OUT = 4, 6


def _transforms():
    operations = LatLonOperations(area_weights=torch.ones(8, 14), grid="legendre-gauss")
    return operations.get_real_sht(), operations.get_real_isht()


def _channel_sizes(param: SpectralFilterParameter) -> tuple[int, int]:
    shape = param.shape if param.is_complex() else param.shape[:-1]
    return shape[param.in_dim], shape[param.out_dim]


def _tagged(module: nn.Module) -> dict[str, SpectralFilterParameter]:
    return {
        name: p
        for name, p in module.named_parameters()
        if isinstance(p, SpectralFilterParameter)
    }


@pytest.mark.parametrize("operator_type", ["diagonal", "dhconv"])
@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (ModulusConv, dict(use_tensorly=False)),
        (MakaniConv, {}),
        (ConditionalMakaniConv, dict(num_groups=2)),
        (FCN3Conv, dict(num_groups=2)),
    ],
)
def test_dense_spectral_conv_weight_is_tagged(cls, kwargs, operator_type):
    sht, isht = _transforms()
    conv = cls(sht, isht, IN, OUT, operator_type=operator_type, **kwargs)
    tagged = _tagged(conv)
    assert list(tagged) == ["weight"]
    groups = kwargs.get("num_groups", 1)
    assert _channel_sizes(tagged["weight"]) == (IN // groups, OUT // groups)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (ModulusConv, dict(use_tensorly=False)),
        (MakaniConv, {}),
        (ConditionalMakaniConv, {}),
        (FCN3Conv, {}),
    ],
)
def test_separable_spectral_conv_weight_is_not_tagged(cls, kwargs):
    """A separable filter is a per-channel scale, not a matrix."""
    sht, isht = _transforms()
    conv = cls(sht, isht, IN, IN, separable=True, **kwargs)
    assert _tagged(conv) == {}


@pytest.mark.parametrize("lora_rank", [0, 3])
def test_conditional_spectral_conv_weights_are_tagged(lora_rank: int):
    sht, isht = _transforms()
    channels = 8
    conv = ConditionalConv(
        sht, isht, channels, channels, num_groups=2, lora_rank=lora_rank
    )
    tagged = _tagged(conv)
    group_channels = channels // 2
    expected = {"weight": (group_channels, group_channels)}
    if lora_rank > 0:
        expected["lora_A"] = (group_channels, lora_rank)
        expected["lora_B"] = (lora_rank, group_channels)
    assert {name: _channel_sizes(p) for name, p in tagged.items()} == expected


def test_conditional_spectral_conv_weight_dims_match_contraction():
    """A weight that maps input channel 1 to output channel 0 in every mode
    moves the signal from channel 1 to channel 0, so in_dim and out_dim are
    not swapped."""
    sht, isht = _transforms()
    conv = ConditionalConv(sht, isht, 2, 2)
    with torch.no_grad():
        conv.weight.zero_()
        index: list[slice | int] = [slice(None)] * conv.weight.ndim
        index[conv.weight.out_dim] = 0
        index[conv.weight.in_dim] = 1
        index[-1] = 0
        conv.weight[tuple(index)] = 1.0
    x = torch.zeros(1, 2, 8, 14)
    x[:, 1] = torch.randn(8, 14)
    y = conv(x)
    y = y[0] if isinstance(y, tuple) else y
    assert y[:, 0].abs().max() > 0
    assert y[:, 1].abs().max() == 0


@pytest.mark.parametrize("cls", [ModulusAttention, ConditionalAttention])
@pytest.mark.parametrize("operator_type", ["diagonal", "l-dependant"])
def test_spectral_attention_weights_are_tagged(cls, operator_type):
    sht, isht = _transforms()
    embed_dim, factor = 4, 2
    attention = cls(
        sht,
        isht,
        embed_dim,
        operator_type=operator_type,
        hidden_size_factor=factor,
        spectral_layers=2,
    )
    hidden = embed_dim * factor
    sizes = {name: _channel_sizes(p) for name, p in _tagged(attention).items()}
    assert sizes == {
        "w.0": (embed_dim, hidden),
        "w.1": (hidden, hidden),
        "wout": (hidden, embed_dim),
    }
