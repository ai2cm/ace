import copy
import io

import pytest
import torch
import torch.nn as nn

import fme
from fme.core.spectral_filter_parameter import SpectralFilterParameter


def _param() -> SpectralFilterParameter:
    return SpectralFilterParameter(torch.randn(2, 3, 4, 5, 2), in_dim=3, out_dim=2)


def _assert_same(a: SpectralFilterParameter, b: SpectralFilterParameter):
    assert type(b) is SpectralFilterParameter
    assert (b.in_dim, b.out_dim) == (a.in_dim, a.out_dim)
    assert b.requires_grad == a.requires_grad
    torch.testing.assert_close(b.detach().cpu(), a.detach().cpu())


def test_deepcopy_keeps_type_and_dims():
    p = _param()
    copied = copy.deepcopy(p)
    _assert_same(p, copied)
    assert copied.data_ptr() != p.data_ptr()


def test_deepcopy_of_module_keeps_type_and_dims():
    module = nn.Module()
    module.weight = _param()
    copied = copy.deepcopy(module)
    _assert_same(module.weight, copied.weight)


def test_pickle_round_trip_keeps_type_and_dims():
    p = _param()
    p.requires_grad_(False)
    buffer = io.BytesIO()
    torch.save(p, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    _assert_same(p, loaded)


def test_to_device_keeps_type_and_dims():
    module = nn.Module()
    module.weight = _param()
    module.to(fme.get_device())
    assert isinstance(module.weight, SpectralFilterParameter)
    assert (module.weight.in_dim, module.weight.out_dim) == (3, 2)


def test_state_dict_holds_plain_tensors():
    """Checkpoints store plain tensors, so they load into untagged models and
    with weights_only=True."""
    module = nn.Module()
    module.weight = _param()
    state = module.state_dict()
    assert type(state["weight"]) is torch.Tensor
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    torch.load(buffer, weights_only=True)


def test_parameter_list_keeps_type():
    params = nn.ParameterList([_param(), _param()])
    assert all(isinstance(p, SpectralFilterParameter) for p in params)


def test_negative_dims_are_normalized():
    p = SpectralFilterParameter(torch.randn(2, 4, 5, 2), in_dim=-1, out_dim=-2)
    assert (p.in_dim, p.out_dim) == (2, 1)


def test_complex_data():
    p = SpectralFilterParameter(
        torch.randn(4, 5, dtype=torch.complex64), in_dim=0, out_dim=1
    )
    assert (p.in_dim, p.out_dim) == (0, 1)


@pytest.mark.parametrize(
    "data, in_dim, out_dim, match",
    [
        (torch.randn(4, 5), 0, 1, "last of size 2"),
        (torch.randn(4, 5, 3), 0, 1, "last of size 2"),
        (torch.randn(4, dtype=torch.complex64), 0, 0, "at least 2"),
        (torch.randn(4, 5, 2), 0, 2, "out of range"),
        (torch.randn(4, 5, 2), 1, -1, "must differ"),
    ],
)
def test_invalid_layout_raises(data, in_dim, out_dim, match):
    with pytest.raises(ValueError, match=match):
        SpectralFilterParameter(data, in_dim=in_dim, out_dim=out_dim)
