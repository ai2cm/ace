import pytest
import torch

from fme import get_device

from .distributed import pad_tensor_at_end, unpad_tensor_at_end


@pytest.mark.parametrize(
    ["padding", "fill_value"],
    [
        pytest.param([0, 0, 0], None, id="no_padding"),
        pytest.param([1, 1, 1], 0.0, id="padding_1"),
        pytest.param([1, 1, 1], 1.0, id="padding_1_fill_one"),
    ],
)
def test_pad_tensor_at_end(padding, fill_value):
    tensor = torch.ones(2, 3, 4)
    padded_tensor = pad_tensor_at_end(tensor, padding, fill_value)
    assert padded_tensor.size() == (2 + padding[0], 3 + padding[1], 4 + padding[2])
    for dim, pad in enumerate(padding):
        if pad > 0:
            assert torch.allclose(
                padded_tensor.select(dim=dim, index=padded_tensor.size(dim) - 1),
                torch.tensor(fill_value),
            )


@pytest.mark.parametrize(
    ["padding"],
    [
        pytest.param([0, 0, 0], id="no_padding"),
        pytest.param([1, 1, 1], id="padding_1"),
    ],
)
def test_pad_unpad_rountrip(padding):
    tensor = torch.ones(2, 3, 4, device=get_device())
    padded_tensor = pad_tensor_at_end(tensor, padding)
    unpadded_tensor = unpad_tensor_at_end(padded_tensor, padding)
    assert unpadded_tensor.size() == tensor.size()
    assert torch.allclose(unpadded_tensor, tensor)
