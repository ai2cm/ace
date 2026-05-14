import torch

from fme.ace.aggregator.residual_spectrum import temporal_diffs
from fme.core.device import get_device


def test_temporal_diffs_basic():
    device = get_device()
    data = {
        "a": torch.tensor([[1.0, 3.0, 6.0], [2.0, 5.0, 9.0]], device=device)
        .unsqueeze(-1)
        .unsqueeze(-1),
    }
    diffs = temporal_diffs(data)
    assert set(diffs.keys()) == {"a"}
    expected = (
        torch.tensor([[2.0, 3.0], [3.0, 4.0]], device=device)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    torch.testing.assert_close(diffs["a"], expected)


def test_temporal_diffs_drops_single_timestep():
    device = get_device()
    data = {"a": torch.randn(2, 1, 3, 3, device=device)}
    assert temporal_diffs(data) == {}


def test_temporal_diffs_multiple_variables():
    device = get_device()
    data = {
        "u": torch.randn(4, 5, 8, 8, device=device),
        "v": torch.randn(4, 5, 8, 8, device=device),
    }
    diffs = temporal_diffs(data)
    assert set(diffs.keys()) == {"u", "v"}
    assert diffs["u"].shape == (4, 4, 8, 8)
    assert diffs["v"].shape == (4, 4, 8, 8)


def test_temporal_diffs_preserves_sign():
    device = get_device()
    t = torch.tensor([[5.0, 2.0]], device=device).unsqueeze(-1).unsqueeze(-1)
    diffs = temporal_diffs({"x": t})
    torch.testing.assert_close(
        diffs["x"],
        torch.tensor([[-3.0]], device=device).unsqueeze(-1).unsqueeze(-1),
    )
