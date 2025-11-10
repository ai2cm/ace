import pytest
import torch
from torch import nn

from fme.core.ema import EMATracker


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10))

    def forward(self, x):
        return self.weight * x


def test_ema_initializes_with_model_parameters():
    model = ExampleModel()
    ema = EMATracker(model, decay=0.9999)
    model_initial_params = [param.data.clone() for param in model.parameters()]
    with ema.applied_params(model):
        for param, model_param in zip(model.parameters(), model_initial_params):
            torch.testing.assert_close(param.data.clone(), model_param)


@pytest.mark.parametrize("faster_decay_at_start", [True, False])
@pytest.mark.parametrize("decay", [0.9999])
def test_ema_updates_parameters(decay: float, faster_decay_at_start: bool):
    model = ExampleModel()
    model.weight.data.fill_(1.0)
    ema = EMATracker(model, decay=decay, faster_decay_at_start=faster_decay_at_start)
    with ema.applied_params(model):
        torch.testing.assert_close(
            model.weight.data.clone(), torch.full_like(model.weight.data, 1.0)
        )
    model.weight.data.fill_(0.0)
    ema(model)
    torch.testing.assert_close(
        model.weight.data.clone(), torch.full_like(model.weight.data, 0.0)
    )
    if faster_decay_at_start:
        target_decay = min(decay, 2.0 / 11.0)
    else:
        target_decay = decay
    with ema.applied_params(model):
        torch.testing.assert_close(
            model.weight.data.clone(), torch.full_like(model.weight.data, target_decay)
        )


def compare_states(state1: dict, state2: dict):
    assert state1.keys() == state2.keys()
    for key in state1.keys():
        if key == "ema_params":
            torch.testing.assert_close(state1[key], state2[key])
        else:
            assert state1[key] == state2[key]


@pytest.mark.parametrize("faster_decay_at_start", [True, False])
@pytest.mark.parametrize("decay", [0.9999])
def test_ema_transferred_state_updates_parameters(
    decay: float, faster_decay_at_start: bool
):
    model = ExampleModel()
    model.weight.data.fill_(1.0)
    ema = EMATracker(model, decay=decay, faster_decay_at_start=faster_decay_at_start)
    with ema.applied_params(model):
        torch.testing.assert_close(
            model.weight.data.clone(), torch.full_like(model.weight.data, 1.0)
        )
    model.weight.data.fill_(0.0)
    ema(model)
    torch.testing.assert_close(
        model.weight.data.clone(), torch.full_like(model.weight.data, 0.0)
    )
    second_ema = EMATracker.from_state(ema.get_state(), model)
    compare_states(ema.get_state(), second_ema.get_state())
    if faster_decay_at_start:
        target_decay = min(decay, 2.0 / 11.0)
    else:
        target_decay = decay
    with second_ema.applied_params(model):
        torch.testing.assert_close(
            model.weight.data.clone(), torch.full_like(model.weight.data, target_decay)
        )
