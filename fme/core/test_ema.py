import logging
import os

import pytest
import torch
from torch import nn

from fme.core.device import get_device
from fme.core.ema import EMAConfig, EMATracker, load_ema_params_if_available


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, device=get_device()))

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


def test_from_state_places_tensors_on_device():
    model = ExampleModel()
    ema = EMATracker(model, decay=0.999)
    state = ema.get_state()
    # move state tensors to cpu to simulate loading from checkpoint
    state["decay"] = state["decay"].cpu()
    state["num_updates"] = state["num_updates"].cpu()
    state["ema_params"] = {k: v.cpu() for k, v in state["ema_params"].items()}
    restored = EMATracker.from_state(state, model)
    device = get_device()
    assert restored.num_updates.device == device
    assert restored.decay.device == device
    for param in restored._ema_params.values():
        assert param.device == device


def test_from_state_decouples_memory():
    model = ExampleModel()
    ema = EMATracker(model, decay=0.999)
    state = ema.get_state()
    restored = EMATracker.from_state(state, model)
    # mutating state should not affect restored object
    state["num_updates"].fill_(999)
    assert restored.num_updates.item() != 999
    for name, param in state["ema_params"].items():
        param.fill_(float("nan"))
        assert not torch.isnan(restored._ema_params[name]).any()


def test_load_ema_state_for_finetuning():
    """load_ema_state_for_finetuning restores ema_params and num_updates
    from a saved state while preserving the current tracker's decay."""
    model = ExampleModel()
    model.weight.data.fill_(1.0)
    source_decay = 0.99
    ema = EMATracker(model, decay=source_decay, faster_decay_at_start=True)
    model.weight.data.fill_(0.0)
    for _ in range(5):
        ema(model)
    saved_state = ema.get_state()

    model2 = ExampleModel()
    model2.weight.data.fill_(2.0)
    target_decay = 0.9999
    target_ema = EMATracker(model2, decay=target_decay, faster_decay_at_start=False)

    target_ema.load_ema_state_for_finetuning(saved_state)

    assert int(target_ema.num_updates) == 5
    assert float(target_ema.decay) == pytest.approx(target_decay)
    assert target_ema._faster_decay_at_start is False
    for name in saved_state["ema_params"]:
        torch.testing.assert_close(
            target_ema._ema_params[name],
            saved_state["ema_params"][name].to(get_device()),
        )


def test_load_ema_state_for_finetuning_missing_ema_params():
    """load_ema_state_for_finetuning raises ValueError when ema_params is
    missing (e.g. from a non-restart checkpoint)."""
    model = ExampleModel()
    ema = EMATracker(model, decay=0.999)
    state = ema.get_state()
    del state["ema_params"]

    target_ema = EMATracker(model, decay=0.999)
    with pytest.raises(ValueError, match="does not contain ema_params"):
        target_ema.load_ema_state_for_finetuning(state)


def test_load_ema_state_for_finetuning_places_tensors_on_device():
    """Loaded EMA params are placed on the training device."""
    model = ExampleModel()
    ema = EMATracker(model, decay=0.999)
    for _ in range(3):
        ema(model)
    state = ema.get_state()
    state["num_updates"] = state["num_updates"].cpu()
    state["ema_params"] = {k: v.cpu() for k, v in state["ema_params"].items()}

    target_ema = EMATracker(model, decay=0.999)
    target_ema.load_ema_state_for_finetuning(state)

    device = get_device()
    assert target_ema.num_updates.device == device
    for param in target_ema._ema_params.values():
        assert param.device == device


def test_ema_config_build_no_resume_path():
    """EMAConfig.build with resume_ema_ckpt_path=None creates a fresh tracker."""
    model = ExampleModel()
    config = EMAConfig(decay=0.99)
    ema = config.build(model)
    assert int(ema.num_updates) == 0
    assert float(ema.decay) == pytest.approx(0.99)


def test_ema_config_build_resumes_ema_state(tmp_path: str):
    """EMAConfig.build with resume_ema_ckpt_path loads EMA state from
    the checkpoint while preserving the configured decay."""
    model = ExampleModel()
    model.weight.data.fill_(1.0)
    source_ema = EMATracker(model, decay=0.99, faster_decay_at_start=True)
    model.weight.data.fill_(0.0)
    for _ in range(5):
        source_ema(model)

    ckpt_path = os.path.join(tmp_path, "ckpt.tar")
    torch.save({"ema": source_ema.get_state(), "stepper": {}}, ckpt_path)

    model2 = ExampleModel()
    model2.weight.data.fill_(2.0)
    new_decay = 0.9999
    config = EMAConfig(decay=new_decay, resume_ema_ckpt_path=ckpt_path)
    loaded_ema = config.build(model2)

    assert int(loaded_ema.num_updates) == 5
    assert float(loaded_ema.decay) == pytest.approx(new_decay)
    source_state = source_ema.get_state()
    for name in source_state["ema_params"]:
        torch.testing.assert_close(
            loaded_ema._ema_params[name],
            source_state["ema_params"][name].to(get_device()),
        )


def test_ema_config_build_missing_ema_key(tmp_path: str):
    """EMAConfig.build raises ValueError when the checkpoint has no 'ema' key."""
    ckpt_path = os.path.join(tmp_path, "bad_ckpt.tar")
    torch.save({"stepper": {}, "optimization": {}}, ckpt_path)

    model = ExampleModel()
    config = EMAConfig(resume_ema_ckpt_path=ckpt_path)
    with pytest.raises(ValueError, match="does not contain EMA state"):
        config.build(model)


def _get_trained_ema_state(model: nn.Module) -> dict:
    """An EMA state whose averaged weights differ from the model's weights."""
    ema = EMATracker(model, decay=0.5, faster_decay_at_start=False)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema(model)
    return ema.get_state()


def test_load_ema_params_if_available_copies_ema_weights(
    caplog: pytest.LogCaptureFixture,
):
    model = ExampleModel()
    state = _get_trained_ema_state(model)
    loaded = ExampleModel()

    with caplog.at_level(logging.INFO):
        assert load_ema_params_if_available({"ema": state}, loaded, "ckpt.tar")

    assert "Using EMA weights from ckpt.tar" in caplog.text
    torch.testing.assert_close(loaded.weight, state["ema_params"]["weight"])


def test_load_ema_params_if_available_keeps_untracked_parameters():
    """A parameter frozen during training is not tracked by the EMA. It keeps
    its checkpoint value even though a model rebuilt for inference no longer
    marks it as frozen."""
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).to(get_device())
    model[1].requires_grad_(False)
    state = _get_trained_ema_state(model)
    loaded = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).to(get_device())
    frozen_before = [p.detach().clone() for p in loaded[1].parameters()]

    assert load_ema_params_if_available({"ema": state}, loaded, "ckpt.tar")

    torch.testing.assert_close(loaded[0].weight, state["ema_params"]["0weight"])
    for before, after in zip(frozen_before, loaded[1].parameters()):
        torch.testing.assert_close(after, before)


@pytest.mark.parametrize("has_ema_state", [True, False])
def test_load_ema_params_if_available_without_ema_weights(
    has_ema_state: bool, caplog: pytest.LogCaptureFixture
):
    """Checkpoints saved without their optimization state have EMA state but
    no EMA weights; checkpoints not saved by a Trainer may have neither."""
    model = ExampleModel()
    checkpoint = {}
    if has_ema_state:
        state = _get_trained_ema_state(model)
        del state["ema_params"]
        checkpoint["ema"] = state
    before = model.weight.detach().clone()

    with caplog.at_level(logging.INFO):
        assert not load_ema_params_if_available(checkpoint, model, "best_ckpt.tar")

    torch.testing.assert_close(model.weight, before)
    assert "best_ckpt.tar does not contain EMA weights" in caplog.text


def test_load_ema_params_if_available_mismatched_model():
    state = _get_trained_ema_state(ExampleModel())
    other = nn.Sequential(nn.Linear(2, 2)).to(get_device())

    with pytest.raises(ValueError, match="not a parameter of the model"):
        load_ema_params_if_available({"ema": state}, other, "ckpt.tar")
