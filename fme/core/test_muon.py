import copy
from typing import Any

import pytest
import torch
import torch.nn as nn

import fme
from fme.core.muon import Muon, zeropower_via_newtonschulz5
from fme.core.optimization import Optimization, OptimizationConfig
from fme.core.scheduler import SchedulerConfig


class _ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(5)
        # zero-initialized 1x1 projection, like MultiResolutionFiLM.W_scale
        self.W_scale = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        nn.init.zeros_(self.W_scale.weight)

    def forward(self, x):
        h = self.norm(self.conv(x))
        return h * (1.0 + self.W_scale(h))


def _muon_config(**overrides) -> OptimizationConfig:
    kwargs: dict[str, Any] = {
        "optimizer_type": "Muon",
        "lr": 0.02,
        "kwargs": {"adamw_lr": 1e-3},
        "scheduler": SchedulerConfig(),
    }
    kwargs.update(overrides)
    return OptimizationConfig(**kwargs)


def _build(model: nn.Module, config: OptimizationConfig) -> Optimization:
    return config.build(nn.ModuleList([model]), max_epochs=10)


def _group_ids(optimization: Optimization) -> tuple[set[int], set[int]]:
    """Return the ids of the Muon and AdamW group parameters."""
    muon: set[int] = set()
    adamw: set[int] = set()
    for group in optimization.optimizer.param_groups:
        ids = {id(p) for p in group["params"]}
        (muon if group["use_muon"] else adamw).update(ids)
    return muon, adamw


def _train_steps(model: nn.Module, optimization: Optimization, x, n: int) -> list:
    losses = []
    for _ in range(n):
        loss = model(x).pow(2).mean()
        losses.append(loss.item())
        optimization.accumulate_loss(loss)
        optimization.step_weights()
    return losses


def _data() -> torch.Tensor:
    return torch.randn(8, 2, 5, 5, device=fme.get_device())


def test_muon_grouping():
    model = _ConvNet().to(fme.get_device())
    optimization = _build(model, _muon_config())
    muon, adamw = _group_ids(optimization)
    assert muon == {id(model.conv.weight), id(model.W_scale.weight)}
    assert adamw == {
        id(model.conv.bias),
        id(model.norm.weight),
        id(model.norm.bias),
    }
    assert optimization.learning_rate == 0.02
    assert optimization.optimizer.param_groups[1]["lr"] == 1e-3


def test_muon_adamw_names_routes_matrix_to_adamw():
    model = _ConvNet().to(fme.get_device())
    optimization = _build(model, _muon_config(adamw_names=["W_scale"]))
    muon, adamw = _group_ids(optimization)
    assert muon == {id(model.conv.weight)}
    assert id(model.W_scale.weight) in adamw


def test_muon_adamw_names_unmatched_raises():
    model = _ConvNet().to(fme.get_device())
    with pytest.raises(ValueError, match="matched no parameter"):
        _build(model, _muon_config(adamw_names=["W_scale", "not_a_param"]))


def test_muon_requires_matrix_parameter():
    model = nn.LayerNorm(4).to(fme.get_device())
    with pytest.raises(ValueError, match="at least one parameter with ndim >= 2"):
        _build(model, _muon_config())


def test_muon_config_validation():
    with pytest.raises(ValueError, match="adamw_lr"):
        _muon_config(kwargs={})
    with pytest.raises(ValueError, match="Unknown kwargs"):
        _muon_config(kwargs={"adamw_lr": 1e-3, "betas": (0.9, 0.95)})
    with pytest.raises(ValueError, match="only used with optimizer_type='Muon'"):
        OptimizationConfig(optimizer_type="AdamW", adamw_names=["W_scale"])


def test_muon_hyperparameters_from_kwargs():
    model = _ConvNet().to(fme.get_device())
    config = _muon_config(
        kwargs={
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 0.1,
            "ns_steps": 3,
            "adamw_lr": 3e-4,
            "adamw_betas": [0.8, 0.95],
            "adamw_eps": 1e-10,
            "adamw_weight_decay": 0.01,
        }
    )
    muon_group, adamw_group = _build(model, config).optimizer.param_groups
    assert muon_group["momentum"] == 0.9
    assert muon_group["nesterov"] is False
    assert muon_group["weight_decay"] == 0.1
    assert muon_group["ns_steps"] == 3
    assert adamw_group["lr"] == 3e-4
    assert adamw_group["betas"] == (0.8, 0.95)
    assert adamw_group["eps"] == 1e-10
    assert adamw_group["weight_decay"] == 0.01


@pytest.mark.parametrize("enable_amp", [False, True])
@pytest.mark.parametrize("max_grad_norm", [None, 0.1])
def test_muon_step_reduces_loss(enable_amp: bool, max_grad_norm: float | None):
    torch.manual_seed(0)
    model = _ConvNet().to(fme.get_device())
    optimization = _build(
        model,
        _muon_config(
            enable_automatic_mixed_precision=enable_amp, max_grad_norm=max_grad_norm
        ),
    )
    x = _data()
    losses = _train_steps(model, optimization, x, n=2)
    assert losses[1] < losses[0]
    if max_grad_norm is not None:
        assert optimization._last_grad_norm is not None


def test_muon_first_step_moves_zero_init_matrix_unless_excluded():
    """Muon gives a zero-initialized matrix a unit-scale step on step 1;
    routing it to AdamW keeps the step at the AdamW lr scale."""
    for adamw_names, max_expected in [([], None), (["W_scale"], 2e-3)]:
        torch.manual_seed(0)
        model = _ConvNet().to(fme.get_device())
        optimization = _build(model, _muon_config(adamw_names=adamw_names))
        _train_steps(model, optimization, _data(), n=1)
        moved = model.W_scale.weight.abs().max().item()
        if max_expected is None:
            assert moved > 2e-3
        else:
            assert moved <= max_expected


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(8, 16)
    X = zeropower_via_newtonschulz5(G, steps=5).float()
    singular_values = torch.linalg.svdvals(X)
    assert singular_values.min() > 0.5
    assert singular_values.max() < 1.5


def test_muon_state_round_trip():
    """Saving and reloading Muon state gives the same trajectory as
    uninterrupted training, including the LR schedule of both groups."""
    torch.manual_seed(0)
    scheduler = SchedulerConfig(type="CosineAnnealingLR", kwargs={"T_max": 6})
    config = _muon_config(scheduler=scheduler)
    model = _ConvNet().to(fme.get_device())
    optimization = _build(model, config)
    x = _data()
    for _ in range(3):
        _train_steps(model, optimization, x, n=1)
        optimization.step_scheduler()
    saved_state = copy.deepcopy(optimization.get_state())
    saved_model = copy.deepcopy(model.state_dict())
    for _ in range(3):
        _train_steps(model, optimization, x, n=1)
        optimization.step_scheduler()

    model2 = _ConvNet().to(fme.get_device())
    model2.load_state_dict(saved_model)
    optimization2 = _build(model2, config)
    optimization2.load_state(saved_state)
    assert [g["lr"] for g in optimization2.optimizer.param_groups] == [
        g["lr"] for g in saved_state["optimizer_state_dict"]["param_groups"]
    ]
    for _ in range(3):
        _train_steps(model2, optimization2, x, n=1)
        optimization2.step_scheduler()

    for (name, p1), p2 in zip(model.state_dict().items(), model2.state_dict().values()):
        torch.testing.assert_close(p1, p2, msg=name)
    assert [g["lr"] for g in optimization.optimizer.param_groups] == [
        g["lr"] for g in optimization2.optimizer.param_groups
    ]


def _trained_state(config: OptimizationConfig) -> dict:
    torch.manual_seed(0)
    model = _ConvNet().to(fme.get_device())
    optimization = _build(model, config)
    _train_steps(model, optimization, _data(), n=1)
    return optimization.get_state()


@pytest.mark.parametrize("finetune", [False, True])
def test_load_adamw_state_into_muon_raises(finetune: bool):
    adamw_state = _trained_state(OptimizationConfig(optimizer_type="AdamW"))
    optimization = _build(_ConvNet().to(fme.get_device()), _muon_config())
    with pytest.raises(ValueError, match="non-Muon optimizer"):
        if finetune:
            optimization.load_optimizer_state_for_finetuning(adamw_state)
        else:
            optimization.load_state(adamw_state)


@pytest.mark.parametrize("finetune", [False, True])
def test_load_muon_state_into_adamw_raises(finetune: bool):
    muon_state = _trained_state(_muon_config())
    optimization = _build(
        _ConvNet().to(fme.get_device()), OptimizationConfig(optimizer_type="AdamW")
    )
    with pytest.raises(ValueError, match="saved by a Muon optimizer"):
        if finetune:
            optimization.load_optimizer_state_for_finetuning(muon_state)
        else:
            optimization.load_state(muon_state)


def test_load_adamw_state_into_single_group_muon_raises():
    """With no AdamW parameters Muon has one group, so torch's group-count
    check alone would not catch an AdamW state."""
    model = nn.Linear(3, 3, bias=False).to(fme.get_device())
    adamw = torch.optim.AdamW(model.parameters())
    model(torch.randn(2, 3, device=fme.get_device())).sum().backward()
    adamw.step()
    muon = Muon(model.parameters(), [], lr=0.02)
    with pytest.raises(ValueError, match="non-Muon optimizer"):
        muon.load_state_dict(adamw.state_dict())


def test_muon_set_learning_rate_preserves_group_ratio():
    model = _ConvNet().to(fme.get_device())
    optimization = _build(model, _muon_config())
    optimization.set_learning_rate(0.04)
    lrs = [g["lr"] for g in optimization.optimizer.param_groups]
    assert lrs == pytest.approx([0.04, 2e-3])
    assert optimization.learning_rate == 0.04
