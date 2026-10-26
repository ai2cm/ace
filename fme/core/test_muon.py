import copy
import math
from typing import Any

import pytest
import torch
import torch.nn as nn

import fme
from fme.core import muon as fme_muon
from fme.core.muon import Muon
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
    with pytest.raises(ValueError, match="adjust_lr_fn"):
        _muon_config(kwargs={"adamw_lr": 1e-3, "adjust_lr_fn": "spectral"})
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
            "adjust_lr_fn": "match_rms_adamw",
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
    assert muon_group["adjust_lr_fn"] == "match_rms_adamw"
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


def test_muon_invalid_adjust_lr_fn_raises():
    with pytest.raises(ValueError, match="adjust_lr_fn"):
        Muon([nn.Parameter(torch.randn(3, 3))], [], lr=0.02, adjust_lr_fn="spectral")


@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("adjust_lr_fn", [None, "match_rms_adamw"])
def test_copied_muon_matches_torch(nesterov: bool, adjust_lr_fn: str | None):
    """fme.core.muon.muon is a copy of torch's functional Muon; where torch
    ships it (torch >= 2.9), the two give identical updates."""
    torch_muon = pytest.importorskip("torch.optim._muon")
    assert (
        torch_muon.DEFAULT_A,
        torch_muon.DEFAULT_B,
        torch_muon.DEFAULT_C,
        torch_muon.DEFAULT_NS_STEPS,
        torch_muon.EPS,
    ) == (
        fme_muon.DEFAULT_A,
        fme_muon.DEFAULT_B,
        fme_muon.DEFAULT_C,
        fme_muon.DEFAULT_NS_STEPS,
        fme_muon.EPS,
    )
    torch.manual_seed(0)
    device = fme.get_device()
    shapes = [(4, 27), (24, 6), (6, 24)]
    params = [torch.randn(shape, device=device) for shape in shapes]
    bufs = [torch.randn_like(p) for p in params]
    copies = {
        name: ([p.clone() for p in params], [b.clone() for b in bufs])
        for name in ("fme", "torch")
    }
    kwargs: dict[str, Any] = dict(
        lr=0.02,
        weight_decay=0.1,
        momentum=0.9,
        nesterov=nesterov,
        ns_coefficients=(
            fme_muon.DEFAULT_A,
            fme_muon.DEFAULT_B,
            fme_muon.DEFAULT_C,
        ),
        ns_steps=fme_muon.DEFAULT_NS_STEPS,
        eps=fme_muon.EPS,
        adjust_lr_fn=adjust_lr_fn,
    )
    for _ in range(3):
        grads = [torch.randn_like(p) for p in params]
        fme_params, fme_bufs = copies["fme"]
        fme_muon.muon(fme_params, grads, fme_bufs, **kwargs)
        torch_params, torch_bufs = copies["torch"]
        torch_muon.muon(torch_params, grads, torch_bufs, **kwargs, has_complex=False)
    for fme_t, torch_t in zip(
        copies["fme"][0] + copies["fme"][1], copies["torch"][0] + copies["torch"][1]
    ):
        torch.testing.assert_close(fme_t, torch_t, rtol=0, atol=0)


def _reference_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz from the previously vendored Keller Jordan Muon."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def _copied_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    return fme_muon._zeropower_via_newtonschulz(
        G,
        (fme_muon.DEFAULT_A, fme_muon.DEFAULT_B, fme_muon.DEFAULT_C),
        steps,
        fme_muon.EPS,
    )


def _reference_muon_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float,
    momentum: float,
    nesterov: bool,
    weight_decay: float,
    newton_schulz=_reference_newtonschulz5,
) -> None:
    """One step of the previously vendored Muon update, in place."""
    momentum_buffer.lerp_(grad, 1 - momentum)
    update = grad.lerp(momentum_buffer, momentum) if nesterov else momentum_buffer
    update = update.reshape(update.shape[0], -1)
    update = newton_schulz(update, 5)
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5
    p.mul_(1 - lr * weight_decay)
    p.add_(update.reshape(p.shape).to(p.dtype), alpha=-lr)


# conv kernel (4, 3, 3, 3) flattens to wide (4, 27); (24, 6) is tall
_EQUIVALENCE_SHAPES = [(4, 3, 3, 3), (24, 6), (6, 24)]


@pytest.mark.parametrize("newton_schulz", ["vendored", "copied"])
@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_matches_vendored_reference(nesterov: bool, newton_schulz: str):
    """With the default adjust_lr_fn the optimizer reproduces the
    previously vendored update.

    With the copied torch Newton-Schulz kernel in the reference, the match is exact.
    The vendored kernel computes the same bfloat16 iteration with unfused
    matmuls, so its updates differ from torch's by bfloat16 rounding (a few
    percent of the update RMS per step).
    """
    torch.manual_seed(0)
    device = fme.get_device()
    lr, momentum, weight_decay, n_steps = 0.02, 0.9, 0.1, 4
    params = [
        nn.Parameter(torch.randn(shape, device=device)) for shape in _EQUIVALENCE_SHAPES
    ]
    initial = [p.detach().clone() for p in params]
    ref_params = [p.detach().clone() for p in params]
    ref_bufs = [torch.zeros_like(p) for p in params]
    optimizer = Muon(
        params,
        [],
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay,
    )
    ns = {"vendored": _reference_newtonschulz5, "copied": _copied_newtonschulz5}
    for _ in range(n_steps):
        grads = [torch.randn_like(p) for p in params]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        optimizer.step()
        for p, g, buf in zip(ref_params, grads, ref_bufs):
            _reference_muon_step(
                p, g, buf, lr, momentum, nesterov, weight_decay, ns[newton_schulz]
            )
    for p, p0, ref, ref_buf in zip(params, initial, ref_params, ref_bufs):
        buf = optimizer.state[p]["momentum_buffer"]
        assert buf.shape == p.shape
        torch.testing.assert_close(buf, ref_buf)
        if newton_schulz == "copied":
            torch.testing.assert_close(p.detach(), ref)
        else:
            delta, ref_delta = p.detach() - p0, ref - p0
            rel_rms = (delta - ref_delta).norm() / ref_delta.norm()
            # measured: relative RMS <= 3%, max abs <= 0.11 * lr after 4 steps
            assert rel_rms < 0.05
            torch.testing.assert_close(delta, ref_delta, rtol=0, atol=0.2 * lr)


def test_muon_non_contiguous_param_and_grad():
    """Updates write through to non-contiguous (channels_last) parameters."""
    torch.manual_seed(0)
    device = fme.get_device()
    weight = torch.randn(4, 3, 3, 3, device=device)
    contiguous = nn.Parameter(weight.clone())
    channels_last = nn.Parameter(weight.clone().to(memory_format=torch.channels_last))
    assert not channels_last.is_contiguous()
    optimizers = [Muon([p], [], lr=0.02) for p in (contiguous, channels_last)]
    for _ in range(2):
        grad = torch.randn_like(weight)
        contiguous.grad = grad.clone()
        channels_last.grad = grad.clone().to(memory_format=torch.channels_last)
        for optimizer in optimizers:
            optimizer.step()
    torch.testing.assert_close(channels_last.detach(), contiguous.detach())
    assert not torch.equal(contiguous.detach(), weight)


@pytest.mark.parametrize("shape", [(32, 8), (8, 32), (16, 4, 3, 3)])
def test_muon_match_rms_adamw_update_rms(shape: tuple[int, ...]):
    """From zero momentum with wd=0, match_rms_adamw scales the "original"
    step by 0.2 * sqrt(max(A, B)) / sqrt(max(1, A / B)) and gives an update
    RMS of about 0.2 * lr."""
    torch.manual_seed(0)
    lr = 0.01
    # zero weight: the parameter change is exactly the update
    weight = torch.zeros(shape, device=fme.get_device())
    grad = torch.randn_like(weight)
    deltas = {}
    for adjust_lr_fn in ["original", "match_rms_adamw"]:
        p = nn.Parameter(weight.clone())
        p.grad = grad.clone()
        Muon([p], [], lr=lr, adjust_lr_fn=adjust_lr_fn).step()
        deltas[adjust_lr_fn] = (p.detach() - weight).reshape(shape[0], -1)
    A, B = deltas["original"].shape
    ratio = 0.2 * math.sqrt(max(A, B)) / math.sqrt(max(1, A / B))
    torch.testing.assert_close(deltas["match_rms_adamw"], deltas["original"] * ratio)
    rms = deltas["match_rms_adamw"].pow(2).mean().sqrt().item()
    # Newton-Schulz singular values are only roughly 1 (in [0.5, 1.5])
    assert 0.6 * 0.2 * lr < rms < 1.2 * 0.2 * lr


def _vendored_layout_state(model: nn.Module) -> dict[str, Any]:
    """An optimizer state dict as saved by the vendored Muon: momentum buffers
    in parameter shape and Muon groups without ``adjust_lr_fn``."""
    optimizer = Muon(
        [model.conv.weight, model.W_scale.weight],
        [model.conv.bias, model.norm.weight, model.norm.bias],
        lr=0.02,
        adamw_lr=1e-3,
    )
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    del state["param_groups"][0]["adjust_lr_fn"]
    return state


def test_muon_loads_vendored_layout_state_dict():
    """A state saved by the vendored Muon loads, and the next step uses its
    momentum buffers as the vendored update would."""
    torch.manual_seed(0)
    model = _ConvNet().to(fme.get_device())
    saved = _vendored_layout_state(model)
    assert saved["state"][0]["momentum_buffer"].shape == model.conv.weight.shape
    optimization = _build(model, _muon_config())
    optimization.load_state(
        {
            "optimizer_state_dict": saved,
            "scheduler_state_dict": optimization.scheduler.state_dict(),
            "gscaler_state_dict": None,
        }
    )
    assert optimization.optimizer.param_groups[0]["adjust_lr_fn"] is None
    expected = model.conv.weight.detach().clone()
    grad = torch.randn_like(expected)
    _reference_muon_step(
        expected,
        grad,
        saved["state"][0]["momentum_buffer"].clone(),
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.0,
        newton_schulz=_copied_newtonschulz5,
    )
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    model.conv.weight.grad = grad
    optimization.optimizer.step()
    torch.testing.assert_close(model.conv.weight.detach(), expected)


def test_muon_bfloat16_without_nesterov_raises():
    """The Newton-Schulz kernel normalizes a bfloat16 input in place, which
    without Nesterov is the momentum buffer itself."""
    p = nn.Parameter(torch.randn(4, 6, dtype=torch.bfloat16))
    p.grad = torch.randn_like(p)
    optimizer = Muon([p], [], lr=0.02, nesterov=False)
    with pytest.raises(ValueError, match="bfloat16"):
        optimizer.step()
