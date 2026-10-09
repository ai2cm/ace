"""Muon optimizer with an auxiliary AdamW parameter group.

Vendored from Keller Jordan's reference implementation
(https://github.com/KellerJordan/Muon, ``muon.py``,
``SingleDeviceMuonWithAuxAdam``; MIT License, Copyright (c) 2024 Keller
Jordan), because torch < 2.9 ships no Muon. Changes from the reference:

- Gradients are not modified in place (the reference's Nesterov update
  overwrites ``p.grad``); parameters with ``grad=None`` are skipped instead
  of being given zero gradients.
- Any parameter with ``ndim > 2`` (not only 4D conv kernels) is flattened
  to ``(shape[0], -1)`` for the orthogonalization.
- ``nesterov`` and ``ns_steps`` are configurable, and AdamW defaults match
  ``torch.optim.AdamW`` (betas ``(0.9, 0.999)``, eps ``1e-8``).
- ``load_state_dict`` refuses a state dict whose parameter groups are not
  Muon/AdamW groups of the same layout (e.g. one saved by plain AdamW).

Muon is single-device: under DDP the gradients are already all-reduced
before ``step``, so every rank computes the same Newton-Schulz update.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import torch

_MUON_FLAG = "use_muon"


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximately orthogonalize a matrix with a quintic Newton-Schulz
    iteration.

    The coefficients maximize the slope at zero, so the result is not exactly
    ``U V^T`` but ``U S' V^T`` with ``S'`` roughly uniform in [0.5, 1.5],
    which does not hurt model performance relative to exact orthogonalization.
    Runs in bfloat16 as in the reference implementation.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float,
    ns_steps: int,
    nesterov: bool,
) -> torch.Tensor:
    """Update the momentum buffer in place and return the orthogonalized
    update, shaped as a 2D ``(shape[0], -1)`` matrix.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim > 2:
        update = update.reshape(update.shape[0], -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    exp_avg_c = exp_avg / (1 - betas[0] ** step)
    exp_avg_sq_c = exp_avg_sq / (1 - betas[1] ** step)
    return exp_avg_c / (exp_avg_sq_c.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """Muon for matrix parameters plus AdamW for the rest.

    Parameter group 0 is the Muon group (its ``lr`` is the Muon learning
    rate); group 1, present only if ``adamw_params`` is non-empty, is the
    AdamW group. Each group carries a ``use_muon`` flag.

    Muon replaces each matrix's momentum by its approximate orthogonalization
    (unit spectral scale regardless of the gradient magnitude), so pre-step
    gradient-norm clipping changes the direction of the Muon update only
    through the momentum mix, not its size; the clip matters mainly for the
    AdamW group.

    Args:
        muon_params: Parameters with ``ndim >= 2`` to update with Muon.
        adamw_params: Parameters to update with AdamW.
        lr: Muon learning rate.
        momentum: Muon momentum coefficient.
        nesterov: Whether Muon uses Nesterov momentum.
        weight_decay: Decoupled weight decay for the Muon group.
        ns_steps: Number of Newton-Schulz iterations.
        adamw_lr: AdamW group learning rate.
        adamw_betas: AdamW group betas.
        adamw_eps: AdamW group epsilon.
        adamw_weight_decay: Decoupled weight decay for the AdamW group.
    """

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
    ):
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)
        if len(muon_params) == 0:
            raise ValueError("Muon requires at least one parameter with ndim >= 2.")
        for p in muon_params:
            if p.ndim < 2:
                raise ValueError(
                    f"Muon parameters must have ndim >= 2, got shape {tuple(p.shape)}."
                )
            if p.is_complex():
                raise ValueError(
                    "Muon does not support complex parameters; route them to "
                    "the AdamW group."
                )
        param_groups: list[dict[str, Any]] = [
            {
                "params": muon_params,
                "lr": lr,
                "momentum": momentum,
                "nesterov": nesterov,
                "weight_decay": weight_decay,
                "ns_steps": ns_steps,
                _MUON_FLAG: True,
            }
        ]
        if len(adamw_params) > 0:
            param_groups.append(
                {
                    "params": adamw_params,
                    "lr": adamw_lr,
                    "betas": tuple(adamw_betas),
                    "eps": adamw_eps,
                    "weight_decay": adamw_weight_decay,
                    _MUON_FLAG: False,
                }
            )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group[_MUON_FLAG]:
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                    update = update.reshape(p.shape).to(p.dtype)
                else:
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        check_muon_state_dict_compatible(self, state_dict)
        super().load_state_dict(state_dict)


def is_muon_state_dict(state_dict: Mapping[str, Any]) -> bool:
    """Whether an optimizer state dict was saved by :class:`Muon`."""
    return any(_MUON_FLAG in g for g in state_dict.get("param_groups", []))


def check_muon_state_dict_compatible(
    optimizer: torch.optim.Optimizer, state_dict: Mapping[str, Any]
) -> None:
    """Raise if ``state_dict`` cannot be loaded into ``optimizer`` because one
    is a Muon optimizer and the other is not, or their Muon/AdamW group layout
    differs.

    ``torch.optim.Optimizer.load_state_dict`` only checks group and parameter
    counts, so without this check e.g. an AdamW state could be loaded into a
    single-group Muon optimizer and replace its hyperparameters.
    """
    current_is_muon = isinstance(optimizer, Muon)
    saved_is_muon = is_muon_state_dict(state_dict)
    if current_is_muon and not saved_is_muon:
        raise ValueError(
            "Cannot load an optimizer state saved by a non-Muon optimizer "
            "(e.g. Adam/AdamW) into a Muon optimizer. Start the optimizer "
            "state fresh instead of resuming it."
        )
    if saved_is_muon and not current_is_muon:
        raise ValueError(
            "Cannot load an optimizer state saved by a Muon optimizer into the "
            f"current {type(optimizer).__name__} optimizer. Start the optimizer state "
            "fresh instead of resuming it."
        )
    if current_is_muon:
        current_flags = [g[_MUON_FLAG] for g in optimizer.param_groups]
        saved_flags = [g.get(_MUON_FLAG) for g in state_dict["param_groups"]]
        if current_flags != saved_flags:
            raise ValueError(
                "Muon optimizer state has a different Muon/AdamW group layout "
                f"(use_muon flags {saved_flags}) than the current optimizer "
                f"({current_flags})."
            )
