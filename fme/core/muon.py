"""Muon optimizer with an auxiliary AdamW parameter group.

The Muon math (momentum, Newton-Schulz orthogonalization, learning-rate
adjustment, decoupled weight decay) is copied from PyTorch's
``torch/optim/_muon.py`` at v2.10.0 [*] (BSD-3-Clause, notice below), so ace
does not require torch >= 2.9 or import a private torch module. The copied
functions are unchanged apart from dropping torch's dynamo and tensor-``lr``
wrappers. ``torch.optim.Muon`` itself would not fit even where available: its
constructor rejects parameters with ``ndim != 2`` (conv kernels are 4D) and
it has no AdamW group for the remaining parameters. The ``Muon`` class here
is the layer around the copied functional update:

- Parameters with ``ndim > 2`` are updated as ``(shape[0], -1)`` views, so
  updates write through to the parameter (a non-contiguous parameter is
  updated through a contiguous copy that is copied back). The momentum
  buffer is stored in the parameter's own shape, as in the previously
  vendored implementation, so its checkpoints load unchanged.
- Parameters with ``grad=None`` are skipped; gradients are not modified.
- An AdamW group (defaults match ``torch.optim.AdamW``: betas
  ``(0.9, 0.999)``, eps ``1e-8``) updates the non-Muon parameters.
- ``load_state_dict`` refuses a state dict whose parameter groups are not
  Muon/AdamW groups of the same layout (e.g. one saved by plain AdamW).

The copied Newton-Schulz iteration fuses its bfloat16 matmuls (``addmm``),
so its updates differ from the previously vendored kernel's by bfloat16
rounding (about 3% of the update RMS); runs resumed from vendored-era
checkpoints are not bitwise reproducible.

Muon is single-device: under DDP the gradients are already all-reduced
before ``step``, so every rank computes the same Newton-Schulz update.

[*] https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/_muon.py
"""

# The functions marked "copied from PyTorch" below carry this notice:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)  # noqa: E501
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)  # noqa: E501
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
#    America and IDIAP Research Institute nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch

_MUON_FLAG = "use_muon"
ADJUST_LR_FNS = (None, "original", "match_rms_adamw")

# Copied from PyTorch (torch/optim/_muon.py, v2.10.0).
# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    """Copied from PyTorch (torch/optim/_muon.py, v2.10.0).

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of
    G. We opt to use a quintic iteration whose coefficients are selected to
    maximize the slope at zero. For the purpose of minimizing steps, it turns
    out to be empirically effective to keep increasing the slope at zero even
    beyond the point where the iteration no longer converges all the way to one
    everywhere on the interval. This iteration therefore does not produce UV^T
    but rather something like US'V^T where S' is diagonal with
    S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model performance
    at all relative to UV^T, where USV^T = G is the SVD.

    Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    with suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Copied from PyTorch (torch/optim/_muon.py, v2.10.0).

    Default learning rate adjustment used by Muon.
    """
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


def muon(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    muon_momentum_bufs: list[torch.Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
) -> None:
    """Copied from PyTorch (``_single_tensor_muon`` in torch/optim/_muon.py,
    v2.10.0), without the complex-parameter flag and tensor-``lr`` handling.

    Functional Muon update of 2D ``params`` in place, from their ``grads``
    and ``muon_momentum_bufs`` (also updated in place).
    """
    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


def _as_matrix(t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return a ``(shape[0], -1)`` matrix of ``t`` and whether it is a copy
    (``t`` not contiguous) that must be written back after an in-place update.
    """
    if t.is_contiguous():
        return t.view(t.shape[0], -1), False
    return t.contiguous().view(t.shape[0], -1), True


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Update the Adam moments in place and return the bias-corrected update.

    From Keller Jordan's Muon reference implementation
    (https://github.com/KellerJordan/Muon, ``muon.py``; MIT License,
    Copyright (c) 2024 Keller Jordan).
    """
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

    ``adjust_lr_fn`` sets the per-matrix step size from the flattened shape
    ``(A, B) = (shape[0], prod(shape[1:]))``:

    - ``None``/``"original"`` (Keller Jordan's reference): step
      ``lr * sqrt(max(1, A / B))``.
    - ``"match_rms_adamw"`` (Moonshot, arXiv:2502.16982): step
      ``lr * 0.2 * sqrt(max(A, B))``, which gives the update an RMS of about
      ``0.2 * lr`` to match AdamW. For a 3x3 conv kernel, flattened to
      ``(C_out, C_in * 9)`` with ``C_out <= C_in * 9``, this is a
      ``0.2 * sqrt(9 * C_in)`` times larger step than ``"original"`` at the
      same ``lr`` (9.6x for ``C_in = 256``, 13.6x for 512), while decoupled
      weight decay still uses the unadjusted ``lr``. Switching between the two
      therefore needs ``lr`` and ``weight_decay`` retuned together; e.g. an
      ``lr`` of 1e-3 under ``"original"`` corresponds to roughly 1e-4 under
      ``"match_rms_adamw"``.

    Args:
        muon_params: Parameters with ``ndim >= 2`` to update with Muon.
        adamw_params: Parameters to update with AdamW.
        lr: Muon learning rate.
        momentum: Muon momentum coefficient.
        nesterov: Whether Muon uses Nesterov momentum.
        weight_decay: Decoupled weight decay for the Muon group.
        ns_steps: Number of Newton-Schulz iterations.
        adjust_lr_fn: Muon step-size adjustment, one of ``None``,
            ``"original"`` (same as ``None``) and ``"match_rms_adamw"``.
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
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
    ):
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)
        if adjust_lr_fn not in ADJUST_LR_FNS:
            raise ValueError(
                f"Muon adjust_lr_fn must be one of {list(ADJUST_LR_FNS)}, "
                f"got {adjust_lr_fn!r}."
            )
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
                "adjust_lr_fn": adjust_lr_fn,
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

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # Muon groups saved before adjust_lr_fn existed used "original".
        for group in self.param_groups:
            if group.get(_MUON_FLAG):
                group.setdefault("adjust_lr_fn", None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group[_MUON_FLAG]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        params: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        bufs: list[torch.Tensor] = []
        # (original, matrix copy) pairs to write back after the update
        write_back: list[tuple[torch.Tensor, torch.Tensor]] = []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(
                    p, memory_format=torch.contiguous_format
                )
            if not group["nesterov"] and p.dtype == torch.bfloat16:
                # _zeropower_via_newtonschulz casts its input with .bfloat16() and
                # then normalizes it in place; without Nesterov that input is
                # the momentum buffer itself, which would be overwritten.
                raise ValueError(
                    "Muon with nesterov=False does not support bfloat16 " "parameters."
                )
            # muon reads the gradient only, so a copy is fine.
            grads.append(p.grad.reshape(p.shape[0], -1))
            for t, out in ((p, params), (state["momentum_buffer"], bufs)):
                matrix, is_copy = _as_matrix(t)
                out.append(matrix)
                if is_copy:
                    write_back.append((t, matrix))
        if len(params) == 0:
            return
        muon(
            params,
            grads,
            bufs,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            ns_coefficients=(DEFAULT_A, DEFAULT_B, DEFAULT_C),
            ns_steps=group["ns_steps"],
            eps=EPS,
            adjust_lr_fn=group["adjust_lr_fn"],
        )
        for t, matrix in write_back:
            t.copy_(matrix.view(t.shape))

    def _adamw_step(self, group: dict[str, Any]) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
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
