# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from fme.downscaling.distillation.student_sampling import (
    sample_student_from_noise,
    sample_student_hi_cascade,
    sample_student_lo_renoise,
)


class _TinyNet(torch.nn.Module):
    """Deterministic denoiser with signature ``net(x, x_lr, t) -> x0``."""

    def __init__(self, c_out: int, c_lr: int) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(c_out + c_lr, c_out, kernel_size=1)
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0.0)
        self.sigmas_called: list[float] = []

    def forward(
        self, x: torch.Tensor, x_lr: torch.Tensor, t: torch.Tensor | float
    ) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            self.sigmas_called.append(float(t.detach().reshape(-1)[0].item()))
        else:
            self.sigmas_called.append(float(t))
        out = self.proj(torch.cat([x.float(), x_lr.float()], dim=1))
        return out.to(dtype=x.dtype)


def test_sample_student_from_noise_shape_and_determinism() -> None:
    B, C_cond, C_out, H, W = 2, 2, 3, 4, 4
    net = _TinyNet(C_out, C_cond).eval()
    condition = torch.randn(B, C_cond, H, W)

    torch.manual_seed(0)
    out1 = sample_student_from_noise(net, condition, C_out, 4, 2, 0.002, 80.0)
    torch.manual_seed(0)
    out2 = sample_student_from_noise(net, condition, C_out, 4, 2, 0.002, 80.0)

    assert out1.shape == (B, 4, C_out, H, W)
    torch.testing.assert_close(out1, out2)


def test_sample_student_lo_renoise_input_is_target_plus_noise() -> None:
    """With eps=0 the initial state is exactly the target, so the net sees it."""
    B, n, C_cond, C_out, H, W = 1, 2, 2, 3, 4, 4
    net = _TinyNet(C_out, C_cond).eval()
    condition = torch.randn(B, C_cond, H, W)
    target = torch.randn(B, n, C_out, H, W)

    out = sample_student_lo_renoise(
        net,
        condition,
        target,
        num_steps=1,
        sigma_min=0.002,
        sigma_max=200.0,
        randn_like=torch.zeros_like,
    )

    assert out.shape == (B, n, C_out, H, W)
    # eps=0 => x_init == target; a different target must change the output,
    # confirming the target (not fresh noise) drives the ensemble.
    other = torch.randn(B, n, C_out, H, W)
    out_other = sample_student_lo_renoise(
        net, condition, other, 1, 0.002, 200.0, randn_like=torch.zeros_like
    )
    assert not torch.allclose(out, out_other)


def test_sample_student_hi_cascade_routes_high_then_low() -> None:
    """The high net is queried above the boundary, the low net at/below it."""
    B, C_cond, C_out, H, W = 1, 2, 1, 4, 4
    hi = _TinyNet(C_out, C_cond).eval()
    lo = _TinyNet(C_out, C_cond).eval()
    condition = torch.randn(B, C_cond, H, W)

    out = sample_student_hi_cascade(
        hi_net=hi,
        lo_net=lo,
        condition=condition,
        c_out=C_out,
        n_samples=2,
        sigma_ranges=[(0.005, 200.0), (200.0, 2000.0)],
        steps_per_range=[1, 1],
    )

    assert out.shape == (B, 2, C_out, H, W)
    assert len(hi.sigmas_called) == 1 and hi.sigmas_called[0] > 200.0
    assert len(lo.sigmas_called) == 1 and lo.sigmas_called[0] <= 200.0
