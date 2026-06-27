# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
Sampling strategies for validating distilled students.

Three ways to produce a student output ensemble from coarse conditioning,
used by ``BestStudentCheckpointCallback``. All operate in normalized space
(``sigma_data = 1``) and return ``(B, n, C_out, H, W)`` normalized output:

- ``sample_student_from_noise``: an end-to-end student denoises fresh noise
  over its full sigma range. The original single-student validation path.
- ``sample_student_lo_renoise``: a low-noise segment student denoises the
  teacher target re-noised to its ``sigma_max`` — i.e. standard diffusion
  validation of a one-shot denoiser, no upstream student or live teacher
  needed (the noise-dominated input makes the exact source immaterial).
- ``sample_student_hi_cascade``: a high-noise segment student is validated
  end-to-end through a frozen low-noise student, routing each cascade step to
  the segment whose sigma range contains it (the bundle deployment path).

See ``MOE_DISTILLATION_STATUS.md`` ("Validation & checkpoint selection").
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from fme.downscaling.samplers import boundary_aligned_t_list, fastgen_sampler


def sample_student_from_noise(
    net: torch.nn.Module,
    condition: torch.Tensor,
    c_out: int,
    n_samples: int,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    """Draw ``n_samples`` student outputs per batch element from fresh noise.

    Args:
        net: Denoiser with signature ``net(x, condition, sigma) -> x0``.
        condition: ``(B, C_cond, H, W)`` conditioning tensor.
        c_out: Number of output channels.
        n_samples: Ensemble members to draw per batch element.
        num_steps: Student denoising steps (FastGen predict-x0-renoise).
        sigma_min: Lower bound of the student sigma range.
        sigma_max: Upper bound of the student sigma range.

    Returns:
        ``(B, n_samples, c_out, H, W)`` normalized student output.
    """
    B, _, H, W = condition.shape
    condition_rep = condition.repeat_interleave(n_samples, dim=0)
    noise = torch.randn(B * n_samples, c_out, H, W, device=condition.device)
    with torch.amp.autocast(condition.device.type, dtype=torch.bfloat16):
        out, _ = fastgen_sampler(
            net,
            noise,
            condition_rep,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    return out.reshape(B, n_samples, c_out, H, W)


def sample_student_lo_renoise(
    net: torch.nn.Module,
    condition: torch.Tensor,
    target_norm: torch.Tensor,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    randn_like: Callable[[torch.Tensor], torch.Tensor] = torch.randn_like,
) -> torch.Tensor:
    """Validate a low-noise student by re-noising the target to ``sigma_max``.

    Builds ``x = target_norm + sigma_max * eps`` and denoises from there, so no
    upstream student or live teacher is needed. The student is deterministic
    given its input, so the ensemble comes from the ``n`` target members (one
    ``eps`` draw each), not from fresh noise draws.

    Args:
        net: Denoiser with signature ``net(x, condition, sigma) -> x0``.
        condition: ``(B, C_cond, H, W)`` conditioning tensor.
        target_norm: ``(B, n, C_out, H, W)`` normalized teacher target members
            to re-noise.
        num_steps: Student denoising steps.
        sigma_min: Lower bound of the student sigma range.
        sigma_max: Upper bound of the student sigma range (the segment
            boundary, e.g. 200).
        randn_like: Noise generator (overridable for deterministic tests).

    Returns:
        ``(B, n, c_out, H, W)`` normalized student output.

    Note:
        The state is built at exactly ``sigma_max`` while the sampler queries
        the net at ``t_list[0]`` (~0.5% below ``sigma_max`` under the default
        schedule). The mismatch is immaterial for a noise-dominated input.
    """
    B, n, c_out, H, W = target_norm.shape
    condition_rep = condition.repeat_interleave(n, dim=0)
    flat = target_norm.reshape(B * n, c_out, H, W)
    x_init = flat + sigma_max * randn_like(flat)
    with torch.amp.autocast(condition.device.type, dtype=torch.bfloat16):
        out, _ = fastgen_sampler(
            net,
            x_init,
            condition_rep,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            skip_noise_scale=True,
        )
    return out.reshape(B, n, c_out, H, W)


def sample_student_hi_cascade(
    hi_net: torch.nn.Module,
    lo_net: torch.nn.Module,
    condition: torch.Tensor,
    c_out: int,
    n_samples: int,
    sigma_ranges: list[tuple[float, float]],
    steps_per_range: list[int],
) -> torch.Tensor:
    """Validate a high-noise student end-to-end through a frozen low-noise one.

    Runs a sigma-dispatched cascade: fresh noise at the top sigma denoises via
    ``hi_net``, re-noises down to the boundary, then ``lo_net`` finishes to 0.
    Each step routes to the segment whose inclusive sigma range contains it.

    Args:
        hi_net: High-noise segment denoiser (covers the top sigma range).
        lo_net: Frozen low-noise segment denoiser (covers the bottom range).
        condition: ``(B, C_cond, H, W)`` conditioning tensor.
        c_out: Number of output channels.
        n_samples: Ensemble members to draw per batch element.
        sigma_ranges: Inclusive ranges sorted ascending by ``sigma_min``,
            e.g. ``[(0.005, 200), (200, 2000)]`` (low first, high last).
        steps_per_range: Step count per range, aligned with ``sigma_ranges``.

    Returns:
        ``(B, n_samples, c_out, H, W)`` normalized student output.
    """
    from fme.downscaling.predictors.serial_denoising import _SigmaDispatchModule

    if len(sigma_ranges) != 2:
        raise ValueError(
            f"sample_student_hi_cascade expects two sigma ranges, "
            f"got {len(sigma_ranges)}."
        )
    B, _, H, W = condition.shape
    # Modules align with sigma_ranges (ascending): low first, high last.
    dispatch = _SigmaDispatchModule(sigma_ranges, [lo_net, hi_net])
    t_list = boundary_aligned_t_list(
        sigma_ranges, steps_per_range, device=condition.device
    )
    condition_rep = condition.repeat_interleave(n_samples, dim=0)
    noise = torch.randn(B * n_samples, c_out, H, W, device=condition.device)
    with torch.amp.autocast(condition.device.type, dtype=torch.bfloat16):
        out, _ = fastgen_sampler(dispatch, noise, condition_rep, t_list=t_list)
    return out.reshape(B, n_samples, c_out, H, W)
