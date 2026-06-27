# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for BestStudentCheckpointCallback validation-mode plumbing.

These exercise the per-expert ``lo_renoise`` path and the shared student
sampling dispatch without constructing a real teacher/student (which would pull
in FastGen).  Lightweight fakes stand in for the teacher model's packer +
normalizer and for the student denoiser.
"""

from __future__ import annotations

import pytest
import torch

from fme.core.packer import Packer
from fme.downscaling.distillation.best_student_callback import (
    BestStudentCheckpointCallback,
)


class _ScaleNormalizer:
    """Normalizer that scales every variable by a known factor (identity-ish)."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def normalize(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v * self.factor for k, v in tensors.items()}

    def denormalize(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v / self.factor for k, v in tensors.items()}


class _FakeNormalizer:
    def __init__(self, factor: float) -> None:
        self.fine = _ScaleNormalizer(factor)


class _FakeTeacherModel:
    """Minimal stand-in exposing only what the validation path reads."""

    def __init__(self, names: list[str], factor: float = 0.5) -> None:
        self.out_packer = Packer(list(names))
        self.normalizer = _FakeNormalizer(factor)


class _TinyNet(torch.nn.Module):
    """Deterministic denoiser with signature ``net(x, x_lr, sigma) -> x0``."""

    def __init__(self, c_out: int, c_cond: int) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(c_out + c_cond, c_out, kernel_size=1)
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0.0)

    def forward(
        self, x: torch.Tensor, x_lr: torch.Tensor, t: torch.Tensor | float
    ) -> torch.Tensor:
        out = self.proj(torch.cat([x.float(), x_lr.float()], dim=1))
        return out.to(dtype=x.dtype)


class _FakeStudent:
    def __init__(
        self, net: torch.nn.Module, sigma_min: float, sigma_max: float
    ) -> None:
        self._ace_module = net
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max


def _make_callback(
    teacher_model: _FakeTeacherModel,
    validation_mode: str = "from_noise",
    n_student_samples: int = 2,
    student_sample_steps: int = 1,
) -> BestStudentCheckpointCallback:
    return BestStudentCheckpointCallback(
        val_dataset_path="unused.zarr",
        coarse_val_data=None,  # type: ignore[arg-type]  # not touched in __init__
        teacher_model=teacher_model,  # type: ignore[arg-type]
        best_checkpoint_path="unused.ckpt",
        n_student_samples=n_student_samples,
        student_sample_steps=student_sample_steps,
        validation_mode=validation_mode,
    )


def test_validation_mode_rejects_unknown():
    tm = _FakeTeacherModel(["a"])
    with pytest.raises(ValueError, match="validation_mode"):
        _make_callback(tm, validation_mode="bogus")


def test_packed_target_norm_packs_normalized_first_n_members():
    names = ["a", "b"]
    tm = _FakeTeacherModel(names, factor=0.5)
    cb = _make_callback(tm, validation_mode="lo_renoise", n_student_samples=2)

    B, n_teacher, H, W = 1, 4, 3, 3
    teacher_phys = {
        "a": torch.arange(B * n_teacher * H * W, dtype=torch.float32).reshape(
            B, n_teacher, H, W
        ),
        "b": torch.ones(B, n_teacher, H, W),
    }
    out = cb._packed_target_norm(teacher_phys)

    # (B, n, C_out, H, W): only the first n=2 members, normalized (×0.5).
    assert out.shape == (B, 2, 2, H, W)
    torch.testing.assert_close(out[:, :, 0], teacher_phys["a"][:, :2] * 0.5)
    torch.testing.assert_close(out[:, :, 1], teacher_phys["b"][:, :2] * 0.5)


def test_packed_target_norm_clamps_to_available_members():
    """n_student_samples above the zarr ensemble size yields n_teacher members."""
    tm = _FakeTeacherModel(["a"], factor=1.0)
    cb = _make_callback(tm, validation_mode="lo_renoise", n_student_samples=8)
    B, n_teacher, H, W = 1, 3, 2, 2
    out = cb._packed_target_norm({"a": torch.randn(B, n_teacher, H, W)})
    assert out.shape == (B, n_teacher, 1, H, W)


def test_packed_target_norm_missing_var_raises():
    tm = _FakeTeacherModel(["a", "b"])
    cb = _make_callback(tm, validation_mode="lo_renoise")
    with pytest.raises(ValueError, match="missing"):
        cb._packed_target_norm({"a": torch.zeros(1, 2, 3, 3)})


def test_sample_student_output_lo_renoise_shape():
    names = ["a", "b"]
    tm = _FakeTeacherModel(names)
    cb = _make_callback(
        tm, validation_mode="lo_renoise", n_student_samples=2, student_sample_steps=1
    )
    C_out, C_cond = 2, 3
    student = _FakeStudent(_TinyNet(C_out, C_cond).eval(), 0.005, 200.0)

    B, H, W = 1, 4, 4
    condition = torch.randn(B, C_cond, H, W)
    teacher_phys = {v: torch.randn(B, 3, H, W) for v in names}

    out = cb._sample_student_output(
        student,  # type: ignore[arg-type]  # fake exposes the attrs used
        condition,
        teacher_phys,
        B,
        H,
        W,
        C_out,
    )
    assert out.shape == (B, 2, C_out, H, W)


def test_sample_student_output_from_noise_shape():
    tm = _FakeTeacherModel(["a", "b"])
    cb = _make_callback(
        tm, validation_mode="from_noise", n_student_samples=3, student_sample_steps=2
    )
    C_out, C_cond = 2, 3
    student = _FakeStudent(_TinyNet(C_out, C_cond).eval(), 0.002, 80.0)

    B, H, W = 2, 4, 4
    condition = torch.randn(B, C_cond, H, W)
    # teacher_phys is unused by the from_noise path.
    out = cb._sample_student_output(
        student,  # type: ignore[arg-type]  # fake exposes the attrs used
        condition,
        {},
        B,
        H,
        W,
        C_out,
    )
    assert out.shape == (B, 3, C_out, H, W)
