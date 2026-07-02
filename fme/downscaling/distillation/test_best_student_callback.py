# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for BestStudentCheckpointCallback validation-mode plumbing.

These exercise the per-expert ``lo_renoise`` path and the shared student
sampling dispatch without constructing a real teacher/student (which would pull
in FastGen).  Lightweight fakes stand in for the teacher model's packer +
normalizer and for the student denoiser.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fme.core.packer import Packer
from fme.downscaling.distillation.best_student_callback import (
    BestStudentCheckpointCallback,
    _normalized_mean,
    _tail_deviation_score,
    _tail_magnitude,
    _tail_quantile_level,
)


class _ScaleNormalizer:
    """Normalizer that scales every variable by a known factor (identity-ish)."""

    def __init__(
        self, factor: float, stds: dict[str, torch.Tensor] | None = None
    ) -> None:
        self.factor = factor
        self.stds = stds or {}

    def normalize(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v * self.factor for k, v in tensors.items()}

    def denormalize(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v / self.factor for k, v in tensors.items()}


class _FakeNormalizer:
    def __init__(
        self, factor: float, stds: dict[str, torch.Tensor] | None = None
    ) -> None:
        self.fine = _ScaleNormalizer(factor, stds)


class _FakeTeacherModel:
    """Minimal stand-in exposing only what the validation path reads."""

    def __init__(
        self,
        names: list[str],
        factor: float = 0.5,
        stds: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.out_packer = Packer(list(names))
        self.normalizer = _FakeNormalizer(factor, stds)


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
        None,  # base_norm (non-residual)
        B,
        H,
        W,
        C_out,
    )
    assert out.shape == (B, 2, C_out, H, W)


def test_sample_student_output_adds_residual_base():
    """For a residual teacher the interpolated-coarse base is added back to the
    student's (residual) output; passing base=0 vs base=const shifts the output
    by exactly that constant."""
    tm = _FakeTeacherModel(["a", "b"])
    cb = _make_callback(tm, validation_mode="from_noise", n_student_samples=2)
    C_out, C_cond = 2, 3
    net = _TinyNet(C_out, C_cond).eval()
    student = _FakeStudent(net, 0.002, 80.0)
    B, H, W = 1, 4, 4
    torch.manual_seed(0)
    condition = torch.randn(B, C_cond, H, W)

    torch.manual_seed(1)
    out_zero = cb._sample_student_output(
        student,  # type: ignore[arg-type]
        condition,
        {},
        torch.zeros(B, C_out, H, W),
        B,
        H,
        W,
        C_out,
    )
    base = torch.full((B, C_out, H, W), 5.0)
    torch.manual_seed(1)
    out_base = cb._sample_student_output(
        student,  # type: ignore[arg-type]
        condition,
        {},
        base,
        B,
        H,
        W,
        C_out,
    )
    # Residual output identical; base add-back shifts every sample by +5.
    torch.testing.assert_close(out_base - out_zero, torch.full_like(out_base, 5.0))


def test_base_prediction_norm_none_for_nonresidual():
    """No base add-back when the teacher isn't a residual model."""
    tm = _FakeTeacherModel(["a"])  # _FakeTeacherModel has no .config
    cb = _make_callback(tm)
    assert cb._base_prediction_norm(object(), np.array([True])) is None


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
        None,  # base_norm (non-residual)
        B,
        H,
        W,
        C_out,
    )
    assert out.shape == (B, 3, C_out, H, W)


# --------------------------------------------------------------------------
# Generation-logic parity: the validation base add-back must match the real
# DiffusionModel generation path (postprocess_generated).  This is the guard
# that catches the residual-base bug (validation forgetting the base) and any
# future divergence of the validation sampler from actual generation.
# --------------------------------------------------------------------------


def _build_residual_model(coarse_shape=(8, 16), fine_shape=(16, 32)):
    """A tiny real residual ``DiffusionModel`` (predict_residual=True).

    Uses only ``fme.downscaling.models`` (no FastGen), so this runs in the
    plain ``fme`` env. PRMSL-like normalization stats exercise the base scaling.
    """
    from fme.core.coordinates import LatLonCoordinates
    from fme.core.loss import LossConfig
    from fme.core.normalizer import NormalizationConfig
    from fme.downscaling.data import StaticInputs
    from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
    from fme.downscaling.modules.diffusion_registry import (
        DiffusionModuleRegistrySelector,
    )

    def _coord(n: int) -> torch.Tensor:
        bounds = torch.linspace(0, float(n), n + 1)
        return (bounds[:-1] + bounds[1:]) / 2

    fine_coords = LatLonCoordinates(
        lat=_coord(fine_shape[0]), lon=_coord(fine_shape[1])
    )
    norm = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 1000.0}, stds={"x": 15.0}),
        NormalizationConfig(means={"x": 1000.0}, stds={"x": 15.0}),
    )
    return DiffusionModelConfig(
        module=DiffusionModuleRegistrySelector(
            "unet_diffusion_song", {"model_channels": 4}
        ),
        loss=LossConfig(type="MSE"),
        in_names=["x"],
        out_names=["x"],
        normalization=norm,
        p_mean=-1.0,
        p_std=1.0,
        sigma_min=0.1,
        sigma_max=1.0,
        churn=0.0,
        num_diffusion_generation_steps=3,
        predict_residual=True,
        use_fine_topography=False,
    ).build(
        coarse_shape,
        downscale_factor=2,
        full_fine_coords=fine_coords,
        static_inputs=StaticInputs(fields=[], coords=fine_coords),
    )


class _CoarseBatch:
    """Minimal stand-in for the coarse batch (only ``.data`` is read)."""

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        self.data = data


def test_validation_base_addback_matches_model_generation():
    """The callback's residual→full-field processing (``_base_prediction_norm`` +
    add-back + denormalize) must exactly equal the model's real generation path
    (``postprocess_generated``). This is the invariant the residual bug broke."""
    model = _build_residual_model()
    cb = _make_callback(model, validation_mode="from_noise", n_student_samples=1)

    B, C = 1, 1
    Hc, Wc = 8, 16
    Hf, Wf = 16, 32
    torch.manual_seed(0)
    coarse = {"x": torch.randn(B, Hc, Wc) * 15.0 + 1000.0}
    residual_norm = torch.randn(B, C, Hf, Wf)  # student residual (B*n, C, H, W), n=1

    # Real generation path: adds base + separates samples + denormalizes.
    full_real, _ = model.postprocess_generated(residual_norm.clone(), coarse, 1)

    # Callback path: base add-back in normalized space, then same denormalize.
    base = cb._base_prediction_norm(_CoarseBatch(coarse), np.array([True] * B))
    assert base is not None  # residual model
    full_norm_cb = residual_norm.reshape(B, 1, C, Hf, Wf) + base.unsqueeze(1)
    full_cb = model.normalizer.fine.denormalize(
        model.out_packer.unpack(full_norm_cb, axis=-3)
    )

    for k in full_real:
        torch.testing.assert_close(full_cb[k], full_real[k])


# --------------------------------------------------------------------------
# Reduction helpers: direction-aware tails + std-normalized cross-var CRPS.
# --------------------------------------------------------------------------


def test_tail_quantile_level_upper_and_lower():
    assert _tail_quantile_level(99.99, "upper") == pytest.approx(0.9999)
    assert _tail_quantile_level(99.99, "lower") == pytest.approx(0.0001)
    assert _tail_quantile_level(99.9999, "lower") == pytest.approx(1e-6)
    with pytest.raises(ValueError, match="upper.*lower"):
        _tail_quantile_level(99.99, "sideways")


def test_normalized_mean_equalizes_variable_magnitudes():
    # PRMSL (~1e3) and precip (~1e-5) differ by ~8 orders of magnitude;
    # dividing each by its own scale puts them on equal footing.
    values = {"PRMSL": 7.0, "PRATEsfc": 5e-5}
    scales = {"PRMSL": 7.0, "PRATEsfc": 5e-5}
    assert _normalized_mean(values, scales) == pytest.approx(1.0)
    # Missing scale falls back to 1.0 (raw value); empty -> inf.
    assert _normalized_mean({"a": 4.0}, {}) == pytest.approx(4.0)
    assert _normalized_mean({}, {}) == float("inf")


def test_normalized_mean_not_dominated_by_large_magnitude_variable():
    scales = {"PRMSL": 7.0, "PRATEsfc": 5e-5}
    base = _normalized_mean({"PRMSL": 7.0, "PRATEsfc": 5e-5}, scales)
    # A 2x worse precip CRPS and a 2x worse PRMSL CRPS move the mean equally —
    # the whole point of normalizing (raw physical means would ignore precip).
    worse_precip = _normalized_mean({"PRMSL": 7.0, "PRATEsfc": 1e-4}, scales)
    worse_prmsl = _normalized_mean({"PRMSL": 14.0, "PRATEsfc": 5e-5}, scales)
    assert worse_precip == pytest.approx(worse_prmsl)
    assert worse_precip > base


def test_tail_deviation_score_penalizes_each_variable_no_cancellation():
    assert _tail_deviation_score({"a": 1.0, "b": 1.0}) == pytest.approx(0.0)
    # Over- and under-prediction both count; a mean-of-ratios would cancel
    # these to 1.0 (score 0), hiding that both variables are 20% off.
    assert _tail_deviation_score({"a": 1.2, "b": 0.8}) == pytest.approx(0.2)
    assert _tail_deviation_score({}) == float("inf")


def test_per_var_scales_reads_normalizer_std():
    tm = _FakeTeacherModel(
        ["PRMSL", "PRATEsfc"],
        stds={"PRMSL": torch.tensor(7.0), "PRATEsfc": torch.tensor(5e-5)},
    )
    scales = BestStudentCheckpointCallback._per_var_scales(tm)  # type: ignore[arg-type]
    assert scales["PRMSL"] == pytest.approx(7.0)
    assert scales["PRATEsfc"] == pytest.approx(5e-5)


def test_per_var_scales_missing_or_nonpositive_std_falls_back_to_one():
    tm = _FakeTeacherModel(["a", "b"], stds={"b": torch.tensor(0.0)})
    scales = BestStudentCheckpointCallback._per_var_scales(tm)  # type: ignore[arg-type]
    assert scales["a"] == 1.0  # no std entry
    assert scales["b"] == 1.0  # non-positive std


def test_tail_magnitude_reference_and_direction():
    # Zero reference recovers the raw value (winds/precip behavior).
    assert _tail_magnitude(40.0, 0.0, "upper") == pytest.approx(40.0)
    # PRMSL lower tail: depth below 1000 hPa.
    assert _tail_magnitude(955.0, 1000.0, "lower") == pytest.approx(45.0)
    with pytest.raises(ValueError, match="upper.*lower"):
        _tail_magnitude(1.0, 0.0, "sideways")


def test_tail_magnitude_makes_prmsl_ratio_sensitive_to_deep_lows():
    """A raw-pressure ratio is ~1.0 for a several-hPa deep-low error; the
    depth-below-1000 ratio exposes it."""
    student_p, target_p = 958.0, 953.0  # student low is 5 hPa too shallow
    raw_ratio = student_p / target_p  # ~1.005 — looks "perfect"
    depth_ratio = _tail_magnitude(student_p, 1000.0, "lower") / _tail_magnitude(
        target_p, 1000.0, "lower"
    )  # 42/47
    assert raw_ratio == pytest.approx(1.0052, abs=1e-3)
    assert depth_ratio == pytest.approx(0.894, abs=1e-3)
    assert abs(depth_ratio - 1.0) > 10 * abs(raw_ratio - 1.0)


def test_default_tail_config_is_per_variable():
    cb = _make_callback(_FakeTeacherModel(["a"]))
    assert cb._tail_directions["PRMSL"] == "lower"
    assert cb._tail_references["PRMSL"] == pytest.approx(1000.0)
    assert {
        "PRMSL",
        "PRATEsfc",
        "eastward_wind_at_ten_meters",
        "northward_wind_at_ten_meters",
    } <= set(cb._tail_hist_ranges)
    # Unlisted variables default to the upper tail at query time.
    assert cb._tail_directions.get("PRATEsfc", "upper") == "upper"


def test_invalid_tail_direction_rejected():
    with pytest.raises(ValueError, match="upper.*lower"):
        BestStudentCheckpointCallback(
            val_dataset_path="x.zarr",
            coarse_val_data=None,  # type: ignore[arg-type]
            teacher_model=_FakeTeacherModel(["a"]),  # type: ignore[arg-type]
            best_checkpoint_path="x.ckpt",
            tail_directions={"a": "sideways"},
            validation_mode="from_noise",
        )
