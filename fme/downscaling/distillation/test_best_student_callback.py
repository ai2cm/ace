# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for BestStudentCheckpointCallback validation-mode plumbing.

These exercise the per-expert ``lo_renoise`` path and the shared student
sampling dispatch without constructing a real teacher/student (which would pull
in FastGen).  Lightweight fakes stand in for the teacher model's packer +
normalizer and for the student denoiser.
"""

from __future__ import annotations

import logging
import pathlib

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
from fme.downscaling.distillation.student_checkpoint import save_candidate_checkpoint


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
    frozen_lo_net: torch.nn.Module | None = None,
    frozen_lo_sample_steps: int = 2,
    best_checkpoint_path: str = "unused.ckpt",
    best_tail_checkpoint_path: str | None = None,
    best_spec_checkpoint_path: str | None = None,
    early_stop_patience: int | None = None,
    spec_patience_window: int = 5,
    combined_checkpoint_dir: str | None = None,
    combined_tolerance: float = 1.05,
    combined_improvement: float = 0.95,
    combined_keep: int = 3,
    state_checkpoint_path: str | None = None,
) -> BestStudentCheckpointCallback:
    return BestStudentCheckpointCallback(
        val_dataset_path="unused.zarr",
        coarse_val_data=None,  # type: ignore[arg-type]  # not touched in __init__
        teacher_model=teacher_model,  # type: ignore[arg-type]
        best_checkpoint_path=best_checkpoint_path,
        n_student_samples=n_student_samples,
        student_sample_steps=student_sample_steps,
        best_tail_checkpoint_path=best_tail_checkpoint_path,
        best_spec_checkpoint_path=best_spec_checkpoint_path,
        early_stop_patience=early_stop_patience,
        spec_patience_window=spec_patience_window,
        combined_checkpoint_dir=combined_checkpoint_dir,
        combined_tolerance=combined_tolerance,
        combined_improvement=combined_improvement,
        combined_keep=combined_keep,
        state_checkpoint_path=state_checkpoint_path,
        validation_mode=validation_mode,
        frozen_lo_net=frozen_lo_net,
        frozen_lo_sample_steps=frozen_lo_sample_steps,
    )


def test_validation_mode_rejects_unknown():
    tm = _FakeTeacherModel(["a"])
    with pytest.raises(ValueError, match="validation_mode"):
        _make_callback(tm, validation_mode="bogus")


def test_hi_cascade_requires_frozen_lo_net():
    tm = _FakeTeacherModel(["a"])
    with pytest.raises(ValueError, match="frozen_lo_net"):
        _make_callback(tm, validation_mode="hi_cascade", frozen_lo_net=None)


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


def test_sample_student_output_hi_cascade_shape():
    """hi_cascade cascades the trained Hi student (sigma [200, 2000]) through a
    frozen Lo net (sigma [0.005, 200]); the ensemble comes from n fresh noise
    draws, so the output is (B, n_student_samples, C_out, H, W)."""
    tm = _FakeTeacherModel(["a", "b"])
    C_out, C_cond = 2, 3
    frozen_lo = _TinyNet(C_out, C_cond).eval()
    cb = _make_callback(
        tm,
        validation_mode="hi_cascade",
        n_student_samples=3,
        student_sample_steps=1,
        frozen_lo_net=frozen_lo,
        frozen_lo_sample_steps=2,
    )
    # The Hi student spans the high segment; its sigma_min IS the boundary.
    student = _FakeStudent(_TinyNet(C_out, C_cond).eval(), 200.0, 2000.0)

    B, H, W = 2, 4, 4
    condition = torch.randn(B, C_cond, H, W)
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


def test_sample_student_output_hi_cascade_adds_residual_base():
    """hi_cascade output is a residual (both segment nets are residual), so the
    interpolated-coarse base is added back — a const base shifts every sample."""
    tm = _FakeTeacherModel(["a", "b"])
    C_out, C_cond = 2, 3
    frozen_lo = _TinyNet(C_out, C_cond).eval()
    cb = _make_callback(
        tm,
        validation_mode="hi_cascade",
        n_student_samples=2,
        student_sample_steps=1,
        frozen_lo_net=frozen_lo,
    )
    student = _FakeStudent(_TinyNet(C_out, C_cond).eval(), 200.0, 2000.0)
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
    torch.manual_seed(1)
    out_base = cb._sample_student_output(
        student,  # type: ignore[arg-type]
        condition,
        {},
        torch.full((B, C_out, H, W), 5.0),
        B,
        H,
        W,
        C_out,
    )
    torch.testing.assert_close(out_base - out_zero, torch.full_like(out_base, 5.0))


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


# ---------------------------------------------------------------------------
# Spectral selector + early stopping (spec-13).
#
# These drive ``_record_validation`` directly with canned metric dicts,
# bypassing the sharded validation pass and the rank-0 guard, and monkeypatch
# ``save_student_checkpoint`` so no checkpoints touch disk.
# ---------------------------------------------------------------------------


def _patch_save(monkeypatch) -> list[str]:
    """Record ``save_student_checkpoint`` destinations without touching disk."""
    calls: list[str] = []
    monkeypatch.setattr(
        "fme.downscaling.distillation.student_checkpoint.save_student_checkpoint",
        lambda student_module, teacher, path, **kwargs: calls.append(str(path)),
    )
    return calls


def _patch_save_touch(monkeypatch) -> list[str]:
    """Like ``_patch_save`` but creates empty files so glob-pruning is real."""
    calls: list[str] = []

    def _touch(student_module, teacher, path, **kwargs):
        pathlib.Path(path).touch()
        calls.append(str(path))

    monkeypatch.setattr(
        "fme.downscaling.distillation.student_checkpoint.save_student_checkpoint",
        _touch,
    )
    return calls


def _fake_student() -> _FakeStudent:
    # save_student_checkpoint is patched out, so the module is never invoked.
    return _FakeStudent(torch.nn.Identity(), sigma_min=0.01, sigma_max=1.0)


def _drive(cb, student, iteration, crps, spec, tail=None):
    """Run one validation with a scalar CRPS mean, spectral MAE, and tail ratio."""
    crps_by_var = {"a": crps, "mean": crps}
    spec_by_var = {"a": {"mae": spec}} if spec is not None else {}
    # 99.9999 is the default top percentile; tail_score == |tail - 1|.
    tail_by_pct = {99.9999: {"a": tail, "mean": tail}} if tail is not None else {}
    cb._record_validation(student, crps_by_var, tail_by_pct, spec_by_var, iteration)


def test_spec_selector_uses_rolling_median_not_spike(monkeypatch):
    calls = _patch_save(monkeypatch)
    cb = _make_callback(
        _FakeTeacherModel(["a"]),
        best_checkpoint_path="crps.ckpt",
        best_spec_checkpoint_path="spec.ckpt",
        spec_patience_window=3,
    )
    student = _fake_student()
    # A transient dip at step 2 must NOT win; the sustained 0.6 level from step 4
    # on is the real minimum of the rolling median.
    specs = [1.0, 1.0, 0.2, 1.0, 0.6, 0.6, 0.6]
    for i, s in enumerate(specs):
        _drive(cb, student, i, crps=5.0, spec=s)
    spec_saves = [p for p in calls if p == "spec.ckpt"]
    # Two sustained-median improvements: inf→1.0 at step 0, 1.0→0.6 at step 4.
    assert len(spec_saves) == 2
    # The transient 0.2 spike never became the selected best.
    assert cb._best_spec == pytest.approx(0.6)


def test_early_stop_triggers_after_patience(monkeypatch):
    _patch_save(monkeypatch)
    cb = _make_callback(
        _FakeTeacherModel(["a"]),
        best_spec_checkpoint_path="spec.ckpt",
        early_stop_patience=3,
        spec_patience_window=1,
    )
    student = _fake_student()
    # Step 0 improves (crps inf→5, spec inf→1); nothing improves afterward.
    _drive(cb, student, 0, crps=5.0, spec=1.0)
    assert not cb.should_stop()
    for i in (1, 2):
        _drive(cb, student, i, crps=5.0, spec=1.0)
        assert not cb.should_stop()
    # Third consecutive non-improving validation hits patience.
    _drive(cb, student, 3, crps=5.0, spec=1.0)
    assert cb.should_stop()


def test_early_stop_counter_resets_on_improvement(monkeypatch):
    _patch_save(monkeypatch)
    cb = _make_callback(
        _FakeTeacherModel(["a"]),
        best_spec_checkpoint_path="spec.ckpt",
        early_stop_patience=3,
        spec_patience_window=1,
    )
    student = _fake_student()
    # CRPS drops at step 3, resetting the counter before it reaches patience.
    for i, crps in enumerate([5.0, 5.0, 5.0, 4.0, 5.0, 5.0]):
        _drive(cb, student, i, crps=crps, spec=1.0)
    # Without the reset the counter would have hit 3 at step 3 and stopped.
    assert not cb.should_stop()
    assert cb._checks_since_improvement == 2


def test_early_stop_and_spec_disabled_by_default(monkeypatch):
    calls = _patch_save(monkeypatch)
    # No spec path, no patience → behavior matches today (CRPS/tail only).
    cb = _make_callback(_FakeTeacherModel(["a"]), best_checkpoint_path="crps.ckpt")
    student = _fake_student()
    for i in range(6):
        _drive(cb, student, i, crps=5.0, spec=1.0)
    assert not cb.should_stop()
    # No spectral checkpoint written; only the CRPS best, once, at step 0.
    assert "spec.ckpt" not in calls
    assert calls == ["crps.ckpt"]


def test_invalid_early_stop_patience_rejected():
    with pytest.raises(ValueError, match="early_stop_patience"):
        _make_callback(_FakeTeacherModel(["a"]), early_stop_patience=0)


def test_invalid_spec_patience_window_rejected():
    with pytest.raises(ValueError, match="spec_patience_window"):
        _make_callback(_FakeTeacherModel(["a"]), spec_patience_window=0)


# ---------------------------------------------------------------------------
# Combined tail+spectral candidate selector.
#
# All callbacks here set best_tail_checkpoint_path (the combined selector needs
# a finite tail best, which only that selector maintains) and
# spec_patience_window=1 so spec_median equals the raw per-validation value.
# ---------------------------------------------------------------------------


def _make_combined_callback(tmp_path, **kwargs) -> BestStudentCheckpointCallback:
    return _make_callback(
        _FakeTeacherModel(["a"]),
        best_checkpoint_path=str(tmp_path / "crps.ckpt"),
        best_tail_checkpoint_path=str(tmp_path / "tail.ckpt"),
        best_spec_checkpoint_path=str(tmp_path / "spec.ckpt"),
        spec_patience_window=1,
        combined_checkpoint_dir=str(tmp_path),
        **kwargs,
    )


def test_combined_fires_on_tail_near_best_and_spec_substantial(monkeypatch, tmp_path):
    calls = _patch_save(monkeypatch)
    cb = _make_combined_callback(tmp_path)
    student = _fake_student()
    # Seed the bests: tail score 0.10, spec 1.0.  No candidate (bests were inf).
    _drive(cb, student, 0, crps=5.0, spec=1.0, tail=1.10)
    # Tail 0.104 <= 0.10 * 1.05 (near, not a new best); spec 0.90 <= 1.0 * 0.95.
    _drive(cb, student, 1, crps=5.0, spec=0.90, tail=1.104)
    combined = str(tmp_path / "best_student_combined_1.ckpt")
    assert combined in calls
    assert cb._combined_saves == 1
    # The tail selector itself did not re-save (tail did not improve).
    assert calls.count(str(tmp_path / "tail.ckpt")) == 1
    # Regression for the pre-update baseline: the spec selector already lowered
    # its best to 0.90 in the same validation, yet the candidate still fired.
    assert cb._best_spec == pytest.approx(0.90)


def test_combined_fires_on_spec_near_best_and_tail_substantial(monkeypatch, tmp_path):
    calls = _patch_save(monkeypatch)
    cb = _make_combined_callback(tmp_path)
    student = _fake_student()
    _drive(cb, student, 0, crps=5.0, spec=1.0, tail=1.10)
    # Tail 0.09 <= 0.10 * 0.95 (substantial); spec 1.04 <= 1.0 * 1.05 (near).
    _drive(cb, student, 1, crps=5.0, spec=1.04, tail=1.09)
    assert str(tmp_path / "best_student_combined_1.ckpt") in calls


def test_combined_requires_substantial_improvement(monkeypatch, tmp_path):
    calls = _patch_save(monkeypatch)
    cb = _make_combined_callback(tmp_path)
    student = _fake_student()
    _drive(cb, student, 0, crps=5.0, spec=1.0, tail=1.10)
    # Both metrics near-best (within 5%) but neither beats its best by 5%.
    _drive(cb, student, 1, crps=5.0, spec=0.98, tail=1.104)
    assert not any("best_student_combined" in p for p in calls)
    assert cb._combined_saves == 0


def test_combined_skips_until_both_bests_finite(monkeypatch, tmp_path):
    calls = _patch_save(monkeypatch)
    cb = _make_combined_callback(tmp_path)
    student = _fake_student()
    # Excellent first validation must not burn a candidate slot: with the
    # bests still at inf, any finite metrics would trivially qualify.
    _drive(cb, student, 0, crps=5.0, spec=0.5, tail=1.01)
    assert not any("best_student_combined" in p for p in calls)


def test_combined_pruning_keeps_most_recent(monkeypatch, tmp_path):
    _patch_save_touch(monkeypatch)
    cb = _make_combined_callback(tmp_path)
    student = _fake_student()
    # Seed, then five candidates: tail stays at its best (near) while spec
    # drops >= 5% each validation (substantial).
    for i, spec in enumerate([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]):
        _drive(cb, student, i, crps=5.0, spec=spec, tail=1.10)
    assert cb._combined_saves == 5
    remaining = sorted(p.name for p in tmp_path.glob("best_student_combined_*"))
    assert remaining == [
        "best_student_combined_3.ckpt",
        "best_student_combined_4.ckpt",
        "best_student_combined_5.ckpt",
    ]


def test_combined_prunes_preexisting_files_on_disk(monkeypatch, tmp_path):
    _patch_save_touch(monkeypatch)
    # Simulates resume: a candidate from before the restart is on disk but
    # unknown to the (reset) in-memory selector state.
    (tmp_path / "best_student_combined_0.ckpt").touch()
    cb = _make_combined_callback(tmp_path)
    student = _fake_student()
    for i, spec in enumerate([1.0, 0.9, 0.8, 0.7]):
        _drive(cb, student, i, crps=5.0, spec=spec, tail=1.10)
    remaining = sorted(p.name for p in tmp_path.glob("best_student_combined_*"))
    assert remaining == [
        "best_student_combined_1.ckpt",
        "best_student_combined_2.ckpt",
        "best_student_combined_3.ckpt",
    ]


def test_combined_disabled_by_default(monkeypatch, tmp_path):
    calls = _patch_save(monkeypatch)
    cb = _make_callback(
        _FakeTeacherModel(["a"]),
        best_checkpoint_path="crps.ckpt",
        best_tail_checkpoint_path="tail.ckpt",
        spec_patience_window=1,
    )
    student = _fake_student()
    _drive(cb, student, 0, crps=5.0, spec=1.0, tail=1.10)
    _drive(cb, student, 1, crps=5.0, spec=0.90, tail=1.104)
    # Combined-qualifying metrics, but no combined dir → save sequence is
    # unchanged from today: CRPS + tail at step 0 only.
    assert calls == ["crps.ckpt", "tail.ckpt"]


def test_invalid_combined_tolerance_rejected():
    with pytest.raises(ValueError, match="combined_tolerance"):
        _make_callback(_FakeTeacherModel(["a"]), combined_tolerance=0.9)


def test_invalid_combined_improvement_rejected():
    with pytest.raises(ValueError, match="combined_improvement"):
        _make_callback(_FakeTeacherModel(["a"]), combined_improvement=1.0)
    with pytest.raises(ValueError, match="combined_improvement"):
        _make_callback(_FakeTeacherModel(["a"]), combined_improvement=0.0)


def test_invalid_combined_keep_rejected():
    with pytest.raises(ValueError, match="combined_keep"):
        _make_callback(_FakeTeacherModel(["a"]), combined_keep=0)


def test_save_candidate_checkpoint_prunes_by_iteration_number(monkeypatch, tmp_path):
    _patch_save_touch(monkeypatch)
    (tmp_path / "best_student_combined_2.ckpt").touch()
    (tmp_path / "best_student_combined_9.ckpt").touch()
    path = save_candidate_checkpoint(
        student_module=torch.nn.Identity(),
        teacher=None,  # type: ignore[arg-type]  # patched save ignores it
        directory=tmp_path,
        iteration=10,
        keep=2,
    )
    assert path == tmp_path / "best_student_combined_10.ckpt"
    remaining = sorted(p.name for p in tmp_path.glob("best_student_combined_*"))
    # Numeric ordering: lexicographic sorting would have pruned "_10" instead.
    assert remaining == [
        "best_student_combined_10.ckpt",
        "best_student_combined_9.ckpt",
    ]


# ---------------------------------------------------------------------------
# Preemption-safe resume: the callback persists its selection state to a
# sidecar after every validation and reloads it at construction, so a restart
# does not reset the bests (silently overwriting best_student*.ckpt with a
# worse student) or the early-stop counter.
# ---------------------------------------------------------------------------


def _make_state_callback(
    tmp_path, state_path, **kwargs
) -> BestStudentCheckpointCallback:
    kwargs.setdefault("spec_patience_window", 1)
    return _make_callback(
        _FakeTeacherModel(["a"]),
        best_checkpoint_path=str(tmp_path / "crps.ckpt"),
        best_tail_checkpoint_path=str(tmp_path / "tail.ckpt"),
        best_spec_checkpoint_path=str(tmp_path / "spec.ckpt"),
        state_checkpoint_path=state_path,
        **kwargs,
    )


def test_state_round_trip_restores_bests(monkeypatch, tmp_path):
    _patch_save(monkeypatch)
    state_path = str(tmp_path / "state.pt")
    cb = _make_state_callback(tmp_path, state_path, early_stop_patience=3)
    student = _fake_student()
    _drive(cb, student, 0, crps=5.0, spec=1.0, tail=1.10)
    _drive(cb, student, 1, crps=5.0, spec=1.0, tail=1.10)  # no improvement

    restored = _make_state_callback(tmp_path, state_path, early_stop_patience=3)
    assert restored._best_crps == pytest.approx(5.0)
    assert restored._best_tail_score == pytest.approx(0.10)
    assert restored._best_spec == pytest.approx(1.0)
    assert restored._spec_history == [pytest.approx(1.0)]
    assert restored._checks_since_improvement == 1
    assert restored._stop_requested is False


def test_state_prevents_silent_overwrite_after_resume(monkeypatch, tmp_path):
    calls = _patch_save(monkeypatch)
    state_path = str(tmp_path / "state.pt")
    crps_ckpt = str(tmp_path / "crps.ckpt")
    student = _fake_student()

    cb = _make_state_callback(tmp_path, state_path)
    _drive(cb, student, 0, crps=3.0, spec=1.0, tail=1.10)  # best_crps -> 3.0

    # Resume: a worse validation must NOT overwrite best_student.ckpt.
    calls.clear()
    restored = _make_state_callback(tmp_path, state_path)
    _drive(restored, student, 1, crps=5.0, spec=1.0, tail=1.10)
    assert crps_ckpt not in calls

    # Contrast with the old behavior (no persistence): inf best -> overwrite.
    calls.clear()
    fresh = _make_state_callback(tmp_path, None)
    _drive(fresh, student, 1, crps=5.0, spec=1.0, tail=1.10)
    assert crps_ckpt in calls


def test_state_early_stop_counter_survives_resume(monkeypatch, tmp_path):
    _patch_save(monkeypatch)
    state_path = str(tmp_path / "state.pt")
    student = _fake_student()

    cb = _make_state_callback(tmp_path, state_path, early_stop_patience=3)
    _drive(cb, student, 0, crps=5.0, spec=1.0, tail=1.10)  # improves, checks=0
    _drive(cb, student, 1, crps=5.0, spec=1.0, tail=1.10)  # checks=1
    _drive(cb, student, 2, crps=5.0, spec=1.0, tail=1.10)  # checks=2
    assert not cb.should_stop()

    # A relaunch that would otherwise reset the counter to 0 must continue.
    restored = _make_state_callback(tmp_path, state_path, early_stop_patience=3)
    assert restored._checks_since_improvement == 2
    _drive(restored, student, 3, crps=5.0, spec=1.0, tail=1.10)  # checks=3 -> stop
    assert restored.should_stop()


def test_state_spec_history_trimmed_to_current_window(monkeypatch, tmp_path):
    _patch_save(monkeypatch)
    state_path = str(tmp_path / "state.pt")
    student = _fake_student()

    cb = _make_state_callback(tmp_path, state_path, spec_patience_window=5)
    for i, spec in enumerate([1.0, 0.9, 0.8, 0.7]):
        _drive(cb, student, i, crps=5.0, spec=spec, tail=1.10)
    assert cb._spec_history == [pytest.approx(v) for v in (1.0, 0.9, 0.8, 0.7)]

    # A relaunch with a smaller window keeps only the most recent values.
    restored = _make_state_callback(tmp_path, state_path, spec_patience_window=2)
    assert restored._spec_history == [pytest.approx(0.8), pytest.approx(0.7)]


def test_state_missing_sidecar_uses_defaults(tmp_path):
    cb = _make_state_callback(tmp_path, str(tmp_path / "does_not_exist.pt"))
    assert cb._best_crps == float("inf")
    assert cb._checks_since_improvement == 0


def test_state_corrupt_sidecar_warns_and_uses_defaults(tmp_path, caplog):
    state_path = tmp_path / "state.pt"
    state_path.write_bytes(b"not a torch checkpoint")
    with caplog.at_level(logging.WARNING):
        cb = _make_state_callback(tmp_path, str(state_path))
    assert cb._best_crps == float("inf")
    assert "could not load selection state" in caplog.text


def test_state_disabled_by_default(monkeypatch, tmp_path):
    _patch_save(monkeypatch)
    cb = _make_state_callback(tmp_path, None)
    student = _fake_student()
    _drive(cb, student, 0, crps=5.0, spec=1.0, tail=1.10)
    # No sidecar param -> nothing written, but get_state() is still a real dict.
    assert not any(tmp_path.glob("*.pt"))
    assert "_best_crps" in cb.get_state()


def test_get_state_is_a_real_method_not_getattr_stub(tmp_path):
    # Guards against the __getattr__ no-op lambda swallowing get_state (the
    # original bug: the callback's catch-all returned lambdas for any method).
    cb = _make_state_callback(tmp_path, None)
    state = cb.get_state()
    assert state is not None
    assert "_best_crps" in state and "_checks_since_improvement" in state
