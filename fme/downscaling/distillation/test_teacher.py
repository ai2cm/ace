# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for AceDiffusionTeacher."""

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.downscaling.data import StaticInputs
from fme.downscaling.distillation.fastgen_teacher import AceDiffusionTeacher
from fme.downscaling.models import DiffusionModelConfig, PairedNormalizationConfig
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector


def _get_monotonic_coordinate(size: int, stop: float) -> torch.Tensor:
    bounds = torch.linspace(0, stop, size + 1)
    return (bounds[:-1] + bounds[1:]) / 2


def _make_fine_coords(fine_shape: tuple[int, int]) -> LatLonCoordinates:
    lat_size, lon_size = fine_shape
    return LatLonCoordinates(
        lat=_get_monotonic_coordinate(lat_size, stop=lat_size),
        lon=_get_monotonic_coordinate(lon_size, stop=lon_size),
    )


def _build_small_model(coarse_shape=(8, 16), fine_shape=(16, 32)):
    normalizer = PairedNormalizationConfig(
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
        NormalizationConfig(means={"x": 0.0}, stds={"x": 1.0}),
    )
    fine_coords = _make_fine_coords(fine_shape)
    static_inputs = StaticInputs(fields=[], coords=fine_coords)
    return DiffusionModelConfig(
        module=DiffusionModuleRegistrySelector(
            "unet_diffusion_song", {"model_channels": 4}
        ),
        loss=LossConfig(type="MSE"),
        in_names=["x"],
        out_names=["x"],
        normalization=normalizer,
        p_mean=-1.0,
        p_std=1.0,
        sigma_min=0.1,
        sigma_max=1.0,
        churn=0.5,
        num_diffusion_generation_steps=3,
        predict_residual=False,
        use_fine_topography=False,
    ).build(
        coarse_shape,
        downscale_factor=2,
        full_fine_coords=fine_coords,
        static_inputs=static_inputs,
    )


def test_teacher_forward_shape():
    """AceDiffusionTeacher.forward() returns an x0 tensor of the same shape as x_t."""
    model = _build_small_model()
    teacher = AceDiffusionTeacher(model)

    B, C, H, W = 2, 1, 16, 32
    C_cond = 1  # one coarse channel, no topography, interpolated to fine
    x_t = torch.randn(B, C, H, W)
    t = torch.full((B,), 0.5)
    condition = torch.randn(B, C_cond, H, W)

    x0_hat = teacher(x_t, t, condition=condition)
    assert x0_hat.shape == (B, C, H, W)


def test_teacher_freeze_unfreeze():
    """freeze() / unfreeze() toggle grad computation on the underlying module."""
    model = _build_small_model()
    teacher = AceDiffusionTeacher(model)

    # Default: unfrozen
    assert any(p.requires_grad for p in teacher._ace_module.parameters())

    teacher.freeze()
    assert not any(p.requires_grad for p in teacher._ace_module.parameters())

    teacher.unfreeze()
    assert any(p.requires_grad for p in teacher._ace_module.parameters())


def test_teacher_deepcopy_independent():
    """deepcopy creates an independent copy with independent parameter gradients."""
    import copy

    teacher = AceDiffusionTeacher(_build_small_model())
    teacher_copy = copy.deepcopy(teacher)

    teacher.freeze()

    # Original is frozen; copy is still trainable
    assert not any(p.requires_grad for p in teacher._ace_module.parameters())
    assert any(p.requires_grad for p in teacher_copy._ace_module.parameters())

    # Forward outputs are numerically identical on the same input
    B, C, H, W, C_cond = 1, 1, 16, 32, 1
    x_t = torch.randn(B, C, H, W)
    t = torch.full((B,), 0.5)
    condition = torch.randn(B, C_cond, H, W)
    with torch.no_grad():
        out_orig = teacher(x_t, t, condition=condition)
        out_copy = teacher_copy(x_t, t, condition=condition)
    assert torch.allclose(out_orig, out_copy)


def test_lazy_deepcopy_roundtrip():
    """instantiate() on a LazyCall factory returns independent deepcopied objects."""
    import copy

    from fastgen.utils import LazyCall as L
    from fastgen.utils import instantiate

    def _copy_ace_teacher(teacher: AceDiffusionTeacher) -> AceDiffusionTeacher:
        return copy.deepcopy(teacher)

    teacher = AceDiffusionTeacher(_build_small_model())
    lazy_net = L(_copy_ace_teacher)(teacher=teacher)

    student = instantiate(lazy_net)
    teacher_copy = instantiate(lazy_net)

    # Both are independent AceDiffusionTeacher instances
    assert isinstance(student, AceDiffusionTeacher)
    assert isinstance(teacher_copy, AceDiffusionTeacher)
    assert student is not teacher_copy
    assert student is not teacher

    # Freeze one does not affect the other
    teacher_copy.freeze()
    assert any(p.requires_grad for p in student._ace_module.parameters())
