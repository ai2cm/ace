# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone build_condition_tensor helper."""

import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
from fme.downscaling.condition import build_condition_tensor
from fme.downscaling.data import StaticInput, StaticInputs


def make_identity_normalizer(names):
    return StandardNormalizer(
        means={n: torch.tensor(0.0) for n in names},
        stds={n: torch.tensor(1.0) for n in names},
    )


def make_static_inputs(shape):
    coords = LatLonCoordinates(
        lat=torch.linspace(-45, 45, shape[0]),
        lon=torch.linspace(0, 90, shape[1]),
    )
    return StaticInputs(fields=[StaticInput(data=torch.ones(shape))], coords=coords)


# -------------------------------------------------------------------------


def test_shape_no_topography():
    """Output is [B, C_in, H_fine, W_fine] when interpolate_input=True."""
    B, coarse_h, coarse_w = 3, 4, 8
    factor = 2
    in_names = ["u", "v"]
    packer = Packer(in_names)
    normalizer = make_identity_normalizer(in_names)
    coarse = {n: torch.randn(B, coarse_h, coarse_w) for n in in_names}

    out = build_condition_tensor(
        coarse=coarse,
        packer=packer,
        coarse_normalizer=normalizer,
        downscale_factor=factor,
        use_fine_topography=False,
        interpolate_input=True,
    )

    assert out.shape == (B, len(in_names), coarse_h * factor, coarse_w * factor)


def test_shape_with_topography():
    """With topography, output channels = C_in + num_static_fields."""
    B, coarse_h, coarse_w = 2, 4, 8
    factor = 2
    fine_h, fine_w = coarse_h * factor, coarse_w * factor
    in_names = ["x"]
    packer = Packer(in_names)
    normalizer = make_identity_normalizer(in_names)
    coarse = {"x": torch.randn(B, coarse_h, coarse_w)}
    static_inputs = make_static_inputs((fine_h, fine_w))

    out = build_condition_tensor(
        coarse=coarse,
        packer=packer,
        coarse_normalizer=normalizer,
        downscale_factor=factor,
        use_fine_topography=True,
        interpolate_input=True,
        static_inputs=static_inputs,
    )

    assert out.shape == (B, 2, fine_h, fine_w)  # 1 coarse + 1 topo channel


def test_no_interpolation_returns_normalized_coarse():
    """When interpolate_input=False, the normalized coarse tensor is returned."""
    B, coarse_h, coarse_w = 2, 4, 8
    in_names = ["x"]
    packer = Packer(in_names)
    normalizer = make_identity_normalizer(in_names)
    coarse = {"x": torch.randn(B, coarse_h, coarse_w)}

    out = build_condition_tensor(
        coarse=coarse,
        packer=packer,
        coarse_normalizer=normalizer,
        downscale_factor=2,
        use_fine_topography=False,
        interpolate_input=False,
    )

    # Should have coarse spatial dims, not fine
    assert out.shape == (B, 1, coarse_h, coarse_w)


def test_normalization_applied():
    """Non-identity normalizer values flow through correctly."""
    B = 2
    in_names = ["x"]
    packer = Packer(in_names)
    # Normalizer shifts by 1 and scales by 0.5: (x - 1) / 0.5
    normalizer = StandardNormalizer(
        means={"x": torch.tensor(1.0)}, stds={"x": torch.tensor(0.5)}
    )
    coarse = {"x": torch.ones(B, 4, 8)}  # all ones

    out = build_condition_tensor(
        coarse=coarse,
        packer=packer,
        coarse_normalizer=normalizer,
        downscale_factor=1,
        use_fine_topography=False,
        interpolate_input=True,
    )

    # (1 - 1) / 0.5 = 0 everywhere
    assert torch.allclose(out, torch.zeros_like(out))


def test_topography_shape_mismatch_raises():
    """Mismatched static input shape raises ValueError."""
    B, coarse_h, coarse_w = 2, 4, 8
    factor = 2
    in_names = ["x"]
    packer = Packer(in_names)
    normalizer = make_identity_normalizer(in_names)
    coarse = {"x": torch.randn(B, coarse_h, coarse_w)}
    # Static inputs with wrong fine shape
    wrong_shape = (coarse_h * factor + 1, coarse_w * factor)
    static_inputs = make_static_inputs(wrong_shape)

    with pytest.raises(ValueError, match="shape"):
        build_condition_tensor(
            coarse=coarse,
            packer=packer,
            coarse_normalizer=normalizer,
            downscale_factor=factor,
            use_fine_topography=True,
            interpolate_input=True,
            static_inputs=static_inputs,
        )


def test_multi_static_fields():
    """Multiple static fields all appear in the output channels."""
    B, coarse_h, coarse_w = 2, 4, 8
    factor = 2
    fine_h, fine_w = coarse_h * factor, coarse_w * factor
    in_names = ["x"]
    packer = Packer(in_names)
    normalizer = make_identity_normalizer(in_names)
    coarse = {"x": torch.randn(B, coarse_h, coarse_w)}
    coords = LatLonCoordinates(
        lat=torch.linspace(-45, 45, fine_h),
        lon=torch.linspace(0, 90, fine_w),
    )
    static_inputs = StaticInputs(
        fields=[
            StaticInput(data=torch.ones(fine_h, fine_w)),
            StaticInput(data=torch.zeros(fine_h, fine_w)),
        ],
        coords=coords,
    )

    out = build_condition_tensor(
        coarse=coarse,
        packer=packer,
        coarse_normalizer=normalizer,
        downscale_factor=factor,
        use_fine_topography=True,
        interpolate_input=True,
        static_inputs=static_inputs,
    )

    assert out.shape == (B, 3, fine_h, fine_w)  # 1 coarse + 2 topo channels
