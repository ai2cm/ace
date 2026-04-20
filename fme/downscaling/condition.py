# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
Standalone conditioning tensor builder extracted from
DiffusionModel._get_input_from_coarse.

Both the teacher training path and the student inference path must consume
bit-identical conditioning tensors, so this function is the single source of truth.
"""

import torch

from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
from fme.core.typing_ import TensorMapping
from fme.downscaling.data import StaticInputs
from fme.downscaling.metrics_and_maths import filter_tensor_mapping, interpolate


def build_condition_tensor(
    coarse: TensorMapping,
    packer: Packer,
    coarse_normalizer: StandardNormalizer,
    downscale_factor: int,
    use_fine_topography: bool,
    interpolate_input: bool,
    static_inputs: StaticInputs | None = None,
    channel_axis: int = -3,
) -> torch.Tensor:
    """Build the dense conditioning tensor from coarse-resolution inputs.

    This is the canonical implementation shared by the teacher training loop and
    any student that must see bit-identical conditioning. It mirrors
    DiffusionModel._get_input_from_coarse exactly.

    Args:
        coarse: Mapping of variable names to coarse-resolution tensors.
        packer: Packer for the input variable names (determines channel order).
        coarse_normalizer: Normalizer applied to coarse inputs.
        downscale_factor: Integer upscaling factor (coarse → fine).
        use_fine_topography: Whether to concatenate static topography channels.
        interpolate_input: Whether to return the interpolated tensor. When False
            (i.e. the underlying network consumes un-interpolated coarse inputs),
            return the normalized coarse tensor directly.
        static_inputs: Optional static fields to concatenate (e.g. fine topography).
        channel_axis: Axis along which channels are stacked (default -3 = dim 1).

    Returns:
        Conditioning tensor of shape [B, C_cond, H_fine, W_fine].
    """
    inputs = filter_tensor_mapping(coarse, packer.names)
    normalized = packer.pack(coarse_normalizer.normalize(inputs), axis=channel_axis)
    interpolated = interpolate(normalized, downscale_factor)

    if use_fine_topography and static_inputs is not None:
        expected_shape = interpolated.shape[-2:]
        if static_inputs.shape != expected_shape:
            raise ValueError(
                f"Subsetted static input shape {static_inputs.shape} does not "
                f"match expected fine spatial shape {expected_shape}."
            )
        n_batches = normalized.shape[0]
        fields: list[torch.Tensor] = [interpolated]
        for field in static_inputs.fields:
            static_field = field.data.unsqueeze(0).repeat(n_batches, 1, 1)
            static_field = static_field.unsqueeze(channel_axis)
            fields.append(static_field)
        interpolated = torch.concat(fields, dim=channel_axis)

    if interpolate_input:
        return interpolated
    return normalized
