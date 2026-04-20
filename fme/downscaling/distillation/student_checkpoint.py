# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
Bridge for saving a distilled student network in ACE's checkpoint format so
the existing ``CheckpointModelConfig.build()`` / ``DiffusionModel.from_state()``
/ ``PatchPredictor`` / ``Evaluator`` pipeline can load and evaluate it without
modification.

Checkpoint format (mirrors ``DiffusionModel.get_state()``):
    {
        "model": {
            "config": DiffusionModelConfig.get_state(),
            "module": state_dict of the bare (non-DDP) denoiser,
            "coarse_shape": (H_coarse, W_coarse),
            "downscale_factor": int,
            "full_fine_coords": {"lat": tensor, "lon": tensor},
            "static_inputs": StaticInputs.get_state() | None,
        }
    }

The student's weight dict overwrites the "module" entry; everything else is
inherited from the teacher so the downstream pipeline sees the same grid,
variable names, and normalisation stats.

Usage
-----
    from fme.downscaling.distillation.student_checkpoint import save_student_checkpoint

    # After distillation training is complete:
    save_student_checkpoint(
        student_module=fastgen_model.net._ace_module,  # unwrapped bare denoiser
        teacher=teacher_diffusion_model,
        path="student.ckpt",
        num_sampling_steps=4,           # override for fast inference
    )
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from fme.downscaling.models import DiffusionModel


def save_student_checkpoint(
    student_module: torch.nn.Module,
    teacher: DiffusionModel,
    path: str | pathlib.Path,
    num_sampling_steps: int | None = None,
) -> None:
    """Save a distilled student in ACE checkpoint format.

    The student checkpoint is compatible with ``CheckpointModelConfig.build()``
    and ``DiffusionModel.from_state()``.  Grid, normalisation, and variable
    metadata are inherited from the teacher; only the denoiser weights differ.

    Args:
        student_module: The bare (non-DDP) ``torch.nn.Module`` that has been
            distilled.  Its ``state_dict()`` replaces the teacher's weights.
        teacher: The fully loaded teacher ``DiffusionModel`` (provides grid,
            config, normalisation, static inputs).
        path: Destination path for the ``.ckpt`` file (created via
            ``torch.save``).
        num_sampling_steps: If provided, overrides
            ``num_diffusion_generation_steps`` in the saved config so that
            ACE's ``stochastic_sampler`` uses the student's reduced step count
            (e.g. 2 or 4) when called through the existing inference pipeline.
    """
    state: dict[str, Any] = dict(teacher.get_state())

    # Overwrite network weights with the distilled student.
    state["module"] = student_module.state_dict()

    # Optionally reduce sampler steps in the saved config.
    if num_sampling_steps is not None:
        state["config"]["num_diffusion_generation_steps"] = num_sampling_steps

    torch.save({"model": state}, path)


def load_student_module_into_teacher(
    path: str | pathlib.Path,
    teacher: DiffusionModel,
) -> None:
    """Load student weights from a checkpoint into an existing DiffusionModel.

    Useful for quick evaluation: load the teacher from its own checkpoint,
    then hot-swap its denoiser weights with the distilled student.

    Args:
        path: Path to a checkpoint produced by ``save_student_checkpoint``.
        teacher: The ``DiffusionModel`` whose ``.module`` weights will be
            replaced in-place.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    ckpt = torch.load(path, map_location="cpu")
    student_weights = ckpt["model"]["module"]

    raw = teacher.module.module if isinstance(teacher.module, DDP) else teacher.module
    raw.load_state_dict(student_weights, strict=True)
