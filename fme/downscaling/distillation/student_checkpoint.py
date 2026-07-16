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
    sampler_type: str = "fastgen",
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
            ACE's sampler uses the student's reduced step count (e.g. 2 or 4)
            when called through the existing inference pipeline.
        sampler_type: Sampler to bake into the saved config (``"fastgen"`` by
            default). FastGen-distilled students were trained against the
            predict-x0-then-renoise trajectory, so the ``"fastgen"`` sampler
            matches their training distribution.  Pass ``"heun"`` only if the
            student was trained to match the Heun trajectory.
    """
    state: dict[str, Any] = dict(teacher.get_state())

    # Overwrite network weights with the distilled student.  Add the
    # "module." prefix so the format matches DDP's state_dict() convention
    # expected by DiffusionModel.from_state() and CheckpointModelConfig.build().
    state["module"] = {"module." + k: v for k, v in student_module.state_dict().items()}

    state["config"]["sampler_type"] = sampler_type

    # Optionally reduce sampler steps in the saved config.
    if num_sampling_steps is not None:
        state["config"]["num_diffusion_generation_steps"] = num_sampling_steps

    torch.save({"model": state}, path)


def save_candidate_checkpoint(
    student_module: torch.nn.Module,
    teacher: DiffusionModel,
    directory: str | pathlib.Path,
    iteration: int,
    keep: int = 3,
    prefix: str = "best_student_combined",
    num_sampling_steps: int | None = None,
    sampler_type: str = "fastgen",
) -> pathlib.Path:
    """Save an iteration-stamped candidate checkpoint, keeping the newest few.

    Writes ``<directory>/<prefix>_<iteration>.ckpt`` via
    ``save_student_checkpoint`` and then prunes older ``<prefix>_*.ckpt``
    files so at most ``keep`` remain.  Pruning globs the directory (rather
    than tracking saved paths in memory) so it stays correct across training
    resumes, where in-memory selector state is reset but earlier candidate
    files may already exist on disk.

    Args:
        student_module: See ``save_student_checkpoint``.
        teacher: See ``save_student_checkpoint``.
        directory: Directory in which candidate checkpoints accumulate.
        iteration: Training iteration stamped into the filename; candidates
            are ranked (for pruning) by this number.
        keep: Number of most-recent candidates retained (>= 1).
        prefix: Filename prefix shared by all candidates in the rotation.
        num_sampling_steps: See ``save_student_checkpoint``.
        sampler_type: See ``save_student_checkpoint``.

    Returns:
        The path of the checkpoint that was written.
    """
    directory = pathlib.Path(directory)
    path = directory / f"{prefix}_{iteration}.ckpt"
    save_student_checkpoint(
        student_module=student_module,
        teacher=teacher,
        path=path,
        num_sampling_steps=num_sampling_steps,
        sampler_type=sampler_type,
    )
    candidates: list[tuple[int, pathlib.Path]] = []
    for existing in directory.glob(f"{prefix}_*.ckpt"):
        suffix = existing.stem[len(prefix) + 1 :]
        try:
            candidates.append((int(suffix), existing))
        except ValueError:
            continue
    candidates.sort(key=lambda pair: pair[0])
    for _, stale in candidates[:-keep] if keep > 0 else candidates:
        stale.unlink(missing_ok=True)
    return path


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

    # Strip the "module." prefix added by save_student_checkpoint for DDP format.
    if student_weights and next(iter(student_weights)).startswith("module."):
        student_weights = {k[len("module.") :]: v for k, v in student_weights.items()}

    raw = teacher.module.module if isinstance(teacher.module, DDP) else teacher.module
    raw.load_state_dict(student_weights, strict=True)
