# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
FastGen callback that evaluates the distilled student against a pre-saved
teacher validation dataset and writes the best ACE-format checkpoint.

The expensive teacher sampler is **not** re-run during training — it was
already run by ``generate_val_dataset.py`` and its outputs are loaded from
the zarr store.  At each checkpoint event, only the cheap student forward
pass runs.

Metric
------
Mean CRPS across variables, treating both the student and teacher outputs as
ensembles.  For each (time, variable, lat, lon) cell:

    CRPS = E|X_student - X_teacher| - ½ E|X_student - X_student'|

where X_student, X_student' are independent draws from the student ensemble
and X_teacher is a draw from the (pre-saved) teacher ensemble.  Averaging
over the teacher ensemble gives the energy-score formulation:

    accuracy term : mean_{i,j} |s_i - t_j|   (n_student × n_teacher pairs)
    spread  term  : mean_{i,j} |s_i - s_j| / 2  (n_student × n_student pairs)

This is a proper scoring rule that rewards both accuracy *and* calibrated
spread, which makes it method-agnostic: it works for SFT (where spread comes
from the diffusion sampler) and for DMD2 (where we care whether the student's
collapsed mode is at least centred on the teacher's distribution).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import xarray as xr

    from fme.downscaling.data.datasets import GriddedData
    from fme.downscaling.distillation.fastgen_teacher import AceDiffusionTeacher
    from fme.downscaling.models import DiffusionModel


def _crps_ensemble(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    """Pixel-wise CRPS treating both inputs as ensembles.

    Args:
        student: ``(n_student, ...)`` student ensemble members.
        teacher: ``(n_teacher, ...)`` teacher ensemble members (treated as
            draws from the verification distribution).

    Returns:
        ``(...)`` CRPS averaged over all student/teacher pairs, in the same
        units as the inputs.  Non-negative for a proper scoring rule.
    """
    # accuracy: mean_{i,j} |s_i - t_j|
    accuracy = (student.unsqueeze(1) - teacher.unsqueeze(0)).abs().mean(dim=(0, 1))
    # spread: mean_{i,j} |s_i - s_j| / 2  (diagonal is 0)
    spread = (student.unsqueeze(1) - student.unsqueeze(0)).abs().mean(dim=(0, 1)) / 2
    return accuracy - spread


class BestStudentCheckpointCallback:
    """Save an ACE-format student checkpoint whenever validation CRPS improves.

    Injected directly into ``Trainer.callbacks._callbacks`` (same pattern as
    ``_CheckpointPruner``) so no FastGen ``_target_`` DictConfig entry is
    needed.  Only rank-0 performs IO; other ranks skip silently.

    At each checkpoint event the student network (``model.net``) is called
    directly to generate samples from the coarse validation conditions.
    Those samples are compared against the pre-saved teacher zarr using CRPS.
    The teacher model itself is **not** re-run during training.

    Args:
        val_dataset_path: Path to a zarr store produced by
            ``fme.downscaling.inference`` with dims
            ``(time, ensemble, latitude, longitude)``.
        coarse_val_data: Built ``GriddedData`` for the coarse validation
            split.  Must cover the same time steps as *val_dataset_path*.
        teacher_model: The fully loaded teacher ``DiffusionModel``.  Used
            only for its normalizer, packer, and static-inputs metadata —
            its weights are never modified.
        best_checkpoint_path: Destination ``.ckpt`` path for the best student
            checkpoint (ACE format, loadable by ``CheckpointModelConfig``).
        n_student_samples: Number of student ensemble members to draw per
            time step.  More members give a better CRPS estimate; 4 is a
            reasonable default.
    """

    def __init__(
        self,
        val_dataset_path: str,
        coarse_val_data: GriddedData,
        teacher_model: DiffusionModel,
        best_checkpoint_path: str,
        n_student_samples: int = 4,
    ) -> None:
        self._val_dataset_path = val_dataset_path
        self._coarse_val_data = coarse_val_data
        self._teacher_model = teacher_model
        self._best_checkpoint_path = best_checkpoint_path
        self._n_student_samples = n_student_samples
        self._best_crps = float("inf")
        self._teacher_ds: xr.Dataset | None = None

    # ------------------------------------------------------------------
    # FastGen callback interface — only on_save_checkpoint_success does work.
    # ------------------------------------------------------------------

    def on_save_checkpoint_success(
        self, model=None, iteration: int = 0, path: str = ""
    ) -> None:
        try:
            from fastgen.utils.distributed import is_rank0
        except ImportError:
            return
        if not is_rank0() or model is None:
            return

        import fastgen.utils.logging_utils as logger

        student: AceDiffusionTeacher = model.net
        crps = self._compute_validation_crps(student)

        if crps < self._best_crps:
            self._best_crps = crps
            from fme.downscaling.distillation.student_checkpoint import (
                save_student_checkpoint,
            )

            save_student_checkpoint(
                student_module=student._ace_module,
                teacher=self._teacher_model,
                path=self._best_checkpoint_path,
            )
            logger.info(
                f"[BestStudentCallback] iteration={iteration} CRPS={crps:.6f} "
                f"(new best) → {self._best_checkpoint_path}"
            )
        else:
            logger.info(
                f"[BestStudentCallback] iteration={iteration} CRPS={crps:.6f} "
                f"(best={self._best_crps:.6f}, no improvement)"
            )

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_teacher_ds(self) -> xr.Dataset:
        if self._teacher_ds is None:
            import xarray as xr

            self._teacher_ds = xr.open_zarr(self._val_dataset_path)
        return self._teacher_ds

    @torch.no_grad()
    def _compute_validation_crps(self, student: AceDiffusionTeacher) -> float:
        teacher_ds = self._load_teacher_ds()
        teacher_model = self._teacher_model
        n = self._n_student_samples

        crps_sum: dict[str, float] = {}
        count: dict[str, int] = {}

        for batch in self._coarse_val_data.get_generator():
            # Build condition tensor from coarse inputs (same path as training).
            static_inputs = teacher_model._subset_static_if_available(batch)
            condition = teacher_model._get_input_from_coarse(batch.data, static_inputs)
            # condition: (B, C_cond, H_fine, W_fine)

            B, _, H, W = condition.shape
            C_out = len(teacher_model.out_packer.names)

            # Draw n student samples by vectorising over the batch dimension.
            condition_rep = condition.repeat_interleave(n, dim=0)  # (B*n, C_cond, H, W)
            noise = torch.randn(B * n, C_out, H, W, device=condition.device)

            # sample() returns normalized outputs: (B*n, C_out, H, W)
            output_norm = student.sample(noise, condition_rep)
            output_norm = output_norm.reshape(B, n, C_out, H, W)

            # Unpack channel dim (axis=-3) then denormalize → physical units.
            # out_packer.unpack on (B, n, C_out, H, W) with axis=-3 gives
            # {var: (B, n, H, W)}.
            output = teacher_model.normalizer.fine.denormalize(
                teacher_model.out_packer.unpack(output_norm, axis=-3)
            )

            times = batch.time.values  # (B,) numpy time stamps

            for var, student_tensor in output.items():
                if var not in teacher_ds:
                    continue

                # teacher_batch: (B, n_teacher, H, W)
                teacher_np = teacher_ds[var].sel(time=times).values
                teacher_batch = torch.from_numpy(teacher_np).to(
                    student_tensor.device, dtype=torch.float32
                )

                # CRPS per batch element, then accumulate spatial sum.
                for i in range(B):
                    crps_map = _crps_ensemble(student_tensor[i], teacher_batch[i])
                    crps_sum[var] = crps_sum.get(var, 0.0) + float(crps_map.sum())
                    count[var] = count.get(var, 0) + crps_map.numel()

        if not crps_sum:
            return float("inf")

        return float(np.mean([crps_sum[v] / count[v] for v in crps_sum]))
