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
    needed.  Validation is sharded across all ranks (each rank consumes
    ``1/world_size`` of the batches and the per-variable sums are
    ``all_reduce``-d before averaging), so all ranks must enter this hook in
    lockstep — otherwise the next training step's NCCL collective deadlocks
    while rank-0 is busy validating.  Only rank-0 writes the checkpoint.

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
        coarse_patch_yx: Coarse-resolution spatial patch size ``(H, W)`` to
            use when iterating over the validation domain.  Should match the
            patch size used during training so that each batch fed to
            ``student.sample()`` has the expected spatial dimensions.  When
            ``None`` the full validation domain is used in a single batch
            (only valid when the domain already fits the model's input shape).
        n_student_samples: Number of student ensemble members to draw per
            time step.  More members give a better CRPS estimate; 4 is a
            reasonable default.

    To run validation on less data, thin at the data-config level via
    ``subset.step`` in the val YAML (see ``fme.core.dataset.time.TimeSlice``).
    That trims time indices before the patched generator expands them, which
    gives a cleaner sub-sample than thinning after patching.
    """

    def __init__(
        self,
        val_dataset_path: str,
        coarse_val_data: GriddedData,
        teacher_model: DiffusionModel,
        best_checkpoint_path: str,
        coarse_patch_yx: tuple[int, int] | None = None,
        n_student_samples: int = 4,
    ) -> None:
        self._val_dataset_path = val_dataset_path
        self._coarse_val_data = coarse_val_data
        self._teacher_model = teacher_model
        self._best_checkpoint_path = best_checkpoint_path
        self._coarse_patch_yx = coarse_patch_yx
        self._n_student_samples = n_student_samples
        self._best_crps = float("inf")
        self._teacher_ds: xr.Dataset | None = None

    # ------------------------------------------------------------------
    # FastGen callback interface — only on_save_checkpoint_success does work.
    # ------------------------------------------------------------------

    def on_save_checkpoint_success(
        self, model=None, iteration: int = 0, path: str = ""
    ) -> None:
        if model is None:
            return
        try:
            from fastgen.utils.distributed import is_rank0
        except ImportError:
            return

        import fastgen.utils.logging_utils as logger

        student: AceDiffusionTeacher = model.net
        # Validation runs on ALL ranks (sharded) — see class docstring.
        crps = self._compute_validation_crps(student)

        if not is_rank0():
            return

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
        import torch.distributed as dist

        teacher_ds = self._load_teacher_ds()
        teacher_model = self._teacher_model
        n = self._n_student_samples

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        if self._coarse_patch_yx is not None:
            batch_iter = self._coarse_val_data.get_patched_generator(
                self._coarse_patch_yx, drop_partial_patches=True
            )
        else:
            batch_iter = self._coarse_val_data.get_generator()

        crps_sum: dict[str, float] = {}
        count: dict[str, int] = {}

        teacher_time_index = teacher_ds.indexes["time"]
        try:
            import fastgen.utils.logging_utils as logger

            _log = logger.info
        except ImportError:
            import logging

            _log = logging.getLogger(__name__).info

        if rank == 0:
            _log(
                f"[BestStudentCallback] teacher_ds time index: "
                f"n={len(teacher_time_index)}, "
                f"dtype={type(teacher_time_index).__name__}, "
                f"first={teacher_time_index[0]!r}, last={teacher_time_index[-1]!r}"
            )
            _log(f"[BestStudentCallback] sharding: rank={rank}/{world_size}")

        global_batch_idx = -1
        n_processed = 0
        n_skipped_batches = 0
        n_skipped_times = 0
        first_batch_logged = False
        for batch in batch_iter:
            global_batch_idx += 1

            # Shard batches across ranks so every rank consumes a disjoint
            # subset of the iterator.
            if global_batch_idx % world_size != rank:
                continue

            times = batch.time.values  # (B,) numpy time stamps

            # Filter to times present in the teacher zarr; skip the batch if
            # none overlap. Avoids wasted student sampling on unmatched times.
            keep_mask = np.array([t in teacher_time_index for t in times])
            if not first_batch_logged:
                first_batch_logged = True
                _log(
                    f"[BestStudentCallback] rank={rank} first batch "
                    f"(global_idx={global_batch_idx}) times: "
                    f"dtype={times.dtype}, sample={list(times[:4])!r}, "
                    f"keep_mask={keep_mask.tolist()}"
                )
            if not keep_mask.all():
                n_skipped_times += int((~keep_mask).sum())
            if not keep_mask.any():
                n_skipped_batches += 1
                continue
            n_processed += 1

            # Build condition tensor from coarse inputs (same path as training).
            static_inputs = teacher_model._subset_static_if_available(batch)
            condition = teacher_model._get_input_from_coarse(batch.data, static_inputs)
            # condition: (B, C_cond, H_fine, W_fine)

            if not keep_mask.all():
                keep_idx = torch.from_numpy(np.flatnonzero(keep_mask)).to(
                    condition.device
                )
                condition = condition.index_select(0, keep_idx)
                times = times[keep_mask]

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
            # Fine-resolution lat/lon for this (possibly patched) batch.
            fine_coords = teacher_model.get_fine_coords_for_batch(batch)
            fine_lats = fine_coords.lat.cpu().numpy()
            fine_lons = fine_coords.lon.cpu().numpy()

            for var, student_tensor in output.items():
                if var not in teacher_ds:
                    continue

                # teacher_batch: (B, n_teacher, H_patch, W_patch)
                teacher_np = (
                    teacher_ds[var]
                    .sel(time=times)
                    .sel(latitude=fine_lats, longitude=fine_lons, method="nearest")
                    .values
                )
                teacher_batch = torch.from_numpy(teacher_np).to(
                    student_tensor.device, dtype=torch.float32
                )

                # CRPS per batch element, then accumulate spatial sum.
                for i in range(B):
                    crps_map = _crps_ensemble(student_tensor[i], teacher_batch[i])
                    crps_sum[var] = crps_sum.get(var, 0.0) + float(crps_map.sum())
                    count[var] = count.get(var, 0) + crps_map.numel()

        # Reduce per-variable sums/counts across ranks. The variable order is
        # taken from the teacher's out_packer ∩ teacher_ds so it's identical on
        # every rank, even if a given rank's shard processed no batches.
        global_keys = [
            name
            for name in teacher_model.out_packer.names
            if name in teacher_ds.data_vars
        ]
        device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        sums = torch.tensor(
            [crps_sum.get(k, 0.0) for k in global_keys],
            device=device,
            dtype=torch.float64,
        )
        counts = torch.tensor(
            [count.get(k, 0) for k in global_keys],
            device=device,
            dtype=torch.float64,
        )
        local_stats = torch.tensor(
            [n_processed, n_skipped_batches, n_skipped_times],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

        if rank == 0:
            total_processed, total_skipped_batches, total_skipped_times = (
                int(local_stats[0].item()),
                int(local_stats[1].item()),
                int(local_stats[2].item()),
            )
            _log(
                f"[BestStudentCallback] validation summary: "
                f"batches_processed={total_processed} (across {world_size} ranks), "
                f"batches_skipped={total_skipped_batches}, "
                f"unmatched_times={total_skipped_times}"
            )

        per_var = []
        for i, _var in enumerate(global_keys):
            c = float(counts[i].item())
            if c > 0:
                per_var.append(float(sums[i].item()) / c)
        if not per_var:
            return float("inf")
        return float(np.mean(per_var))
