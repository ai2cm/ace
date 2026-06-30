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


def _quantile_from_histogram(
    hist: torch.Tensor, lo: float, hi: float, q: float
) -> float | None:
    """Estimate the ``q``-quantile of the distribution described by ``hist``.

    ``hist[i]`` is the count of values that fell in the i-th equal-width bin
    over ``[lo, hi]``.  Returns ``None`` when the histogram is empty.  Uses
    linear interpolation within the containing bin so the estimate's
    resolution is finer than the bin width.
    """
    total = float(hist.sum().item())
    if total <= 0:
        return None
    cumsum = hist.cumsum(0)
    threshold = q * total
    # First bin index where cumsum >= threshold.
    idx_t = torch.searchsorted(cumsum, torch.tensor(threshold, device=cumsum.device))
    idx = int(idx_t.item())
    n_bins = hist.numel()
    if idx >= n_bins:
        return hi
    bin_width = (hi - lo) / n_bins
    cum_before = float(cumsum[idx - 1].item()) if idx > 0 else 0.0
    bin_count = float(hist[idx].item())
    if bin_count <= 0:
        return lo + idx * bin_width
    frac = (threshold - cum_before) / bin_count
    return lo + (idx + frac) * bin_width


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


# Per-variable tail config defaults.  The extreme that matters is variable-
# specific: deep low-pressure systems live in PRMSL's *lower* tail, while wind
# and precip extremes live in the *upper* tail.  Histogram ranges are in the
# variables' physical (denormalized) units and must bracket the full
# distribution so the CDF-integrated quantile is correct (PRMSL in hPa — see
# scripts/downscaling/plot_events.py UNITS).
_DEFAULT_TAIL_HIST_RANGES: dict[str, tuple[float, float]] = {
    "PRATEsfc": (0.0, 0.1),  # kg/m^2/s
    "PRMSL": (900.0, 1080.0),  # hPa
    "eastward_wind_at_ten_meters": (-70.0, 70.0),  # m/s
    "northward_wind_at_ten_meters": (-70.0, 70.0),  # m/s
}
# Variables not listed default to "upper".
_DEFAULT_TAIL_DIRECTIONS: dict[str, str] = {"PRMSL": "lower"}


def _tail_quantile_level(percentile: float, direction: str) -> float:
    """Map an extremeness ``percentile`` + tail ``direction`` to a CDF quantile.

    ``"upper"`` → ``percentile / 100`` (e.g. 99.99 → 0.9999); ``"lower"`` →
    its reflection ``1 - percentile / 100`` (e.g. 99.99 → 0.0001).
    """
    if direction == "upper":
        return percentile / 100.0
    if direction == "lower":
        return 1.0 - percentile / 100.0
    raise ValueError(f"tail direction must be 'upper' or 'lower', got {direction!r}")


def _normalized_mean(
    values_by_var: dict[str, float], scales: dict[str, float]
) -> float:
    """Mean over variables of ``value / scale`` (per-var ``scale``, default 1.0).

    Used to reduce per-variable CRPS (in physical units, wildly different
    magnitudes) into a single scale-free selection number, so e.g. precip
    (~1e-5) is not drowned out by PRMSL (~1e3).
    """
    normed = [v / scales.get(k, 1.0) for k, v in values_by_var.items()]
    return float(np.mean(normed)) if normed else float("inf")


def _tail_deviation_score(ratios_by_var: dict[str, float]) -> float:
    """Mean over variables of ``|ratio - 1|`` (0 = student tail matches teacher).

    Each variable's discrepancy counts, so over-prediction in one variable
    cannot cancel under-prediction in another (which a mean-of-ratios would
    allow).
    """
    devs = [abs(r - 1.0) for r in ratios_by_var.values()]
    return float(np.mean(devs)) if devs else float("inf")


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
        student_sample_steps: Number of student denoising steps used when
            sampling for validation.  Should match the distillation
            ``student_sample_steps`` so validation reflects the trajectory
            the student was actually trained on.  Defaults to 4.
        tail_percentiles: Tail extremeness levels to track per variable
            (default ``[99.99, 99.9999]``).  Applied per ``tail_directions``:
            an ``"upper"`` variable uses quantile ``pct/100``, a ``"lower"``
            variable its reflection ``1 - pct/100``.  Each is logged as
            ``val/tail_<pct>_<var>`` = student_pXX / target_pXX (values < 1
            mean the student under-predicts the tail extreme, > 1 over-
            predicts).  The (std-normalized, var-averaged) CRPS remains the
            ``best_checkpoint_path`` criterion; see
            ``best_tail_checkpoint_path`` for tail-driven selection.
        tail_hist_ranges: Per-variable ``(lo, hi)`` ranges (physical units)
            used to bin values for percentile estimation; must bracket the
            full distribution.  Variables not in this dict get the tail metric
            skipped (CRPS is still computed).  Defaults to
            ``_DEFAULT_TAIL_HIST_RANGES`` (precip/PRMSL/u10/v10).
        tail_directions: Per-variable tail side, ``"upper"`` or ``"lower"``.
            Defaults to ``_DEFAULT_TAIL_DIRECTIONS`` (PRMSL → ``"lower"`` for
            deep-low extremes, everything else → ``"upper"``).
        tail_hist_bins: Number of equal-width bins inside each variable's
            range (default 10000).  Resolution is ``(hi - lo) / tail_hist_bins``
            — for PRATEsfc that's 1e-5 kg/m²/s.
        best_tail_checkpoint_path: Optional destination path for a checkpoint
            selected by the tail metric rather than CRPS.  When provided,
            saves whenever the highest-percentile tail score improves, where
            the score is the mean over variables of ``|ratio - 1|`` (each
            variable's discrepancy counts; no cross-variable cancellation).
            Useful when CRPS has plateaued but tail fidelity is still evolving.
        validation_mode: How the student output ensemble is produced before
            comparing to the teacher zarr (see ``student_sampling``):

            - ``"from_noise"`` (default): an end-to-end student denoises fresh
              noise over its full sigma range to a clean x0.  Correct for a
              single full-range student.
            - ``"lo_renoise"``: a low-noise *segment* student is validated by
              re-noising the teacher target members to its ``sigma_max`` and
              denoising from there — no upstream student or live teacher
              needed.  Requires every output variable to be present in the
              teacher zarr (the target is re-noised, not just compared).  Use
              for per-expert Student-Lo.

            ``"hi_cascade"`` (Student-Hi through a frozen Lo) is not yet wired
            in here; it needs a frozen-Lo checkpoint argument.

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
        student_sample_steps: int = 4,
        tail_percentiles: list[float] | None = None,
        tail_hist_ranges: dict[str, tuple[float, float]] | None = None,
        tail_directions: dict[str, str] | None = None,
        tail_hist_bins: int = 10000,
        best_tail_checkpoint_path: str | None = None,
        validation_mode: str = "from_noise",
    ) -> None:
        self._tail_percentiles: list[float] = (
            [99.99, 99.9999] if tail_percentiles is None else list(tail_percentiles)
        )
        if not all(0 < p < 100 for p in self._tail_percentiles):
            raise ValueError("All tail_percentiles must be in (0, 100)")
        self._val_dataset_path = val_dataset_path
        self._coarse_val_data = coarse_val_data
        self._teacher_model = teacher_model
        self._best_checkpoint_path = best_checkpoint_path
        self._coarse_patch_yx = coarse_patch_yx
        self._n_student_samples = n_student_samples
        self._student_sample_steps = student_sample_steps
        self._tail_hist_ranges: dict[str, tuple[float, float]] = (
            dict(tail_hist_ranges)
            if tail_hist_ranges is not None
            else dict(_DEFAULT_TAIL_HIST_RANGES)
        )
        self._tail_directions: dict[str, str] = (
            dict(tail_directions)
            if tail_directions is not None
            else dict(_DEFAULT_TAIL_DIRECTIONS)
        )
        for var, direction in self._tail_directions.items():
            if direction not in ("upper", "lower"):
                raise ValueError(
                    f"tail_directions[{var!r}] must be 'upper' or 'lower', "
                    f"got {direction!r}."
                )
        # Per-variable CRPS scale (the variable's std) so CRPS in disparate
        # physical units can be averaged across variables without the
        # large-magnitude fields dominating.  Read from the teacher's fine
        # normalizer; missing entries fall back to 1.0 (no normalization).
        self._crps_scales: dict[str, float] = self._per_var_scales(teacher_model)
        self._tail_hist_bins = tail_hist_bins
        self._best_tail_checkpoint_path = best_tail_checkpoint_path
        if validation_mode not in ("from_noise", "lo_renoise"):
            raise ValueError(
                f"validation_mode must be 'from_noise' or 'lo_renoise', "
                f"got {validation_mode!r}."
            )
        self._validation_mode = validation_mode
        self._best_crps = float("inf")
        # Tracks min mean-over-vars |ratio - 1.0| for best_tail_checkpoint_path
        # selection, using the highest (most extreme) percentile.
        self._best_tail_score = float("inf")
        self._teacher_ds: xr.Dataset | None = None

    @staticmethod
    def _per_var_scales(teacher_model: DiffusionModel) -> dict[str, float]:
        """Per-variable CRPS scale (the fine normalizer's std) for each output.

        Returns ``{var: std}`` over ``teacher_model.out_packer.names``.  Std
        tensors are reduced with ``mean()`` (they are per-variable scalars in
        practice); a missing or non-positive std falls back to ``1.0`` so the
        variable contributes its raw physical CRPS rather than dividing by zero.
        """
        stds = getattr(teacher_model.normalizer.fine, "stds", {}) or {}
        scales: dict[str, float] = {}
        for name in teacher_model.out_packer.names:
            std = stds.get(name) if hasattr(stds, "get") else None
            if std is None:
                scales[name] = 1.0
                continue
            value = float(torch.as_tensor(std).float().mean().item())
            scales[name] = value if value > 0 else 1.0
        return scales

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
        crps_by_var, tail_by_pct, spec_by_var, spec_curves = (
            self._compute_validation_crps(student)
        )
        crps = crps_by_var["mean"]

        if not is_rank0():
            return

        from fme.downscaling.distillation.student_checkpoint import (
            save_student_checkpoint,
        )

        # CRPS-best checkpoint.
        tail_summary = self._tail_summary_str(tail_by_pct)
        if crps < self._best_crps:
            self._best_crps = crps
            save_student_checkpoint(
                student_module=student._ace_module,
                teacher=self._teacher_model,
                path=self._best_checkpoint_path,
            )
            logger.info(
                f"[BestStudentCallback] iteration={iteration} CRPS={crps:.6f}"
                f"{tail_summary} (new best) → {self._best_checkpoint_path}"
            )
        else:
            logger.info(
                f"[BestStudentCallback] iteration={iteration} CRPS={crps:.6f}"
                f"{tail_summary} (best={self._best_crps:.6f}, no improvement)"
            )

        # Tail-best checkpoint: highest percentile, mean-over-vars |ratio - 1|.
        if self._best_tail_checkpoint_path and self._tail_percentiles:
            top_pct = max(self._tail_percentiles)
            ratios = {
                var: ratio
                for var, ratio in tail_by_pct.get(top_pct, {}).items()
                if var != "mean"
            }
            tail_score = _tail_deviation_score(ratios)
            if np.isfinite(tail_score) and tail_score < self._best_tail_score:
                self._best_tail_score = tail_score
                save_student_checkpoint(
                    student_module=student._ace_module,
                    teacher=self._teacher_model,
                    path=self._best_tail_checkpoint_path,
                )
                logger.info(
                    f"[BestStudentCallback] iteration={iteration}"
                    f" tail_{top_pct} mean|ratio-1|={tail_score:.4f}"
                    f" (new tail best) → {self._best_tail_checkpoint_path}"
                )

        # Log after the best-checkpoint update so val/crps_best includes
        # this iteration's result.
        self._log_to_wandb(
            crps_by_var, tail_by_pct, spec_by_var, spec_curves, iteration
        )

    def _tail_summary_str(self, tail_by_pct: dict[float, dict[str, float]]) -> str:
        parts = []
        for pct in sorted(tail_by_pct):
            mean = tail_by_pct[pct].get("mean")
            if mean is not None:
                parts.append(f"tail_{pct}={mean:.4f}")
        return (", " + ", ".join(parts)) if parts else ""

    def _log_to_wandb(
        self,
        crps_by_var: dict[str, float],
        tail_by_pct: dict[float, dict[str, float]],
        spec_by_var: dict[str, dict[str, float]],
        spec_curves: dict[str, dict[str, list[float]]],
        iteration: int,
    ) -> None:
        """Log CRPS, tail-fraction, and spectral scalars + PSD curves to wandb."""
        if not np.isfinite(crps_by_var.get("mean", float("inf"))):
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        # Holds scalars plus wandb custom-chart objects (PSD curves).
        payload: dict[str, object] = {
            f"val/crps_{k}": v for k, v in crps_by_var.items()
        }
        if np.isfinite(self._best_crps):
            payload["val/crps_best"] = self._best_crps
        for pct, by_var in tail_by_pct.items():
            for k, v in by_var.items():
                payload[f"val/tail_{pct}_{k}"] = v
        if self._best_tail_checkpoint_path and np.isfinite(self._best_tail_score):
            payload["val/tail_best_score"] = self._best_tail_score
        for var, metrics in spec_by_var.items():
            for band, v in metrics.items():
                payload[f"val/spec_{band}_{var}"] = v
        if spec_by_var:
            payload["val/spec_mae_mean"] = float(
                np.mean([m["mae"] for m in spec_by_var.values()])
            )
        # Raw mean-PSD curves (student vs teacher) as log10 line charts, one per
        # variable.  Lets a large band MAE be judged against the absolute energy
        # at each wavenumber — a big log-ratio where the teacher PSD is tiny
        # (smooth field) is a metric artifact, not a real spectral failure.
        # Guarded so a viz hiccup can never drop the scalar metrics.
        for var, curves in spec_curves.items():
            student = curves.get("student")
            teacher = curves.get("teacher")
            if not student or not teacher:
                continue
            try:
                wavenumber = list(range(len(student)))
                payload[f"val/psd_{var}"] = wandb.plot.line_series(
                    xs=wavenumber,
                    ys=[np.log10(student).tolist(), np.log10(teacher).tolist()],
                    keys=["student", "teacher"],
                    title=f"log10 mean PSD: {var}",
                    xname="zonal wavenumber",
                )
            except Exception:  # noqa: BLE001 - viz is best-effort, never fatal
                continue
        wandb.log(payload, step=iteration)

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

    def _sample_student_output(
        self,
        student: AceDiffusionTeacher,
        condition: torch.Tensor,
        teacher_phys: dict[str, torch.Tensor],
        B: int,
        H: int,
        W: int,
        C_out: int,
    ) -> torch.Tensor:
        """Return the student ensemble ``(B, n, C_out, H, W)`` in normalized space.

        Dispatches on ``self._validation_mode``.  ``student._ace_module`` is a
        ``UNetDiffusionModule`` wrapping ``EDMPrecond`` whose forward signature
        is ``(x, condition, sigma)`` — the denoiser the sampling helpers expect.
        """
        from fme.downscaling.distillation.student_sampling import (
            sample_student_from_noise,
            sample_student_lo_renoise,
        )

        if self._validation_mode == "lo_renoise":
            target_norm = self._packed_target_norm(teacher_phys)
            return sample_student_lo_renoise(
                student._ace_module,
                condition,
                target_norm,
                num_steps=self._student_sample_steps,
                sigma_min=student._sigma_min,
                sigma_max=student._sigma_max,
            )
        return sample_student_from_noise(
            student._ace_module,
            condition,
            c_out=C_out,
            n_samples=self._n_student_samples,
            num_steps=self._student_sample_steps,
            sigma_min=student._sigma_min,
            sigma_max=student._sigma_max,
        )

    def _packed_target_norm(
        self, teacher_phys: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Pack+normalize the teacher target members for lo_renoise input.

        The lo_renoise ensemble comes from re-noising distinct teacher members
        (one ``eps`` draw each), so we take the first ``n_student_samples``
        members (clamped to what the zarr holds).  Every output variable must
        be present in the zarr — the full field is re-noised, not just compared.

        Returns:
            ``(B, n, C_out, H, W)`` normalized packed target.
        """
        model = self._teacher_model
        names = model.out_packer.names
        missing = [v for v in names if v not in teacher_phys]
        if missing:
            raise ValueError(
                "lo_renoise validation needs every output variable in the "
                f"teacher zarr to re-noise; missing {missing}."
            )
        n_teacher = teacher_phys[names[0]].shape[1]
        n = min(self._n_student_samples, n_teacher)
        phys = {v: teacher_phys[v][:, :n] for v in names}  # each (B, n, H, W)
        norm = model.normalizer.fine.normalize(phys)
        return model.out_packer.pack(norm, axis=-3)  # (B, n, C_out, H, W)

    @torch.no_grad()
    def _compute_validation_crps(
        self, student: AceDiffusionTeacher
    ) -> tuple[
        dict[str, float],
        dict[float, dict[str, float]],
        dict[str, dict[str, float]],
        dict[str, dict[str, list[float]]],
    ]:
        """Return ``(crps_by_var, tail_by_pct, spec_by_var, spec_curves)``.

        ``crps_by_var``: ``{"<var>": crps, "mean": <var-averaged>}``.
        ``tail_by_pct``: ``{percentile: {"<var>": student_pXX / target_pXX,
        "mean": <var-averaged>}}`` for each entry in ``self._tail_percentiles``.
        Variables not in ``self._tail_hist_ranges`` are absent from every
        inner dict.
        ``spec_by_var``: ``{"<var>": {"mae": ..., "mae_lo": ..., "mae_mid": ...,
        "mae_hi": ...}}`` where each value is the mean absolute log10-ratio of
        the zonal power spectra (student / teacher), split into lo/mid/hi thirds
        of the wavenumber axis.
        ``spec_curves``: ``{"<var>": {"student": [...], "teacher": [...]}}`` —
        the raw mean PSD per zonal wavenumber for each, so the band MAEs above
        can be read against the absolute energy at each scale.

        Sampling uses the FastGen "predict x0 → renoise" loop (via
        ``fastgen_sampler``) instead of ACE's Heun sampler so validation
        reflects the trajectory the student was distilled against.  Empty
        CRPS result (no overlapping variables) returns ``{"mean": inf}``;
        empty tail/spec results return empty dicts.
        """
        import torch.distributed as dist

        from fme.downscaling.metrics_and_maths import compute_zonal_power_spectrum

        teacher_ds = self._load_teacher_ds()
        teacher_model = self._teacher_model

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
        # Per-variable histograms for the tail-fraction metric.  Allocated on
        # the same device the student tensors live on; populated lazily on the
        # first batch so we know the device without assuming one here.
        hist_student: dict[str, torch.Tensor] = {}
        hist_target: dict[str, torch.Tensor] = {}
        # Per-variable accumulated PSD sums (shape: nw) for the spectral metric.
        psd_student_sum: dict[str, torch.Tensor] = {}
        psd_teacher_sum: dict[str, torch.Tensor] = {}
        psd_ns: dict[str, int] = {}
        psd_nt: dict[str, int] = {}

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

            # Fine-resolution lat/lon for this (possibly patched) batch.
            fine_coords = teacher_model.get_fine_coords_for_batch(batch)
            fine_lats = fine_coords.lat.cpu().numpy()
            fine_lons = fine_coords.lon.cpu().numpy()

            # Teacher target members (physical units) for every output var that
            # is present in the zarr, fetched once and reused for both student
            # input construction (lo_renoise re-noises these) and the metric
            # comparison below.  Shape per var: (B, n_teacher, H, W).
            teacher_phys: dict[str, torch.Tensor] = {}
            for var in teacher_model.out_packer.names:
                if var not in teacher_ds:
                    continue
                teacher_np = (
                    teacher_ds[var]
                    .sel(time=times)
                    .sel(latitude=fine_lats, longitude=fine_lons, method="nearest")
                    .values
                )
                teacher_phys[var] = torch.from_numpy(teacher_np).to(
                    condition.device, dtype=torch.float32
                )

            # Produce the student output ensemble (B, n, C_out, H, W) per the
            # configured validation mode (see student_sampling).
            output_norm = self._sample_student_output(
                student, condition, teacher_phys, B, H, W, C_out
            )

            # Unpack channel dim (axis=-3) then denormalize → physical units.
            # out_packer.unpack on (B, n, C_out, H, W) with axis=-3 gives
            # {var: (B, n, H, W)}.
            output = teacher_model.normalizer.fine.denormalize(
                teacher_model.out_packer.unpack(output_norm, axis=-3)
            )

            for var, student_tensor in output.items():
                teacher_batch = teacher_phys.get(var)
                if teacher_batch is None:
                    continue

                # CRPS per batch element, then accumulate spatial sum.
                for i in range(B):
                    crps_map = _crps_ensemble(student_tensor[i], teacher_batch[i])
                    crps_sum[var] = crps_sum.get(var, 0.0) + float(crps_map.sum())
                    count[var] = count.get(var, 0) + crps_map.numel()

                # Tail-fraction histogram accumulation.  torch.histc clips
                # out-of-range values to the boundary bin, so the chosen range
                # must cover the realistic tail; we default to a wide bound.
                if var in self._tail_hist_ranges:
                    lo, hi = self._tail_hist_ranges[var]
                    if var not in hist_student:
                        hist_student[var] = torch.zeros(
                            self._tail_hist_bins,
                            device=student_tensor.device,
                            dtype=torch.float64,
                        )
                        hist_target[var] = torch.zeros(
                            self._tail_hist_bins,
                            device=student_tensor.device,
                            dtype=torch.float64,
                        )
                    hist_student[var] += torch.histc(
                        student_tensor.float().flatten(),
                        bins=self._tail_hist_bins,
                        min=lo,
                        max=hi,
                    ).to(torch.float64)
                    hist_target[var] += torch.histc(
                        teacher_batch.float().flatten(),
                        bins=self._tail_hist_bins,
                        min=lo,
                        max=hi,
                    ).to(torch.float64)

                # Zonal power spectrum — PSD-then-mean, averaged over lat.
                student_flat = student_tensor.float().reshape(-1, H, W)
                psd_s = (
                    compute_zonal_power_spectrum(student_flat)
                    .to(torch.float64)
                    .sum(dim=0)
                )
                teacher_flat = teacher_batch.float().reshape(-1, H, W)
                psd_t = (
                    compute_zonal_power_spectrum(teacher_flat)
                    .to(torch.float64)
                    .sum(dim=0)
                )
                if var not in psd_student_sum:
                    psd_student_sum[var] = psd_s
                    psd_teacher_sum[var] = psd_t
                    psd_ns[var] = student_flat.shape[0]
                    psd_nt[var] = teacher_flat.shape[0]
                else:
                    psd_student_sum[var] += psd_s
                    psd_teacher_sum[var] += psd_t
                    psd_ns[var] += student_flat.shape[0]
                    psd_nt[var] += teacher_flat.shape[0]

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

        crps_result: dict[str, float] = {}
        for i, var in enumerate(global_keys):
            c = float(counts[i].item())
            if c > 0:
                crps_result[var] = float(sums[i].item()) / c
        if crps_result:
            # Per-var values stay in physical units (logged as val/crps_<var>);
            # "mean" is the std-normalized cross-variable mean used for
            # checkpoint selection so disparate-magnitude fields contribute
            # comparably (e.g. precip ~1e-5 is not drowned out by PRMSL ~1e3).
            crps_result["mean"] = _normalized_mean(crps_result, self._crps_scales)
        else:
            crps_result = {"mean": float("inf")}

        # Tail-fraction reduction.  Variables that any rank populated need a
        # matching entry on every rank for all_reduce; create zero histograms
        # on ranks that didn't see the variable so the collective is well-
        # formed.  In practice every rank sees the same variable set because
        # the val data is replicated across ranks (force_non_distributed=True)
        # — this is defensive.
        tail_keys = sorted(
            v for v in self._tail_hist_ranges if v in teacher_ds.data_vars
        )
        for var in tail_keys:
            if var not in hist_student:
                hist_student[var] = torch.zeros(
                    self._tail_hist_bins, device=device, dtype=torch.float64
                )
                hist_target[var] = torch.zeros(
                    self._tail_hist_bins, device=device, dtype=torch.float64
                )
            if world_size > 1:
                dist.all_reduce(hist_student[var], op=dist.ReduceOp.SUM)
                dist.all_reduce(hist_target[var], op=dist.ReduceOp.SUM)

        tail_by_pct: dict[float, dict[str, float]] = {}
        for pct in self._tail_percentiles:
            by_var: dict[str, float] = {}
            for var in tail_keys:
                lo, hi = self._tail_hist_ranges[var]
                q = _tail_quantile_level(pct, self._tail_directions.get(var, "upper"))
                student_p = _quantile_from_histogram(hist_student[var], lo, hi, q)
                target_p = _quantile_from_histogram(hist_target[var], lo, hi, q)
                if target_p is None or student_p is None or target_p <= 0:
                    continue
                by_var[var] = student_p / target_p
            if by_var:
                by_var["mean"] = float(np.mean(list(by_var.values())))
                tail_by_pct[pct] = by_var

        # Spectral PSD reduction.  First broadcast nw so ranks with no batches
        # can allocate zeros of the right shape for the collective.
        psd_nw_local = 0
        if psd_student_sum:
            psd_nw_local = next(iter(psd_student_sum.values())).shape[0]
        nw_tensor = torch.tensor([psd_nw_local], device=device, dtype=torch.int64)
        if world_size > 1:
            dist.all_reduce(nw_tensor, op=dist.ReduceOp.MAX)
        psd_nw = int(nw_tensor.item())

        spec_by_var: dict[str, dict[str, float]] = {}
        # Per-variable mean PSD curves (student + teacher) for raw-spectrum
        # logging, so a large |log-ratio| can be judged against the absolute
        # energy at that wavenumber (a huge ratio where the teacher PSD is
        # ~0 is a metric artifact of a smooth field, not a real failure).
        spec_curves: dict[str, dict[str, list[float]]] = {}
        if psd_nw > 0:
            ns_counts = torch.tensor(
                [psd_ns.get(k, 0) for k in global_keys],
                device=device,
                dtype=torch.float64,
            )
            nt_counts = torch.tensor(
                [psd_nt.get(k, 0) for k in global_keys],
                device=device,
                dtype=torch.float64,
            )
            zeros = torch.zeros(psd_nw, device=device, dtype=torch.float64)
            psd_s_mat = torch.stack(
                [psd_student_sum.get(k, zeros) for k in global_keys]
            )
            psd_t_mat = torch.stack(
                [psd_teacher_sum.get(k, zeros) for k in global_keys]
            )
            if world_size > 1:
                dist.all_reduce(ns_counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(nt_counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(psd_s_mat, op=dist.ReduceOp.SUM)
                dist.all_reduce(psd_t_mat, op=dist.ReduceOp.SUM)

            lo_end = psd_nw // 3
            hi_start = 2 * psd_nw // 3
            for i, var in enumerate(global_keys):
                ns = float(ns_counts[i].item())
                nt = float(nt_counts[i].item())
                if ns <= 0 or nt <= 0:
                    continue
                mean_s = psd_s_mat[i] / ns
                mean_t = psd_t_mat[i] / nt
                log_ratio = torch.log10(
                    mean_s.clamp(min=1e-30) / mean_t.clamp(min=1e-30)
                )
                spec_by_var[var] = {
                    "mae": float(log_ratio.abs().mean().item()),
                    "mae_lo": float(log_ratio[:lo_end].abs().mean().item()),
                    "mae_mid": float(log_ratio[lo_end:hi_start].abs().mean().item()),
                    "mae_hi": float(log_ratio[hi_start:].abs().mean().item()),
                }
                spec_curves[var] = {
                    "student": mean_s.clamp(min=1e-30).cpu().tolist(),
                    "teacher": mean_t.clamp(min=1e-30).cpu().tolist(),
                }

        return crps_result, tail_by_pct, spec_by_var, spec_curves
