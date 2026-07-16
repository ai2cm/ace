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

import statistics
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

# Per-variable reference value from which tail *extremity* is measured, so the
# metric is a ratio of ANOMALIES rather than raw values.  Critical for PRMSL: a
# ratio of raw pressures (~1000 hPa offset) is ~1.0 even for a several-hPa
# deep-low error, so it's blind to the extreme that matters.  Measuring depth
# below a standard 1000 hPa (``1000 - p_0.01``) makes the ratio sensitive.
# Zero-referenced variables (winds ~0-centered, precip ≥0) keep the raw-value
# ratio (reference 0.0), i.e. unchanged behavior.
_DEFAULT_TAIL_REFERENCES: dict[str, float] = {"PRMSL": 1000.0}  # hPa


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


def _tail_magnitude(quantile_value: float, reference: float, direction: str) -> float:
    """Extremity magnitude of a tail quantile, measured from ``reference``.

    ``"upper"`` → ``quantile_value - reference`` (how far above); ``"lower"`` →
    ``reference - quantile_value`` (how far below).  With ``reference=0`` this is
    the raw value (unchanged for zero-referenced variables); for PRMSL
    (``reference=1000``, ``"lower"``) it is the depth below 1000 hPa, so the
    student/target ratio reflects deep-low anomalies rather than absolute
    pressures (a ratio of ~1000 hPa values would be ~1.0 regardless of error).
    """
    if direction == "upper":
        return quantile_value - reference
    if direction == "lower":
        return reference - quantile_value
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
            ``val/tail_<pct>_<var>`` = ratio of the student vs target tail
            *anomaly* (distance from ``tail_references[var]`` in the tail
            direction; values < 1 mean the student under-predicts the extreme,
            > 1 over-predicts).  The (std-normalized, var-averaged) CRPS remains
            the ``best_checkpoint_path`` criterion; see
            ``best_tail_checkpoint_path`` for tail-driven selection.
        tail_hist_ranges: Per-variable ``(lo, hi)`` ranges (physical units)
            used to bin values for percentile estimation; must bracket the
            full distribution.  Variables not in this dict get the tail metric
            skipped (CRPS is still computed).  Defaults to
            ``_DEFAULT_TAIL_HIST_RANGES`` (precip/PRMSL/u10/v10).
        tail_directions: Per-variable tail side, ``"upper"`` or ``"lower"``.
            Defaults to ``_DEFAULT_TAIL_DIRECTIONS`` (PRMSL → ``"lower"`` for
            deep-low extremes, everything else → ``"upper"``).
        tail_references: Per-variable reference from which the tail *anomaly* is
            measured, so the ratio is of anomalies not raw values.  Defaults to
            ``_DEFAULT_TAIL_REFERENCES`` (PRMSL → 1000 hPa, so the metric is the
            depth-below-1000 ratio; a raw-pressure ratio is ~1.0 regardless of
            deep-low error).  Unlisted variables use 0.0 (raw-value ratio,
            unchanged) — appropriate for the ~0-centered winds and ≥0 precip.
        tail_hist_bins: Number of equal-width bins inside each variable's
            range (default 10000).  Resolution is ``(hi - lo) / tail_hist_bins``
            — for PRATEsfc that's 1e-5 kg/m²/s.
        best_tail_checkpoint_path: Optional destination path for a checkpoint
            selected by the tail metric rather than CRPS.  When provided,
            saves whenever the highest-percentile tail score improves, where
            the score is the mean over variables of ``|ratio - 1|`` (each
            variable's discrepancy counts; no cross-variable cancellation).
            Useful when CRPS has plateaued but tail fidelity is still evolving.
        best_spec_checkpoint_path: Optional destination path for a checkpoint
            selected by the spectral metric (``val/spec_mae_mean`` = mean over
            variables of the zonal-PSD log-ratio MAE).  Because that metric is
            spiky per-validation, selection uses its rolling median over the
            last ``spec_patience_window`` validations rather than the raw value,
            so a lucky single-snapshot dip cannot win.  ``None`` disables the
            selector (no ``best_student_spec.ckpt`` written).
        early_stop_patience: Number of consecutive validations with no
            improvement in *any* selector (CRPS, tail, or spectral) after which
            training is asked to stop (the conservative "any-improved" policy).
            ``None`` (default) disables early stopping; the run trains to
            ``max_iter`` as before.  See ``should_stop``.
        spec_patience_window: Window (in validations) of the rolling median used
            for the spectral selector and the spectral improvement signal that
            feeds early stopping (default 5).
        combined_checkpoint_dir: Optional directory for iteration-stamped
            candidate checkpoints (``best_student_combined_<iteration>.ckpt``)
            selected by the *combined* tail+spectral rule: a candidate is saved
            when one metric is within ``combined_tolerance`` of its all-time
            best while the other beats its all-time best by at least
            ``combined_improvement`` — i.e. a jointly good checkpoint that a
            single-metric selector would miss.  Bests are compared as they were
            *before* the current validation's updates, and the spectral signal
            is the same rolling median the spectral selector uses.  Because the
            rule needs finite prior bests, no candidate is saved on the first
            validation, and the selector is only live when
            ``best_tail_checkpoint_path`` is set (that selector maintains the
            tail best).  ``None`` (default) disables candidate saving.
        combined_tolerance: "Within range" multiplier for the combined rule: a
            metric counts as near-best when ``metric <= best * tolerance``
            (default 1.05, i.e. within 5%).  Must be >= 1.
        combined_improvement: "Substantial improvement" multiplier: a metric
            counts as substantially improved when
            ``metric <= best * improvement`` (default 0.95, i.e. at least 5%
            better).  Must be in (0, 1).
        combined_keep: Number of most-recent combined candidates retained on
            disk; older ones are pruned (default 3).  Must be >= 1.
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
            - ``"hi_cascade"``: a high-noise *segment* student (the one being
              trained) is validated end-to-end through a frozen low-noise
              student — fresh noise at ``sigma_max`` → this student → re-noise
              to the boundary → ``frozen_lo_net`` → clean x0.  The bundle
              deployment path.  Requires ``frozen_lo_net``.  Use for per-expert
              Student-Hi.
        frozen_lo_net: The frozen low-noise segment student denoiser
            (``net(x, condition, sigma) -> x0``) to cascade through in
            ``"hi_cascade"`` mode.  Required for that mode, ignored otherwise.
        frozen_lo_sample_steps: FastGen step count for the frozen Lo segment of
            the ``hi_cascade`` cascade (default 2, matching the 2-step Lo).
        frozen_lo_sigma_min: Lower sigma bound of the frozen Lo's segment
            (default 0.005).  The segment boundary (Lo's ``sigma_max`` = this
            student's ``sigma_min``) is taken from the trained Hi student, not
            from this argument, so the handoff is exact regardless of what the
            frozen Lo checkpoint recorded as its range.

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
        tail_references: dict[str, float] | None = None,
        tail_hist_bins: int = 10000,
        best_tail_checkpoint_path: str | None = None,
        best_spec_checkpoint_path: str | None = None,
        early_stop_patience: int | None = None,
        spec_patience_window: int = 5,
        combined_checkpoint_dir: str | None = None,
        combined_tolerance: float = 1.05,
        combined_improvement: float = 0.95,
        combined_keep: int = 3,
        validation_mode: str = "from_noise",
        frozen_lo_net: torch.nn.Module | None = None,
        frozen_lo_sample_steps: int = 2,
        frozen_lo_sigma_min: float = 0.005,
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
        self._tail_references: dict[str, float] = (
            dict(tail_references)
            if tail_references is not None
            else dict(_DEFAULT_TAIL_REFERENCES)
        )
        # Per-variable CRPS scale (the variable's std) so CRPS in disparate
        # physical units can be averaged across variables without the
        # large-magnitude fields dominating.  Read from the teacher's fine
        # normalizer; missing entries fall back to 1.0 (no normalization).
        self._crps_scales: dict[str, float] = self._per_var_scales(teacher_model)
        self._tail_hist_bins = tail_hist_bins
        self._best_tail_checkpoint_path = best_tail_checkpoint_path
        if validation_mode not in ("from_noise", "lo_renoise", "hi_cascade"):
            raise ValueError(
                f"validation_mode must be 'from_noise', 'lo_renoise', or "
                f"'hi_cascade', got {validation_mode!r}."
            )
        if validation_mode == "hi_cascade" and frozen_lo_net is None:
            raise ValueError(
                "validation_mode='hi_cascade' requires frozen_lo_net (the "
                "frozen low-noise segment student to cascade through)."
            )
        self._validation_mode = validation_mode
        self._frozen_lo_net = frozen_lo_net
        if frozen_lo_net is not None:
            frozen_lo_net.eval()
        self._frozen_lo_sample_steps = frozen_lo_sample_steps
        self._frozen_lo_sigma_min = frozen_lo_sigma_min
        self._best_crps = float("inf")
        # Tracks min mean-over-vars |ratio - 1.0| for best_tail_checkpoint_path
        # selection, using the highest (most extreme) percentile.
        self._best_tail_score = float("inf")
        # Spectral selector + early stopping.  The spectral metric is selected
        # on a rolling median (it is spiky per-validation), and early stop uses
        # the improvement of any selector (CRPS/tail/spec) — see _record_validation.
        self._best_spec_checkpoint_path = best_spec_checkpoint_path
        if spec_patience_window < 1:
            raise ValueError("spec_patience_window must be >= 1")
        self._spec_patience_window = spec_patience_window
        self._best_spec = float("inf")
        self._spec_history: list[float] = []
        # Combined tail+spectral candidate selector (see class docstring).
        if combined_tolerance < 1.0:
            raise ValueError("combined_tolerance must be >= 1.0")
        if not 0.0 < combined_improvement < 1.0:
            raise ValueError("combined_improvement must be in (0, 1)")
        if combined_keep < 1:
            raise ValueError("combined_keep must be >= 1")
        self._combined_checkpoint_dir = combined_checkpoint_dir
        self._combined_tolerance = combined_tolerance
        self._combined_improvement = combined_improvement
        self._combined_keep = combined_keep
        self._combined_saves = 0
        if early_stop_patience is not None and early_stop_patience < 1:
            raise ValueError("early_stop_patience must be >= 1 when set")
        self._early_stop_patience = early_stop_patience
        self._checks_since_improvement = 0
        self._stop_requested = False
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

        student: AceDiffusionTeacher = model.net
        # Validation runs on ALL ranks (sharded) — see class docstring.
        crps_by_var, tail_by_pct, spec_by_var, spec_curves = (
            self._compute_validation_crps(student)
        )

        if not is_rank0():
            return

        # Update the best checkpoints (rank-0 writes) and the early-stop
        # counter, then log so val/*_best reflect this iteration's result.
        self._record_validation(
            student, crps_by_var, tail_by_pct, spec_by_var, iteration
        )
        self._log_to_wandb(
            crps_by_var, tail_by_pct, spec_by_var, spec_curves, iteration
        )

    def should_stop(self) -> bool:
        """Whether early stopping has requested training to terminate.

        Set on rank-0 by ``_record_validation`` once ``early_stop_patience``
        consecutive validations pass with no improvement in any selector.  Read
        by the FastGen ``AutoResumeInterface`` injected in ``fastgen_train`` (see
        ``_EarlyStopAutoResume``), which the trainer polls each iteration and
        which handles the DDP rank-0 → all-ranks broadcast.  Always ``False``
        when ``early_stop_patience`` is ``None``.
        """
        return self._stop_requested

    @staticmethod
    def _spec_mae_mean(spec_by_var: dict[str, dict[str, float]]) -> float:
        """Mean over variables of the spectral log-ratio MAE (``inf`` if empty).

        Mirrors ``val/spec_mae_mean``; used for both the spectral selector and
        wandb logging so the two never diverge.
        """
        maes = [m["mae"] for m in spec_by_var.values() if "mae" in m]
        return float(np.mean(maes)) if maes else float("inf")

    def _record_validation(
        self,
        student: AceDiffusionTeacher,
        crps_by_var: dict[str, float],
        tail_by_pct: dict[float, dict[str, float]],
        spec_by_var: dict[str, dict[str, float]],
        iteration: int,
    ) -> None:
        """Update the CRPS/tail/spectral best checkpoints and early-stop counter.

        Runs on rank-0 only (it writes checkpoints and mutates selection state).
        Each selector saves on strict improvement of its own metric.  The
        early-stop counter resets whenever *any* selector improves and
        increments otherwise, so a run is only cut once none of CRPS, tail, or
        spectral fidelity is still improving — the conservative "any-improved"
        policy.  Kept free of FastGen imports so it is unit-testable directly.
        """
        from fme.downscaling.distillation.student_checkpoint import (
            save_candidate_checkpoint,
            save_student_checkpoint,
        )

        try:
            import fastgen.utils.logging_utils as _logger

            _log = _logger.info
        except ImportError:
            import logging

            _log = logging.getLogger(__name__).info

        crps = crps_by_var["mean"]
        tail_summary = self._tail_summary_str(tail_by_pct)

        # The combined selector compares against the per-metric bests as they
        # were BEFORE this validation's own updates — otherwise a substantial
        # improvement would raise the bar before the combined check sees it.
        prev_best_tail = self._best_tail_score
        prev_best_spec = self._best_spec

        # CRPS-best checkpoint.
        crps_improved = crps < self._best_crps
        if crps_improved:
            self._best_crps = crps
            save_student_checkpoint(
                student_module=student._ace_module,
                teacher=self._teacher_model,
                path=self._best_checkpoint_path,
            )
            _log(
                f"[BestStudentCallback] iteration={iteration} CRPS={crps:.6f}"
                f"{tail_summary} (new best) → {self._best_checkpoint_path}"
            )
        else:
            _log(
                f"[BestStudentCallback] iteration={iteration} CRPS={crps:.6f}"
                f"{tail_summary} (best={self._best_crps:.6f}, no improvement)"
            )

        # Tail score: highest percentile, mean-over-vars |ratio - 1|.  Computed
        # whenever tail data is present (the combined selector below also reads
        # it); the tail-best save and best update remain gated on the tail
        # checkpoint path, exactly as before.
        tail_score: float | None = None
        if self._tail_percentiles:
            top_pct = max(self._tail_percentiles)
            ratios = {
                var: ratio
                for var, ratio in tail_by_pct.get(top_pct, {}).items()
                if var != "mean"
            }
            tail_score = _tail_deviation_score(ratios)
        tail_improved = False
        if self._best_tail_checkpoint_path and tail_score is not None:
            if np.isfinite(tail_score) and tail_score < self._best_tail_score:
                tail_improved = True
                self._best_tail_score = tail_score
                save_student_checkpoint(
                    student_module=student._ace_module,
                    teacher=self._teacher_model,
                    path=self._best_tail_checkpoint_path,
                )
                _log(
                    f"[BestStudentCallback] iteration={iteration}"
                    f" tail_{top_pct} mean|ratio-1|={tail_score:.4f}"
                    f" (new tail best) → {self._best_tail_checkpoint_path}"
                )

        # Spectral-best checkpoint: rolling median of val/spec_mae_mean (spiky
        # per-validation, so a single-snapshot dip cannot win).  The median is
        # tracked even without a checkpoint path so it can feed the early-stop
        # improvement signal below.
        spec_improved = False
        spec_median: float | None = None
        spec_mae_mean = self._spec_mae_mean(spec_by_var)
        if np.isfinite(spec_mae_mean):
            self._spec_history.append(spec_mae_mean)
            if len(self._spec_history) > self._spec_patience_window:
                self._spec_history = self._spec_history[-self._spec_patience_window :]
            spec_median = statistics.median(self._spec_history)
            if spec_median < self._best_spec:
                spec_improved = True
                self._best_spec = spec_median
                if self._best_spec_checkpoint_path:
                    save_student_checkpoint(
                        student_module=student._ace_module,
                        teacher=self._teacher_model,
                        path=self._best_spec_checkpoint_path,
                    )
                    _log(
                        f"[BestStudentCallback] iteration={iteration}"
                        f" spec_mae_mean(median)={spec_median:.6f}"
                        f" (new spec best) → {self._best_spec_checkpoint_path}"
                    )

        # Combined tail+spectral candidate: one metric within tolerance of its
        # pre-update all-time best while the other substantially beats its
        # pre-update all-time best.  Skipped until both prior bests are finite
        # (with the infs of the first validation any finite metric would
        # trivially qualify and burn a candidate slot on an arbitrary early
        # checkpoint).  Deliberately absent from the early-stop signal below: a
        # combined save implies a substantial tail-or-spec improvement, so the
        # corresponding individual selector already resets the counter.
        if (
            self._combined_checkpoint_dir is not None
            and tail_score is not None
            and np.isfinite(tail_score)
            and spec_median is not None
            and np.isfinite(prev_best_tail)
            and np.isfinite(prev_best_spec)
        ):
            tail_near = tail_score <= prev_best_tail * self._combined_tolerance
            tail_sub = tail_score <= prev_best_tail * self._combined_improvement
            spec_near = spec_median <= prev_best_spec * self._combined_tolerance
            spec_sub = spec_median <= prev_best_spec * self._combined_improvement
            if (tail_near and spec_sub) or (spec_near and tail_sub):
                candidate_path = save_candidate_checkpoint(
                    student_module=student._ace_module,
                    teacher=self._teacher_model,
                    directory=self._combined_checkpoint_dir,
                    iteration=iteration,
                    keep=self._combined_keep,
                )
                self._combined_saves += 1
                _log(
                    f"[BestStudentCallback] iteration={iteration} combined"
                    f" candidate (tail={tail_score:.4f} vs best"
                    f" {prev_best_tail:.4f}, spec_median={spec_median:.6f} vs"
                    f" best {prev_best_spec:.6f}) → {candidate_path}"
                )

        # Early stop: reset on any improvement, else count non-improving
        # validations and request termination once patience is exceeded.
        if self._early_stop_patience is not None:
            if crps_improved or tail_improved or spec_improved:
                self._checks_since_improvement = 0
            else:
                self._checks_since_improvement += 1
                if (
                    self._checks_since_improvement >= self._early_stop_patience
                    and not self._stop_requested
                ):
                    self._stop_requested = True
                    _log(
                        "[BestStudentCallback] early stop requested at "
                        f"iteration={iteration}: no improvement in any selector "
                        f"for {self._checks_since_improvement} validations "
                        f"(patience={self._early_stop_patience}). Best so far: "
                        f"CRPS={self._best_crps:.6f}, "
                        f"tail={self._best_tail_score:.4f}, "
                        f"spec={self._best_spec:.6f}."
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
        spec_mae_mean = self._spec_mae_mean(spec_by_var)
        if np.isfinite(spec_mae_mean):
            payload["val/spec_mae_mean"] = spec_mae_mean
        if self._best_spec_checkpoint_path and np.isfinite(self._best_spec):
            payload["val/spec_best"] = self._best_spec
        if self._combined_checkpoint_dir is not None:
            # Cumulative candidate count; step increases mark the iterations
            # at which combined candidates were written.
            payload["val/combined_saves"] = self._combined_saves
        if self._early_stop_patience is not None:
            payload["val/checks_since_improvement"] = self._checks_since_improvement
        # Raw mean-PSD curves (student vs teacher), one loglog figure per
        # variable, logged as the **raw matplotlib figure** (NOT wandb.Image) so
        # wandb auto-converts it to an interactive chart — matching the evaluator's
        # ``power_spectrum/<var>`` panels (aggregators/main.py logs the bare fig
        # too).  Keyed ``val/power_spectrum/<var>``.  The figures are closed only
        # *after* wandb.log has consumed them.  Guarded so a viz hiccup never drops
        # the scalar metrics.
        figs_to_close: list = []
        try:
            import matplotlib.pyplot as plt

            for var, curves in spec_curves.items():
                student = curves.get("student")
                teacher = curves.get("teacher")
                if not student or not teacher:
                    continue
                nw = len(student)
                fig = plt.figure()
                plt.loglog(teacher, label="teacher")
                plt.loglog(student, linestyle="--", label="student")
                # lo/mid/hi band boundaries used by spec_mae_{lo,mid,hi}.
                plt.axvline(nw // 3, color="gray", linestyle=":", linewidth=0.8)
                plt.axvline(2 * nw // 3, color="gray", linestyle=":", linewidth=0.8)
                plt.xlabel("zonal wavenumber")
                plt.ylabel("mean power")
                plt.title(f"{var} mean PSD (iter {iteration})")
                plt.legend()
                plt.grid(True, which="both", alpha=0.3)
                payload[f"val/power_spectrum/{var}"] = fig
                figs_to_close.append(fig)
        except Exception:  # noqa: BLE001 - viz is best-effort, never fatal
            pass
        wandb.log(payload, step=iteration)
        # Close figures only after wandb has serialized them (raw figs are
        # consumed at log time), to avoid leaking a figure per validation.
        if figs_to_close:
            try:
                import matplotlib.pyplot as plt

                for fig in figs_to_close:
                    plt.close(fig)
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass

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
        base_norm: torch.Tensor | None,
        B: int,
        H: int,
        W: int,
        C_out: int,
    ) -> torch.Tensor:
        """Return the student ensemble ``(B, n, C_out, H, W)`` in normalized space.

        Dispatches on ``self._validation_mode``.  ``student._ace_module`` is a
        ``UNetDiffusionModule`` wrapping ``EDMPrecond`` whose forward signature
        is ``(x, condition, sigma)`` — the denoiser the sampling helpers expect.

        For a **residual** teacher (``predict_residual=True``) the student net
        predicts ``fine − interpolate(coarse)`` in normalized space, so
        ``base_norm`` (the interpolated-coarse base, ``(B, C_out, H, W)``) must
        be **added back** to recover the full field before comparison — mirroring
        ``DiffusionModel.postprocess_generated``. It is ``None`` for
        non-residual teachers. For ``lo_renoise`` the base is also **subtracted**
        from the target before re-noising, since the student operates on
        ``residual + noise`` (its training-input distribution), not ``full +
        noise``.
        """
        from fme.downscaling.distillation.student_sampling import (
            sample_student_from_noise,
            sample_student_hi_cascade,
            sample_student_lo_renoise,
        )

        if self._validation_mode == "lo_renoise":
            target_norm = self._packed_target_norm(teacher_phys)
            if base_norm is not None:
                target_norm = target_norm - base_norm.unsqueeze(1)  # → residual
            out = sample_student_lo_renoise(
                student._ace_module,
                condition,
                target_norm,
                num_steps=self._student_sample_steps,
                sigma_min=student._sigma_min,
                sigma_max=student._sigma_max,
            )
        elif self._validation_mode == "hi_cascade":
            # The segment boundary (Lo's sigma_max = this Hi student's
            # sigma_min) comes from the trained Hi student, so the handoff node
            # is placed exactly at it regardless of the frozen Lo's recorded
            # range.  Ranges ascending: low segment first, then this student.
            assert self._frozen_lo_net is not None  # guaranteed by __init__
            boundary = student._sigma_min
            sigma_ranges = [
                (self._frozen_lo_sigma_min, boundary),
                (boundary, student._sigma_max),
            ]
            out = sample_student_hi_cascade(
                hi_net=student._ace_module,
                lo_net=self._frozen_lo_net,
                condition=condition,
                c_out=C_out,
                n_samples=self._n_student_samples,
                sigma_ranges=sigma_ranges,
                steps_per_range=[
                    self._frozen_lo_sample_steps,
                    self._student_sample_steps,
                ],
            )
        else:
            out = sample_student_from_noise(
                student._ace_module,
                condition,
                c_out=C_out,
                n_samples=self._n_student_samples,
                num_steps=self._student_sample_steps,
                sigma_min=student._sigma_min,
                sigma_max=student._sigma_max,
            )
        if base_norm is not None:
            out = out + base_norm.unsqueeze(1)  # residual → full field
        return out

    def _base_prediction_norm(
        self, batch, keep_mask: np.ndarray
    ) -> torch.Tensor | None:
        """Interpolated-coarse base for a residual teacher, in normalized space.

        Returns ``(B_kept, C_out, H_fine, W_fine)`` = ``interpolate(pack(
        coarse_normalizer.normalize(coarse_out_vars)), downscale_factor)`` —
        exactly the base ``DiffusionModel`` adds back at generation. Returns
        ``None`` when the teacher is not a residual model (nothing to add).
        ``keep_mask`` subsets the batch to times present in the teacher zarr,
        matching the condition/output subsetting.
        """
        model = self._teacher_model
        cfg = getattr(model, "config", None)
        if cfg is None or not getattr(cfg, "predict_residual", False):
            return None
        from fme.downscaling.metrics_and_maths import interpolate

        names = model.out_packer.names
        coarse = batch.data
        keep_idx = (
            None if keep_mask.all() else torch.from_numpy(np.flatnonzero(keep_mask))
        )
        phys: dict[str, torch.Tensor] = {}
        for k in names:
            v = coarse[k]
            if keep_idx is not None:
                v = v.index_select(0, keep_idx.to(v.device))
            phys[k] = v
        norm = model.normalizer.coarse.normalize(phys)
        packed = model.out_packer.pack(norm, axis=-3)  # (B, C_out, H_c, W_c)
        return interpolate(packed, model.downscale_factor)  # (B, C_out, H, W)

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

            # Interpolated-coarse base to add back for a residual teacher
            # (None otherwise); computed before sampling so lo_renoise can also
            # subtract it from the re-noise target.
            base_norm = self._base_prediction_norm(batch, keep_mask)

            # Produce the student output ensemble (B, n, C_out, H, W) per the
            # configured validation mode (see student_sampling).
            output_norm = self._sample_student_output(
                student, condition, teacher_phys, base_norm, B, H, W, C_out
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
                direction = self._tail_directions.get(var, "upper")
                reference = self._tail_references.get(var, 0.0)
                q = _tail_quantile_level(pct, direction)
                student_p = _quantile_from_histogram(hist_student[var], lo, hi, q)
                target_p = _quantile_from_histogram(hist_target[var], lo, hi, q)
                if target_p is None or student_p is None:
                    continue
                # Ratio of anomalies (distance from ``reference`` in the tail
                # direction) so PRMSL's ~1000 hPa offset can't wash out deep-low
                # errors; reference 0 recovers the raw-value ratio.
                student_mag = _tail_magnitude(student_p, reference, direction)
                target_mag = _tail_magnitude(target_p, reference, direction)
                if target_mag <= 0:
                    continue
                by_var[var] = student_mag / target_mag
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
