# 13 — Spectral-aware early stopping + checkpoint selection

> **Status: implemented (2026-07-15; proposed 2026-07-13).** Motivated by run `xgcaf2rt` (mid+hi spectral
> arm) — see
> [`../experiments/reports/2026-07-10-prate-spectral-midhi-xgcaf2rt.md`](../experiments/reports/2026-07-10-prate-spectral-midhi-xgcaf2rt.md).
> Research/robustness spec, independent of the 01–12 refactor series.

## Problem

Distilled-student runs waste large amounts of compute and their saved checkpoints
often miss the spectral optimum. Concretely, from `xgcaf2rt` vs its baseline
`i26sidsm`:

- **Wasted compute.** `xgcaf2rt` ran to **52k steps**; its useful spectral optimum
  was at **~2.6k**. After the optimum the run only drifts: `val/spec_mae_mean`
  best→last +691%, `val/tail_99.99_PRATEsfc` → 2.2. There is no early stop.
- **The existing selectors are decoupled from spectral quality.**
  `BestStudentCheckpointCallback` saves two checkpoints
  (`best_student_callback.py:418-457`):
  - `best_student.ckpt` ← argmin `val/crps_mean`. But `crps_mean` is **flat to ~1%**
    across the whole useful region (midhi min 0.1047 vs base 0.1045), so its argmin
    step is essentially noise-determined.
  - `best_student_tail.ckpt` ← argmin `|tail_99.9999 − 1|`. Landed at **3%** of the
    run for midhi vs **29%** for base — un-converged in the first case.
  Meanwhile `spec_mae` swings 5–10× across those steps. Result: the spectrally-best
  saved checkpoint is a *different* file in each run (midhi ← CRPS ckpt; base ← tail
  ckpt), and cross-run comparison is dominated by which selector happened to coincide
  with the spectral optimum. Every analysis in `../experiments/` ends up hand-picking a
  mid-training checkpoint because **no selector tracks the spectrum.**

## Goal

Add a spectral-based checkpoint selector **and** a patience-based early stop to
`BestStudentCheckpointCallback`, so that (a) runs stop once the spectrum stops
improving (saving compute), and (b) there is a saved checkpoint that actually sits at
the spectral optimum — making every future arm's baseline comparison honest (all runs
select/stop at their own spectral optimum instead of an arbitrary flat-CRPS argmin).

## Proposed design

In `best_student_callback.py` (`fme/downscaling/distillation/best_student_callback.py`):

1. **`best_student_spec.ckpt` selector.** Mirror the existing CRPS/tail blocks: track
   `self._best_spec = inf`; the selection metric is `val/spec_mae_mean` (already
   computed — `_log_to_wandb` derives it at line ~505 as
   `mean(m["mae"] for m in spec_by_var.values())`; hoist that into the selection path).
   On strict improvement, `save_student_checkpoint(...)` to a new
   `best_spec_checkpoint_path` ctor arg, and log `val/spec_best`.
   - **Noise guard:** `spec_mae_mean` is spiky (single-snapshot 5–10× swings). Select
     on a **rolling median** over the last `spec_patience_window` validations (default
     ~5, ≈650 steps) rather than the raw per-val value, so a lucky spike doesn't win.
2. **Patience early stop.** Track validations since the last spec improvement; when it
   exceeds `spec_patience` (default e.g. 10 validations), request training to stop.
   Emit a clear log line with the best step + value.
3. **Config knob** following the existing pattern: `run.sh --early-stop-patience` →
   `ACE_EARLY_STOP_PATIENCE` read in the spike config (`configs/fdistill_kl_spike.py`),
   threaded into the callback ctor. Default **off** (patience `None`) to preserve
   current behavior for backward compat; opt-in per run.

## ★ Open question — how to actually stop the FastGen loop (verify FIRST)

The training loop is FastGen's `Trainer` (an **unmodified** NVlabs clone — per
`../ARCHITECTURE.md`, never patch it; ACE behavior goes in adapters). The callback
implements FastGen `Callback` hooks (`fastgen_train.py:145` `on_save_checkpoint_success`,
with a `__getattr__` no-op stub for the rest). **Before implementing, verify what
mechanism FastGen's `Trainer` honors for stopping**, e.g.:
- a `Callback` hook return value or a `should_stop`/`stop_training` flag the loop polls;
- a hook that can lower the loop's `total_kimg`/max-iter bound at runtime;
- raising a dedicated `StopTraining` exception the entrypoint catches to still run the
  clean shutdown/checkpoint-flush path (least invasive if no native hook exists).

If none exists cleanly, the fallback is an entrypoint-level guard in `fastgen_train.py`
that checks the callback's stop flag after each validation and breaks — kept in the ACE
adapter layer, not in `Fastgen/`. **Record the finding in this spec before coding.**

### Resolution (verified 2026-07-15, implemented)

FastGen exposes a **designed** extension point that requires no patch: `Trainer(config,
auto_resume=...)` accepts an `AutoResumeInterface` (`FastGen/fastgen/utils/autoresume.py`).
The loop polls `self.auto_resume.termination_requested()` **every iteration** via
`auto_resume_exit` (`FastGen/fastgen/trainer.py:213`, def at `:484`); when it returns True
the trainer runs its clean-shutdown callbacks (`on_train_end`/`on_app_end`) and returns.
It also performs the DDP rank-0 → all-ranks broadcast itself (`trainer.py:500-516`).

Implementation: `BestStudentCheckpointCallback` tracks a `_stop_requested` flag (set only
on rank-0, inside `_record_validation`) and exposes `should_stop()`. `fastgen_train.py`
injects a small duck-typed `_EarlyStopAutoResume` wrapping the callback whose
`termination_requested()` returns `should_stop()` and whose `request_resume`/`init`/
`get_resume_details` are no-ops (so termination is a clean stop, not a requeue). Zero
changes to `FastGen/`.

Why this over the two fallbacks considered:
- **Raising a `StopTraining` exception** (the spec's least-invasive suggestion) is unsafe
  under DDP: one rank raising mid-NCCL-collective deadlocks the others; it also skips the
  clean-shutdown/checkpoint-flush path. The AutoResume route is DDP-safe by construction.
- **Mutating `config.trainer.max_iter` at runtime does not work** — the loop's
  `range(iter_start+1, max_iter)` is materialized once when the `for` begins, so later
  mutation has no effect.

**Deviation from §Proposed design (per 2026-07-15 decision):** the early-stop *condition*
is broadened from spec-metric-only to **any-improved** — the patience counter resets when
*any* selector (CRPS, tail, or the new spectral rolling-median) improves, and stops only
once none is improving. This is the more conservative policy (won't cut a run while a
deployed `best_student_tail.ckpt` is still improving). The spectral selector +
`best_student_spec.ckpt` (item 1) are implemented as specified and also feed this signal.
Default patience `10` validations, rolling window `5`; opt-in via `--early-stop-patience` /
`ACE_EARLY_STOP_PATIENCE` (0/unset = off) and `--spec-patience-window` /
`ACE_SPEC_PATIENCE_WINDOW`, threaded through `fastgen_train.py` (not the spike config, to
match the existing `--val-*` callback-arg plumbing).

## Tests (`best_student_callback` unit tests)

- Spec selector saves on rolling-median improvement, not on a single spike (feed a
  spiky sequence; assert the checkpoint is written at the sustained min, not the spike).
- Patience counter resets on improvement and triggers the stop request after
  `spec_patience` non-improving validations.
- Backward compat: patience `None` + no `best_spec_checkpoint_path` → behavior
  byte-identical to today (no new checkpoint, no stop).

## Acceptance criteria

- A run with `--early-stop-patience N` stops within ~N validations of its spectral
  optimum and leaves a `best_student_spec.ckpt` at (near) the running-min
  `spec_mae_mean`.
- The three selectors (`crps`, `tail`, `spec`) coexist; teacher/other paths unchanged.
- FastGen (`Fastgen/`) is untouched; the stop mechanism lives in the ACE adapter.

## Out of scope

- Multi-variable weighting of the spectral selection metric (single `spec_mae_mean`
  for now; per-var weighting can follow the `variable_weights` experiment work).
- Changing the CRPS/tail selectors — this only *adds* a spectral one.
