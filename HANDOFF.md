# Handoff: residual-normalized prediction + hybrid residual steppers

**Branch**: `troya/hybrid-residual-prediction-names` (one feature commit on
`origin/main`). **Goal**: open a PR to ai2cm/ace main. **Remove this file
before the PR is marked ready** (it is a working note, not repo content).

## What the change is

One commit touching three files (all `fme/`, ~200 added lines, no config or
script changes):

1. `fme/core/step/single_module.py` — two new `SingleModuleStepConfig`
   options:
   - `residual_normalized_prediction` (bool): with residual prediction, the
     network's prognostic outputs are tendencies in residual-normalized
     units; the step rescales them by `residual_std / field_std` in
     normalized space before adding to the input. **Scale-only, never
     means**: the stats convention pairs full-field centering with tendency
     stds — an early version that added residual means injected a ~291 K/step
     temperature drift. Do not "fix" the transform to include means.
   - `residual_prediction_names` (list | None): restricts the residual path
     to a subset of prognostics; the rest are full-field. This enables
     hybrid steppers. `get_loss_normalizer` follows the same per-variable
     convention (residual-stepped names scored in tendency-std units,
     full-field names in full-field-std units) — without that, full-field
     state errors are inflated by `(field_std / tendency_std)^2`, which in
     practice was a 1,600x too-large starting loss.
2. `fme/core/step/test_step.py` — `test_step_with_adjustments_hybrid_residual_names`
   (subset gets input + scaled output; others full-field; default = all).
3. `fme/ace/stepper/test_single_module.py` — end-to-end residual-normalized
   step test + config-validation test.

## Why it exists (motivation for the PR description)

Developed for a coupled ocean-atmosphere emulator (SamudrACE on CM4) where
tendency-trained oceans strongly outperform full-field training on ENSO
forecast skill but the all-residual variant is rollout-unstable. On a
240-initial-condition verification, the hybrid (temperature-only residual)
matches the all-residual model's skill (Nino3.4 ACC 0.57/0.39 vs 0.59/0.34
at leads 6/9, paired CIs overlapping; baseline 0.44/0.20) while producing
finite 20-year rollouts with the correct ENSO band power (ratio 0.999) at
every training epoch. Report (internal): reports repo branch
`troya/2026-09-09-tendency-hybrid-ocean-240ic`. Training runs: wandb
ai2cm/ace-samudra-cm4 runs `b44mgz9u` (all-residual), `ippq0ioi` (hybrid).

## State: verified so far

- `pytest fme/core/step/test_step.py fme/ace/stepper/test_single_module.py
  fme/core/test_normalizer.py` → all pass (fme conda env, CPU).
- Pre-commit hooks (ruff, ruff-format, mypy) pass on the commit.
- The port preserves main's determinism work (frozenset name properties,
  sorted() normalizer name lists) — the feature was developed on an older
  base and re-derived onto main by hand; the diff is purely additive except
  two replaced lines inside the feature itself.

## What remains (your job)

1. Run the full test suite the repo's CI runs (at least `fme/core` and
   `fme/ace/stepper`); fix anything the narrow selection missed.
2. Read the three-file diff as a reviewer. Known soft spots:
   - `step_with_adjustments` now takes `residual_names` and
     `residual_transform`; other callers of it exist
     (`fme/core/step/radiation.py`, `fme/core/step/secondary_module.py`) and
     rely on the defaults — confirm the defaults reproduce old behavior
     there (they should: `None` → all prognostics, no transform).
   - Config-serialization round trip: `from_state` uses dacite strict —
     confirm an old checkpoint (without the new keys) still loads
     (dataclass defaults should cover it; a small test would be welcome).
3. Draft the PR description from "Why it exists" above; end it with the
   attribution line below. Do NOT open the PR — the branch owner does that.
4. Delete HANDOFF.md in the final commit.

PR description footer (required):
🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Gotchas from the development history

- Never add residual means (see above).
- A `dacite.UnionMatchError` on an fme config can be a swallowed
  `__post_init__` ValueError — validation errors inside union members are
  masked; keep the new validations' messages precise.
- Downstream contract (worth a docstring sentence if you touch docs): any
  config that loads a checkpoint's *weights* but declares its own step
  config must declare the same `residual_prediction_names` the checkpoint
  was trained with — weights carry no output-convention semantics. We lost
  a training run to a coupled fine-tune config that omitted them.
