# 07 — Documentation consolidation

## Goal

One accurate, current set of docs: a user-facing "how to run distillation"
that matches the post-productionization interface, and the maintainer-facing
architecture doc.  Stale spike-era instructions are gone.

## Current state (verified)

- `../README.md` — spike-era runbook.  Known-stale or soon-stale content:
  - "FastGen submodule / PYTHONPATH" prerequisites (replaced by the
    `[distillation]` extra, spec 01);
  - `pip install hydra-core boto3 torchvision` manual step (folds into the
    extra);
  - env-var table (`ACE_*` vars die in specs 02–03);
  - dryrun expected-output block shows a 7-channel example
    (`real: (2, 7, 512, 512)`) from an older teacher;
  - all `torchrun ... fastgen_train --config <spike>.py` invocations
    (replaced by the YAML entrypoint, spec 03);
  - W&B `log_config` override instructions (move into the YAML schema docs);
  - Step 4 (manual `save_student_checkpoint` snippet) — superseded by
    `BestStudentCheckpointCallback` writing `best_student{,_tail}.ckpt`
    automatically when validation is configured; keep the manual path
    documented as a fallback only.
- `../ARCHITECTURE.md` — written 2026-06-10, accurate for the spike.
  Sections that must be updated by the specs that change them (each spec's
  PR must touch this file):
  - "Component map" (specs 03/04/05 rename files),
  - "the three places sigma parameters live" → layer 2 becomes automatic
    (spec 02),
  - "Config override mechanics" (spec 03),
  - env-var discussion and `run.sh` references (specs 02/05).
- This `specs/` folder — delete each spec as it is completed (git history
  preserves them), or convert to GitHub issues if the team prefers; do not
  let completed specs rot here.

## Tasks

1. After specs 01–05 land, rewrite `../README.md` around the new flow:
   install extra → write/choose a `DistillationConfig` YAML → submit via
   the spec-05 launcher (cluster) or `torchrun -m
   fme.downscaling.distillation.train config.yaml` (manual) → evaluate
   `best_student.ckpt` with the standard downscaling inference pipeline.
   Keep: the method comparison table, the go/no-go tail metric
   (`generation/histogram/prediction_frac_of_target/99.9999th-percentile`,
   within ~5% of teacher), troubleshooting entries that still apply.
2. Reconcile `ARCHITECTURE.md` with the final shape (the per-teacher
   parameter table stays — it documents *checkpoint contents*, which remain
   true regardless of auto-derivation).
3. Decide whether distillation gets a page in the Sphinx docs
   (`docs/`, extras group `docs` in `pyproject.toml`).  Recommendation:
   yes, a single page summarizing capability + linking to the in-repo
   README, since the public package now exposes the `[distillation]`
   extra.  Follow existing docs structure; don't duplicate the README.
4. Check `fme/downscaling/distillation/generate_val_dataset.py` — it is
   documentation-as-module (41 lines, docstring describing how to use the
   standard inference pipeline to build the val zarr).  Fold its content
   into the README's validation section and delete the module, or keep it
   if anything imports it (`grep -rn generate_val_dataset` first; run.sh
   comments reference it conceptually only).

## Acceptance criteria

- A new team member can run an MoE-teacher fdistill job following only
  `README.md`, with no reference to env vars or spike configs.
- `grep -rn "submodule\|PYTHONPATH\|ACE_SIGMA\|ACE_NOISE_DIST\|ACE_C_OUT"
  fme/downscaling/distillation/` → no hits outside git history.
- ARCHITECTURE.md statements spot-checked against the merged code (no
  references to deleted files or env vars).
