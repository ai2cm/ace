# Distillation experiment workflow

A repeatable playbook for training distilled downscaling students and reporting
findings. Four phases: **implementation → configuration → experiment →
report-out**. The central log is [`LOG.md`](LOG.md); per-run reports live in
[`reports/`](reports/); templates are in [`templates/`](templates/).

> Historical MoE-distillation notes are frozen in
> [`../MOE_DISTILLATION_STATUS.md`](../MOE_DISTILLATION_STATUS.md). New work is
> tracked here.

> **Agent note — capture recurring operations.** If an assistant finds itself
> repeating a step of this workflow that isn't already captured under
> `experiments/` (i.e. not in this file, `LOG.md`, `reports/`, `templates/`, or
> `../specs/`) — a manual fetch, a fix-up applied to every generated report, a
> command variant run each time, an environment/auth quirk worked around — fold it
> back into the durable artifacts so it survives across sessions: a new step or note
> here, a change to `check_runs.py`, an update to a template, or a spec. Nothing that
> gets done more than once should live only in an assistant's transcript.

---

## Phase 1 · Implementation

Code changes to the distillation method / model / loss.

- Branch `feature/<slug>` off `experiment/fastgen-distill`. Add tests.
- Durable pipeline changes (not just an experiment knob) get a numbered spec under
  [`../specs/`](../specs/) first, registered in `../specs/README.md`.
- **Ends when the change is merged onto `experiment/fastgen-distill` and pushed** —
  gantry clones the *pushed* commit, so an unpushed commit cannot be launched.

## Phase 2 · Configuration

Decide the run's knobs and record the intent *before* launch.

- Method: `fdistill` | `dmd2` | `scm`. Teacher: default single-model or `--moe-teacher`.
- Knobs are `run.sh` flags → `ACE_*` env vars read by the spike config
  (`configs/fdistill_kl_spike.py` etc.): `--gan-weight`, `--gan-r1`,
  `--spectral-weight` / `--spectral-band-gamma` / `--spectral-min-wavenumber`,
  `--student-steps`, `--lr-decay-steps`, `--disc-feature-depth`, `--expert`.
- Copy [`templates/run_report.md`](templates/run_report.md) to
  `reports/<YYYY-MM-DD>-<suffix>-<TBD>.md`, and fill the **hypothesis** + the exact
  launch command + the (pushed) commit. Rename with the wandb id once known.

## Phase 3 · Experiment

Launch and capture the artifact quartet.

```
conda run -n fme bash configs/experiments/2026-05-18-distillation-with-val/run.sh \
    <method> --suffix <suffix> [knobs...]
```

- gantry prints the **Beaker experiment id + URL** and the resolved **commit/branch**
  to stdout — record them. The **experiment name** is the `JOB_NAME`
  (`ace-downscaling-distillation-<method>-with-val-<suffix>[-moe-teacher][-expertN]`).
- The **wandb id + URL** appears in the job logs (`beaker experiment logs <ULID> |
  grep 'View run at'`).
- **Verify it reached the training loop** before walking away:
  ```
  beaker experiment logs <ULID> | grep -iE "input_shape|Spectral matching|gan_loss_weight_gen|iter 1|error|traceback"
  ```
  Expect `input_shape ... C_out=N`, any enabled-loss lines, and an `iteration 1`
  log with no traceback.
- Add a **registry row** to `LOG.md` with `state=running`
  (`check_runs.py --registry-row <id>` prints it).

> Launching from a git worktree: gantry needs a remote tracking branch. If the
> worktree branch has none, point it at the pushed remote branch with the same tip:
> `git branch --set-upstream-to=origin/<branch> <worktree-branch>`.

## Phase 4 · Report-out

When there is enough history (or the run finishes/crashes):

1. **Generate the filled report:**
   ```
   conda run -n fme python -m fme.downscaling.distillation.check_runs \
       --report <wandb_id> --beaker <ULID> --out fme/downscaling/distillation/experiments/reports/
   ```
   This pulls the three indicator families (training behavior + GAN health + loss
   domination; tails per variable; spectrum hi/mid/lo per variable), applies the
   heuristics, and fills everything except the **Verdict**. The Artifacts table now
   includes the run's git commit as a GitHub link (auto-pulled from wandb metadata).
   > **`--beaker` for an old run whose experiment ULID you've lost:** if you only
   > have a checkpoint *dataset* ULID (e.g. from a bundle config), resolve it to the
   > experiment with `beaker dataset get <dataset-ULID> --format json` →
   > `.sourceExecution` (a job ULID) → `beaker job get <job-ULID> --format json` →
   > `.execution.experiment`.
2. **Write the Verdict**: win / flat / degrade vs baseline, and the recommended
   checkpoint — pick a **mid-training** checkpoint if `best@frac` is well before the
   end (the checkpoint-selection trap: `best_student.ckpt` by CRPS is often
   spectrally worse than a mid-training one).
3. **Update `LOG.md`**: set the registry row's verdict + best-ckpt, and add a
   one-line bullet to the Outcomes log linking the report.
4. If a later run invalidates an earlier finding, **prepend a ⚠️ caveat** to the
   superseded report/row and link forward — do not silently rewrite history.

### Eval-bundle comparisons

For a held-out eval of a distilled bundle vs its teacher (project
`andrep-downscaling`, e.g. `configs/experiments/2026-07-07-distilled-moe-eval/`):

```
conda run -n fme python -m fme.downscaling.distillation.check_runs \
    --compare-eval <teacher_wandb> <distilled_wandb> --project andrep-downscaling \
    --out .../reports/
```

Figures (histograms / spectra) are rendered separately from the per-event netCDFs:
`scripts/downscaling/utils.fetch_beaker_dataset` →
`scripts/downscaling/plot_compared_histograms.py` / `plot_beaker_histograms.py`.

---

## Indicator → metric reference

Authoritative val keys come from `best_student_callback._log_to_wandb`; eval keys
from the evaluator aggregators.

| Indicator | Training-run keys (`val/…`, `train/…`) | Eval-bundle keys |
|---|---|---|
| GAN health | `train/gan_loss_gen`, `train/gan_loss_disc`, `train/gan_loss_ar1`, `train/fake_score_loss` | — |
| Loss domination | `train/f_distill_loss`, `train/spectral_loss[_weighted]`, `train/total_loss` | `loss` |
| CRPS | `val/crps_<VAR>`, `val/crps_mean`, `val/crps_best` | `metrics/crps/<VAR>` |
| Tails | `val/tail_99.99_<VAR>`, `val/tail_99.9999_<VAR>`, `_mean` | `histogram/prediction_frac_of_target/<pct>th-percentile/<VAR>` |
| Power spectrum | `val/spec_mae_{lo,mid,hi}_<VAR>`, `val/spec_mae_mean`, figure `val/power_spectrum/<VAR>` | `power_spectrum/mean_abs_norm_bias/<VAR>` |

Reporting convention for a metric over training: **`first → best@frac → last`** —
where `best` minimizes error (spec_mae, crps) or is closest to 1.0 (tail ratio),
and `@frac` is the fraction of the run at which the best occurred.
