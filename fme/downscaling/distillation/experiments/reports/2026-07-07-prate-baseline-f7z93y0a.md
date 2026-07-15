# Run report — `ace-downscaling-distillation-fdistill-with-val-prate-baseline`

_Reference run: GAN-only f-distill (no spectral loss). The baseline the spectral
arms are measured against._

## Artifacts

| | |
|---|---|
| Experiment name | `ace-downscaling-distillation-fdistill-with-val-prate-baseline` |
| wandb run | `f7z93y0a` — https://wandb.ai/ai2cm/fastgen/runs/f7z93y0a |
| Beaker experiment | `01KWX5CVJQ2BP53VH95WKPVPED` — https://beaker.org/ex/01KWX5CVJQ2BP53VH95WKPVPED |
| Commit | `26868ca` — https://github.com/ai2cm/ace/commit/26868ca |
| State / last step | `crashed` @ `29510` |

## Config

- **Method:** fdistill
- **GAN generator weight (assumed for ratios):** 0.001
- **Suffix:** `prate-baseline`

> **Metric selection:** validation metrics (§1 val lines, §2, §3) are read at the
> **`best_student_tail` checkpoint — step 2470** (~8% of the run; where
> `val/tail_best_score` bottomed at 0.00272, which also coincides with the run's best
> `val/crps_mean`). That is the checkpoint bundled into the held-out X-SHiELD AMIP eval
> (`flzvb6tp`/`fg9byv9y`). Training/loss metrics (§1 loss table, GAN health) are read at
> the **end of run** (last logged step 29380). Trajectories are shown as
> `first → best_tail@2470 → last`.

## 1 · Training behavior

- **GAN health:** `healthy` (end of run) — gen 1.35, disc 1.09; both engaged, no
  collapse throughout.
- **Loss domination** (end of run, `gan_weight=1e-3`):

  | term | last | note |
  |---|---|---|
  | `train/f_distill_loss` | 0.0761 | dominates the generator objective (expected) |
  | `train/gan_loss_gen` | 1.354 | ×weight 1e-3 = 0.00135; ~0.018× f_distill |
  | `train/fake_score_loss` | 0.0257 |  |
  | `train/gan_loss_disc` | 1.089 | discriminator |
  | `train/total_loss` | 0.907 |  |

  _f_distill_loss dominates the generator objective; GAN term is a small fraction. No
  spectral constraint — the GAN carries all small-scale energy._

- **`val/crps_mean`:** 0.1139 → **0.1046** → 0.1589 (best_tail is also the run's CRPS
  minimum; +52% drift best→last).
- **`val/spec_mae_mean`:** 1.035 → **0.359** → 0.981 (+173% drift best→last).

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under. Values at
`best_student_tail` (step 2470).

| variable | tail_99.99 | tail_99.9999 |
|---|---|---|
| PRATEsfc | 0.764 → **1.033** → 3.163 | 0.744 → **0.997** → 2.665 |

Worst tail: **PRATEsfc** (only variable). At best_tail both tails are near-ideal
(~1.0); by end of run extremes are grossly over-produced (3.16 / 2.66).

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better). Values at `best_student_tail`
(step 2470).

| variable | lo | mid | hi |
|---|---|---|---|
| PRATEsfc | 1.203 → **0.255** → 1.054 | 1.270 → **0.483** → 1.020 | 0.635 → **0.338** → 0.869 |

Worst spectrum: **mid band** (0.483 at best_tail) · PSD figures:
`val/power_spectrum/<VAR>` in wandb.

## Verdict

- **Outcome vs baseline:** n/a — this *is* the baseline. Characteristics for the
  spectral arms to beat, **read at the deployed `best_student_tail` checkpoint (step
  2470)**: near-ideal tails (`tail_99.99` 1.03, `tail_99.9999` 1.00) but a mediocre
  spectrum (`spec_mae` lo 0.26 / mid 0.48 / hi 0.34) — no spectral constraint, so the
  GAN alone cannot pin small-scale energy. **Note:** the run drifts badly if trained to
  the end (tail_99.99 → 3.16, spec_mean → 0.98 by step 29510); the earlier "tail ~3.16 /
  spec_hi ~0.87" figures describe that drifted late state, **not** the checkpoint that
  was actually evaluated. Tail-based checkpoint selection is what rescues this run.
- **Recommended checkpoint:** `best_student_tail` @ step 2470 (the one used in the
  X-SHiELD eval; near-ideal tails and best CRPS/spectrum of the run).
- **Next action:** compare all spectral arms against this run at their own best_tail
  checkpoints. The severe best→last drift (spec +173%, tail overshoot) also motivates
  spectral early-stopping (spec 13).

## Caveats

- ⚠️ _Prepend here if a later run invalidates this one._
