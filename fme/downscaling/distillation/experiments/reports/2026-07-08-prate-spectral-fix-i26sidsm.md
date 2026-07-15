# Run report — `ace-downscaling-distillation-fdistill-with-val-prate-spectral-fix`

_Hypothesis: does the corrected spectral-matching loss (teacher **sample** target,
spectrum-then-average) restore high-k power vs the GAN-only baseline `f7z93y0a`
without fighting distillation?_

## Artifacts

| | |
|---|---|
| Experiment name | `ace-downscaling-distillation-fdistill-with-val-prate-spectral-fix` |
| wandb run | `i26sidsm` — https://wandb.ai/ai2cm/fastgen/runs/i26sidsm |
| Beaker experiment | `01KX00N9SE3ZVQFHQJ54XS0TAP` — https://beaker.org/ex/01KX00N9SE3ZVQFHQJ54XS0TAP |
| Commit | `e29f797` — https://github.com/ai2cm/ace/commit/e29f797 |
| State / last step | `crashed` @ `27820` |

## Config

- **Method:** fdistill
- **Spectral weight (derived):** 0.01
- **GAN generator weight (assumed for ratios):** 0.001
- **Suffix:** `prate-spectral-fix`

> **Metric selection:** validation metrics (§1 val lines, §2, §3) are read at the
> **`best_student_tail` checkpoint — step 7930** (~28% of the run; where
> `val/tail_best_score` bottomed at 0.00268). That is the checkpoint bundled into the
> held-out X-SHiELD AMIP eval (`x2nyzmzh`/`l6vv7yx0`). Training/loss metrics (§1 loss
> table, GAN health) are read at the **end of run**. Trajectories are shown as
> `first → best_tail@7930 → last`. This matches the baseline report `f7z93y0a`, so §4's
> baseline comparison is checkpoint-matched (both at their own best_tail).

## 1 · Training behavior

- **GAN health:** `healthy` (end of run) — gen 1.19, disc 1.17 (both engaged).
- **Loss domination** (end of run):

  | term | last | note |
  |---|---|---|
  | `train/f_distill_loss` | 0.08891 | dominates the generator objective (expected) |
  | `train/spectral_loss_weighted` | 0.002552 | 0.029× f_distill (gentle) |
  | `train/gan_loss_gen` | 1.188 | ×weight 0.001 = 0.00119; 0.013× f_distill |
  | `train/fake_score_loss` | 0.02464 |  |
  | `train/total_loss` | 0.9709 |  |

  _f_distill_loss dominates; the spectral term is a gentle ~3% add-on, not fighting
  distillation. `f_distill_loss` ≈ the GAN-only baseline (0.089 vs 0.076)._

- **`val/crps_mean`:** 0.1163 → **0.1050** → 0.1206 (best_tail ≈ CRPS min 0.1045;
  +15% drift best→last).
- **`val/spec_mae_mean`:** 0.921 → **0.110** → 0.230 (best→last +109%; the per-run
  minimum is 0.031 mid-training — see the checkpoint-selection note in §3/§4).

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under. Values at
`best_student_tail` (step 7930).

| variable | tail_99.99 | tail_99.9999 |
|---|---|---|
| PRATEsfc | 0.821 → **1.058** → 1.977 | 0.776 → **0.997** → 1.866 |

Worst tail: **PRATEsfc** (only variable). At best_tail both tails are near-ideal
(1.06 / 1.00); by end of run they over-produce (1.98 / 1.87).

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better). Values at `best_student_tail`
(step 7930).

| variable | lo | mid | hi |
|---|---|---|---|
| PRATEsfc | 1.13 → **0.060** → 0.415 | 1.12 → **0.124** → 0.164 | 0.512 → **0.144** → 0.112 |

Worst spectrum: **hi band** at best_tail (0.144) · PSD figures:
`val/power_spectrum/<VAR>` in wandb.

⚠️ **Per-step spectral noise:** `spec_mae` is noisy val-to-val (neighboring vals at
7670–8190 span lo 0.06–0.15 / mid 0.02–0.12 / hi 0.008–0.14; step 7930 is a local
spike). The best_tail-step band values above carry that step-noise; the robust,
selector-independent statement is the ~3–4× improvement over the baseline's best_tail
(§4). The run's true spectral optimum (`spec_mae_mean` 0.031) lands mid-training and is
missed by both the CRPS and tail selectors.

## 4 · vs GAN-only baseline `f7z93y0a` (checkpoint-matched, each at its own best_tail)

Baseline best_tail = step 2470; spectral best_tail = step 7930. Both are the checkpoints
actually deployed to the held-out eval, so this is the honest apples-to-apples read.

| metric (PRATEsfc) | baseline @2470 | spectral @7930 | read |
|---|---|---|---|
| `spec_mae_lo` | 0.255 | **0.060** | spectral 4.2× better |
| `spec_mae_mid` | 0.483 | **0.124** | spectral 3.9× better |
| `spec_mae_hi` | 0.338 | **0.144** | spectral 2.3× better |
| `spec_mae_mean` | 0.359 | **0.110** | spectral 3.3× better |
| `tail_99.99` | 1.033 | 1.058 | ~tied (both ~ideal) |
| `tail_99.9999` | 0.997 | 0.997 | tied |
| `crps_mean` | 0.1046 | 0.1050 | tied |

The ~3–4× spectrum gain with tied tails/CRPS is the same direction and magnitude as the
held-out X-SHiELD AMIP eval (PSD bias 0.46→0.13, −71% CONUS; −78% maritime; CRPS ~3%
better) — the training-val win transfers out-of-sample.

## Verdict

- **Outcome vs baseline:** ✅ **win.** At the checkpoint actually deployed
  (`best_student_tail`, each run at its own), the corrected spectral loss beats the
  GAN-only baseline **~3–4× on `spec_mae`** (mean 0.11 vs 0.36; lo/mid/hi 0.06/0.12/0.14
  vs 0.26/0.48/0.34) while keeping the *independent* metrics tied — `crps_mean` 0.105 vs
  0.105, tails both near-ideal (~1.0). `f_distill_loss` ≈ baseline, so it is not fighting
  distillation; the spectral term is a gentle ~3% of the f_distill term. GAN stayed
  engaged. This is confirmed out-of-sample by the held-out X-SHiELD eval (−71–78% PSD
  bias). _(An earlier version of this verdict quoted a 5–20× win from **last-step**
  values; that overstated it by comparing the two runs' drifted end states — the
  checkpoint-matched best_tail gain is ~3–4×.)_
- **Recommended checkpoint:** `best_student_tail` @ step 7930 (the one used in the eval;
  near-ideal tails, ~3–4× baseline spectrum). Note the run's true spectral optimum
  (`spec_mae_mean` 0.031) sits mid-training and is missed by both CRPS- and tail-based
  selectors — a spectral-aware selector (`best_student_spec`, planned) would capture it.
- **Next action:** reduce-GAN arm `6dotglmg` (gan 1e-3→3e-4) to test whether leaning
  off the GAN removes the late tail-overshoot / drift.

## Caveats

- ⚠️ _Prepend here if a later run invalidates this one._
- ⚠️ `spec_mae` is noisy per validation; the best_tail-step band values (§3) carry
  step-noise. Read the ~3–4× baseline ratio (§4), not the absolute band value, as the
  headline. Both runs' spectral optima are missed by the current CRPS/tail selectors.
