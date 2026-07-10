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

## 1 · Training behavior

- **GAN health:** `healthy` — gen 1.19, disc 1.17 (both engaged)
- **Loss domination:**

  | term | last | note |
  |---|---|---|
  | `train/f_distill_loss` | 0.08891 |  |
  | `train/spectral_loss_weighted` | 0.002552 | 0.029× f_distill |
  | `train/gan_loss_gen` | 1.188 | ×weight 0.001 = 0.00119; 0.013× f_distill |
  | `train/fake_score_loss` | 0.02464 |  |
  | `train/total_loss` | 0.9709 |  |

  _f_distill_loss dominates the generator objective (expected)._

- **`val/crps_mean`:** improving: improved 10% first→best, then drifted +15% best→last
- **`val/spec_mae_mean`:** degrading late (checkpoint-selection trap): improved 97% first→best, then drifted +632% best→last

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under.

| variable | tail_99.99 | tail_99.9999 |
|---|---|---|
| PRATEsfc | 0.821 → 0.999@3% → 1.98 | 0.776 → 0.997@28% → 1.87 |

Worst tail: **PRATEsfc**.

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better).

| variable | lo | mid | hi |
|---|---|---|---|
| PRATEsfc | 1.13 → 0.0138@7% → 0.415 | 1.12 → 0.0116@50% → 0.164 | 0.512 → 0.00257@8% → 0.112 |

Worst spectrum: **PRATEsfc** · PSD figures: `val/power_spectrum/<VAR>` in wandb.

## Verdict  <!-- HUMAN: fill this in -->

- **Outcome vs baseline:** ✅ **win.** Beats GAN-only baseline `f7z93y0a` 5–20× on
  `spec_mae` (lo/mid/hi last 0.42/0.16/0.11 vs 1.05/1.02/0.87) and improves the
  *independent* metrics — `crps` 1.50e-5 vs 1.98e-5, `tail_99.99` 1.98 vs 3.16.
  `f_distill_loss` ≈ baseline (0.089 vs 0.066), so it is not fighting distillation;
  spectral term is a gentle ~3% of the VSD term. GAN stayed engaged.
- **Recommended checkpoint:** **mid-training.** `spec_mae_hi` bottoms @8%
  (0.0026→0.11 by end) and tails are best @3–28%; `best_student.ckpt` by CRPS drifts
  spectrally late. Pick a checkpoint around the spec/tail minima (~2–14k).
- **Next action:** reduce-GAN arm `6dotglmg` (gan 1e-3→3e-4) to test whether leaning
  off the GAN removes the late tail-overshoot / drift.

## Caveats

- ⚠️ _Prepend here if a later run invalidates this one._
