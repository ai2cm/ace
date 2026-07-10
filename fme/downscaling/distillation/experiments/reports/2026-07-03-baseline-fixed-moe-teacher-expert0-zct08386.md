# Run report — `ace-downscaling-distillation-fdistill-with-val-baseline-fixed-moe-teacher-expert0`

_Hypothesis: per-expert **Student-Lo** (teacher expert 0, σ∈[0.005,200],
"baseline-fixed" config), the low-noise specialist that finishes the 2-step cascade
and carries the fine-scale content. One of the two base models bundled (with
Student-Hi `4mez4kmn`) into the distilled 2-step MoE `rmoodemk`._

## Artifacts

| | |
|---|---|
| Experiment name | `ace-downscaling-distillation-fdistill-with-val-baseline-fixed-moe-teacher-expert0` |
| wandb run | `zct08386` — https://wandb.ai/ai2cm/fastgen/runs/zct08386 |
| Beaker experiment | `01KWJAFKZ96YBR73F0TETBKC0Q` — https://beaker.org/ex/01KWJAFKZ96YBR73F0TETBKC0Q |
| Commit | `184fa29` — https://github.com/ai2cm/ace/commit/184fa298b6dadad9ad40252d83e0d697b73d0c84 |
| State / last step | `crashed` @ `46800` |

## Config

- **Method:** fdistill
- **GAN generator weight (assumed for ratios):** 0.001
- **Suffix:** `baseline-fixed-moe-teacher-expert0`

## 1 · Training behavior

- **GAN health:** `healthy` — gen 0.844, disc 1.3 (both engaged)
- **Loss domination:**

  | term | last | note |
  |---|---|---|
  | `train/f_distill_loss` | 0.07852 |  |
  | `train/gan_loss_gen` | 0.8439 | ×weight 0.001 = 0.000844; 0.011× f_distill |
  | `train/fake_score_loss` | 0.0387 |  |
  | `train/total_loss` | 1.091 |  |

  _f_distill_loss dominates the generator objective (expected)._

- **`val/crps_mean`:** flat: improved 2% first→best, then drifted +1% best→last
- **`val/spec_mae_mean`:** degrading late (checkpoint-selection trap): improved 86% first→best, then drifted +66% best→last

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under.

| variable | tail_99.99 | tail_99.9999 |
|---|---|---|
| PRATEsfc | 0.868 → 1@16% → 1 | 0.974 → 1@13% → 1.13 |
| PRMSL | 1.01 → 1@88% → 0.998 | 1.14 → 1.01@89% → 1.21 |
| eastward_wind_at_ten_meters | 0.952 → 1@96% → 1 | 1.12 → 1.01@89% → 1.21 |
| northward_wind_at_ten_meters | 0.966 → 1@40% → 1 | 1.09 → 1@20% → 1.16 |

Worst tail: **eastward_wind_at_ten_meters**.

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better).

| variable | lo | mid | hi |
|---|---|---|---|
| PRATEsfc | 0.89 → 0.0176@93% → 0.135 | 1.2 → 0.0104@99% → 0.208 | 0.631 → 0.0462@94% → 0.23 |
| PRMSL | 0.0249 → 0.00409@92% → 0.00917 | 0.178 → 0.00845@89% → 0.0168 | 0.386 → 0.00998@26% → 0.0256 |
| eastward_wind_at_ten_meters | 0.321 → 0.0279@30% → 0.0744 | 0.455 → 0.101@88% → 0.138 | 0.0825 → 0.0439@82% → 0.0778 |
| northward_wind_at_ten_meters | 0.33 → 0.0103@30% → 0.0732 | 0.318 → 0.0733@89% → 0.125 | 0.0448 → 0.0189@89% → 0.024 |

Worst spectrum: **PRATEsfc** · PSD figures: `val/power_spectrum/<VAR>` in wandb.

## Verdict

- **Outcome vs baseline:** ✅ **the low-noise expert distills cleanly.** GAN stayed
  `healthy` (gen 0.84, disc 1.3 — engaged, no collapse) and `f_distill_loss` is low
  (0.079). Spectra reach excellent per-band minima mid/late (PRATEsfc lo/mid/hi
  ~0.02/0.01/0.05; PRMSL <0.01) and tails sit near 1.0 across all four variables —
  this is the expert that supplies the bundle's fine-scale detail.
- **Recommended checkpoint:** mid/late but **before the final drift**. `spec_mae`
  bottoms near the end (best@~90%+) then drifts up (`spec_mae_mean` +66% best→last)
  and the finest tails overshoot to ~1.1–1.2 — the checkpoint-selection trap. Pick a
  checkpoint near the per-variable spectral minima (~step 43–46k), not the crashed
  last step. The bundle used `best_student_tail.ckpt`.
- **Next action:** bundled with Student-Hi `4mez4kmn` and evaluated end-to-end — see
  the eval comparison [report](2026-07-08-moe-eval-distilled-vs-teacher.md).

## Caveats

- Derived from the frozen
  [`MOE_DISTILLATION_STATUS.md`](../../MOE_DISTILLATION_STATUS.md); numbers above
  regenerated from wandb.
- ⚠️ _Prepend here if a later run invalidates this one._
