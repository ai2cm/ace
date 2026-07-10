# Run report — `ace-downscaling-distillation-fdistill-with-val-hi-1step-moe-teacher-expert1`

_Hypothesis: first per-expert **Student-Hi** (teacher expert 1, σ∈[200,2000], 1-step,
GAN 1e-3 / R1 off), validated end-to-end via `hi_cascade` through the frozen
Student-Lo. Tests whether a coarse critic can supervise the high-σ expert — and
whether the cascade metric can even see Hi's contribution. One of the two base
models bundled (with Student-Lo `zct08386`) into the distilled 2-step MoE
`rmoodemk`._

## Artifacts

| | |
|---|---|
| Experiment name | `ace-downscaling-distillation-fdistill-with-val-hi-1step-moe-teacher-expert1` |
| wandb run | `4mez4kmn` — https://wandb.ai/ai2cm/fastgen/runs/4mez4kmn |
| Beaker experiment | `01KWTXGADFPB4GKVZ33C7ZGJP4` — https://beaker.org/ex/01KWTXGADFPB4GKVZ33C7ZGJP4 |
| Commit | `e920ca7` — https://github.com/ai2cm/ace/commit/e920ca7f425be97fbbfbddae7a700b97ac04e536 |
| State / last step | `crashed` @ `20280` |

## Config

- **Method:** fdistill
- **GAN generator weight (assumed for ratios):** 0.001
- **Suffix:** `hi-1step-moe-teacher-expert1`

## 1 · Training behavior

- **GAN health:** `not-engaged` — gen≈ln2 (0.693), disc≈2·ln2 (1.39) — discriminator stuck at chance
- **Loss domination:**

  | term | last | note |
  |---|---|---|
  | `train/f_distill_loss` | 0.8481 |  |
  | `train/gan_loss_gen` | 0.6933 | ×weight 0.001 = 0.000693; 0.00082× f_distill |
  | `train/fake_score_loss` | 0.0154 |  |
  | `train/total_loss` | 1.291 |  |

  _f_distill_loss dominates the generator objective (expected)._

- **`val/crps_mean`:** flat: improved 0% first→best, then drifted +0% best→last
- **`val/spec_mae_mean`:** flat: improved 1% first→best, then drifted +1% best→last

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under.

| variable | tail_99.99 | tail_99.9999 |
|---|---|---|
| PRATEsfc | 0.895 → 0.906@81% → 0.905 | 0.961 → 1@88% → 1.04 |
| PRMSL | 0.979 → 0.98@23% → 0.978 | 1.1 → 1.07@48% → 1.1 |
| eastward_wind_at_ten_meters | 0.968 → 0.97@52% → 0.968 | 1.16 → 1.11@36% → 1.15 |
| northward_wind_at_ten_meters | 0.978 → 0.979@40% → 0.978 | 1.08 → 1.06@67% → 1.08 |

Worst tail: **eastward_wind_at_ten_meters**.

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better).

| variable | lo | mid | hi |
|---|---|---|---|
| PRATEsfc | 0.187 → 0.181@80% → 0.183 | 0.234 → 0.228@56% → 0.232 | 0.347 → 0.338@86% → 0.348 |
| PRMSL | 0.0775 → 0.0772@51% → 0.0774 | 0.0528 → 0.0525@74% → 0.0527 | 0.0181 → 0.018@9% → 0.0182 |
| eastward_wind_at_ten_meters | 0.158 → 0.158@66% → 0.158 | 0.314 → 0.313@21% → 0.314 | 0.139 → 0.139@90% → 0.139 |
| northward_wind_at_ten_meters | 0.151 → 0.15@37% → 0.151 | 0.234 → 0.233@37% → 0.234 | 0.0489 → 0.0483@38% → 0.0494 |

Worst spectrum: **PRATEsfc** · PSD figures: `val/power_spectrum/<VAR>` in wandb.

## Verdict

- **Outcome vs baseline:** ➖ **f-distill-only, as designed.** The GAN never engaged
  (`not-engaged` above: gen≈ln2, disc≈2·ln2 the whole run) — a coarse critic at
  σ∈[200,2000] (near-pure-noise) has no coherent structure to grade, exactly the
  design-note prediction. Crucially **no GAN collapse** (contrast every Lo run).
  `f_distill_loss` did the real work (1.40 → ~0.72 by ~9k, flat after). PRMSL coarse
  spectra are excellent (`spec_mae_hi` 0.018, no collapse); precip is under-powered at
  high-k (`spec_mae_hi_PRATEsfc` 0.35, the familiar too-smooth deficit).
- **Recommended checkpoint:** mid/late by the `f_distill_loss` plateau (~step 10k+),
  **not** by val CRPS. The `hi_cascade` validation is nearly insensitive to Hi's
  weights (val metrics flat to 4 sig figs across 18k steps) — at the σ=200 handoff the
  `200·ε` term washes out which x0 Hi produced — so `best_student.ckpt` selection is
  ~arbitrary here. The bundle used `best_student_tail.ckpt`.
- **Next action:** the discriminating test is the assembled bundle, not `hi_cascade`
  — see the eval comparison
  [report](2026-07-08-moe-eval-distilled-vs-teacher.md). Don't tune the Hi GAN
  (tap/weight/R1): the coarse critic is structurally blind at high σ.

## Caveats

- Derived from the frozen
  [`MOE_DISTILLATION_STATUS.md`](../../MOE_DISTILLATION_STATUS.md) (2026-07-07
  Student-Hi result); numbers above regenerated from wandb.
- ⚠️ _Prepend here if a later run invalidates this one._
