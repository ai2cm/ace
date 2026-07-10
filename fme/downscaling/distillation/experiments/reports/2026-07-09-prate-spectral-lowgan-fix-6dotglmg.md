# Run report — `ace-downscaling-distillation-fdistill-with-val-prate-spectral-lowgan-fix`

_Hypothesis: first **valid** low-GAN test — does dropping the generator GAN weight
1e-3→3e-4 (fixed target, W=1e-2) cut the late drift seen in `i26sidsm` while keeping
the spectral gains? (The pre-fix `gpx5574t` low-GAN run was invalid.)_

> ⏳ Running — this is an early snapshot; regenerate this report when it has more
> history, then write the verdict.

## Artifacts

| | |
|---|---|
| Experiment name | `ace-downscaling-distillation-fdistill-with-val-prate-spectral-lowgan-fix` |
| wandb run | `6dotglmg` — https://wandb.ai/ai2cm/fastgen/runs/6dotglmg |
| Beaker experiment | `01KX4DRYQ0RSQEWRY5F6QBP9BY` — https://beaker.org/ex/01KX4DRYQ0RSQEWRY5F6QBP9BY |
| Commit | `e29f797` — https://github.com/ai2cm/ace/commit/e29f797 |
| State / last step | `running` @ `650` |

## Config

- **Method:** fdistill
- **Spectral weight (derived):** 0.01
- **GAN generator weight (assumed for ratios):** 0.0003
- **Suffix:** `prate-spectral-lowgan-fix`

## 1 · Training behavior

- **GAN health:** `healthy` — gen 1.43, disc 0.996 (both engaged)
- **Loss domination:**

  | term | last | note |
  |---|---|---|
  | `train/f_distill_loss` | 0.08855 |  |
  | `train/spectral_loss_weighted` | 0.005583 | 0.063× f_distill |
  | `train/gan_loss_gen` | 1.432 | ×weight 0.0003 = 0.00043; 0.0049× f_distill |
  | `train/fake_score_loss` | 0.01499 |  |
  | `train/total_loss` | 0.828 |  |

  _f_distill_loss dominates the generator objective (expected)._

- **`val/crps_mean`:** flat: improved 7% first→best, then drifted +0% best→last
- **`val/spec_mae_mean`:** improving: improved 14% first→best, then drifted +0% best→last

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under.

| variable | tail_99.99 | tail_99.9999 |
|---|---|---|
| PRATEsfc | 0.823 → 0.87@75% → 0.848 | 0.776 → 0.882@75% → 0.848 |

Worst tail: **PRATEsfc**.

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better).

| variable | lo | mid | hi |
|---|---|---|---|
| PRATEsfc | 1.13 → 0.772@100% → 0.772 | 1.12 → 0.979@75% → 0.997 | 0.512 → 0.512@0% → 0.608 |

Worst spectrum: **PRATEsfc** · PSD figures: `val/power_spectrum/<VAR>` in wandb.

## Verdict  <!-- HUMAN: fill this in -->

- **Outcome vs baseline:** TODO
- **Recommended checkpoint:** TODO (mid-training if late drift)
- **Next action:** TODO

## Caveats

- ⚠️ _Prepend here if a later run invalidates this one._
