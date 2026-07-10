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

## 1 · Training behavior

- **GAN health:** `unknown` — gan losses not logged
- **Loss domination:**

  | term | last | note |
  |---|---|---|

  _f_distill_loss dominates the generator objective (expected)._

- **`val/crps_mean`:** —
- **`val/spec_mae_mean`:** —

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under.

| variable | tail_99.99 | tail_99.9999 |
|---|---|---|
| PRATEsfc | — | — |

Worst tail: **—**.

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better).

| variable | lo | mid | hi |
|---|---|---|---|
| PRATEsfc | — | — | — |

Worst spectrum: **—** · PSD figures: `val/power_spectrum/<VAR>` in wandb.

## Verdict  <!-- HUMAN: fill this in -->

- **Outcome vs baseline:** n/a — this *is* the baseline. Characteristics to beat:
  poor high-k spectra (`spec_mae_hi` ~0.87 last) and over-produced extremes
  (`tail_99.99` ~3.16). No spectral constraint; GAN carries all small-scale energy.
- **Recommended checkpoint:** n/a (reference).
- **Next action:** compare all spectral arms against this run.

## Caveats

- ⚠️ _Prepend here if a later run invalidates this one._
