<!--
Per-run report template for a distilled-student training run.

`check_runs.py --report <wandb_id> [--beaker <ULID>]` produces a PRE-FILLED copy
of this structure under experiments/reports/. The generator fills everything
EXCEPT the **Verdict** section, which the human writes. Keep the section order
stable so reports diff cleanly against each other.
-->
# Run report — `<experiment-name>`

_One-line hypothesis: what this run is testing and against which baseline._

## Artifacts

| | |
|---|---|
| Experiment name | `ace-downscaling-distillation-...` |
| wandb run | `<id>` — https://wandb.ai/<entity>/<project>/runs/<id> |
| Beaker experiment | `<ULID>` — https://beaker.org/ex/<ULID> |
| Commit | `<short-sha>` — https://github.com/ai2cm/ace/commit/<full-sha> |
| State / last step | `running|finished|crashed` @ `<step>` |

## Config

- **Method / teacher:** fdistill | dmd2 | scm — single-model | MoE teacher
- **Launch command:**
  ```
  conda run -n fme bash configs/experiments/2026-05-18-distillation-with-val/run.sh <method> --suffix <s> ...
  ```
- **Key knobs:** gan_weight, gan_r1, spectral_weight/band_gamma/min_wavenumber,
  student_steps, lr_decay_steps, disc_feature_depth, expert (as applicable).

## 1 · Training behavior

- **GAN health:** `{healthy | disc-winning collapse | not-engaged}` — from
  `train/gan_loss_disc` / `train/gan_loss_gen` trajectories (not-engaged ≈ stuck at
  `ln2≈0.693` / `2·ln2≈1.386`; disc-winning ≈ disc→0 while gen↑).
- **Loss domination:** last-value table of the generator loss terms + ratios.

  | term | last | note |
  |---|---|---|
  | `train/f_distill_loss` | | reference term |
  | `train/spectral_loss_weighted` | | ratio vs f_distill |
  | `train/gan_loss_gen` (×weight) | | ratio vs f_distill |
  | `train/fake_score_loss` | | |
  | `train/total_loss` | | |

- **Overall trajectory:** `val/crps_mean` and `val/spec_mae_mean`
  `first → best@frac → last` — improving / flat / degrading (+ late-drift note).

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under-produces.

| variable | tail_99.99 (first→best→last) | tail_99.9999 (first→best→last) |
|---|---|---|
| ... | | |

Worst variable / band: _..._

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better). `best@frac` exposes the
checkpoint-selection trap — note if the minimum is mid-training.

| variable | lo (first→best→last) | mid | hi |
|---|---|---|---|
| ... | | | |

Problematic variable/band: _..._ · PSD figures: `val/power_spectrum/<VAR>` in wandb.

## Verdict  <!-- HUMAN: fill this in -->

- **Outcome vs baseline:** win / flat / degrade — _why_.
- **Recommended checkpoint:** step / fraction (mid-training if late drift).
- **Next action:** _..._

## Caveats

- ⚠️ _Prepend a caveat here (and link forward) if a later run invalidates this one._
