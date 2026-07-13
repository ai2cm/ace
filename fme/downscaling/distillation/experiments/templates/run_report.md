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

## 4 · Training trajectory vs baseline  <!-- HUMAN: fill this in -->

How the knob changed the *trajectory*, not just the endpoint. Compare vs the baseline
run at **step-controlled** quantities (matched step, or best-sustained rolling-median —
never a raw single-point min; per-step val metrics spike 5–10×).

- **Convergence speed:** step (and % of run) at which each run first reaches within
  ~10% of its own best-sustained `spec_mae_mean` / `crps_mean`. Faster to target = the
  knob helps optimization; slower = it fights it. (Note if a metric like `crps_mean`
  converges in the first validation — then it carries no convergence signal and no
  stopping signal.)
- **Constrained vs less-constrained metrics.** Split validation metrics by whether the
  training loss *directly* constrains them, and ask whether the change improved the
  **less-constrained** ones — the stronger generalization signal (improving a metric you
  didn't optimize beats improving the one you did):
  - _Constrained:_ whatever the loss targets — e.g. the spectral bands inside
    `[min_wavenumber, nw)`, distribution match (CRPS-adjacent), GAN realism.
  - _Less-constrained (held-out-ish):_ tail extremes (never in the loss), and spectral
    bands **excluded** from the loss (e.g. lo `[0,min_wavenumber)` when `min_wavenumber>0`).
  - State explicitly which bands the knob moved between the two categories, and whether
    the endpoint gain (if any) came from genuinely better performance or just from
    re-labeling a band as unconstrained.

## Verdict  <!-- HUMAN: fill this in -->

- **Outcome vs baseline:** win / flat / degrade — _why_.
- **Recommended checkpoint:** step / fraction (mid-training if late drift).
- **Next action:** _..._

## Caveats

- ⚠️ _Prepend a caveat here (and link forward) if a later run invalidates this one._
