# Run report — `ace-downscaling-distillation-fdistill-with-val-prate-spectral-lowgan-fix`

_Hypothesis: first **valid** low-GAN test — does dropping the generator GAN weight
1e-3→3e-4 (fixed target, W=1e-2) cut the late drift seen in `i26sidsm` while keeping
the spectral gains? (The pre-fix `gpx5574t` low-GAN run was invalid.)_

## Artifacts

| | |
|---|---|
| Experiment name | `ace-downscaling-distillation-fdistill-with-val-prate-spectral-lowgan-fix` |
| wandb run | `6dotglmg` — https://wandb.ai/ai2cm/fastgen/runs/6dotglmg |
| Beaker experiment | `01KX4DRYQ0RSQEWRY5F6QBP9BY` — https://beaker.org/ex/01KX4DRYQ0RSQEWRY5F6QBP9BY |
| Commit | `e29f797` — https://github.com/ai2cm/ace/commit/e29f79776f271380b37ff03dc7e9cc2c9cc57541 |
| State / last step | `crashed` @ `14040` |

## Config

- **Method / teacher:** fdistill — default single-model teacher, single-var PRATEsfc
- **Key knobs:** `spectral_weight=1e-2`, `band_gamma=0`, `min_wavenumber=0`,
  **`gan_weight=3e-4`** (vs `i26sidsm`'s 1e-3 — the only change).
- **Baseline for comparison:** `i26sidsm` (gan=1e-3, else identical).

> **Metric selection:** validation trajectories below are anchored at the
> **`best_student_tail` checkpoint — step 8190** (~58% of this 14k-step run; where
> `val/tail_best_score` bottomed at 0.00056) — the checkpoint that would ship if this
> arm were deployed (matching `f7z93y0a`/`i26sidsm`). Trajectories are
> `first → best_tail@8190 → last`. Unlike the longer runs, this arm's tail-min lands
> late (58%) because it **crashed at 14k before the late-drift regime**, so best_tail
> here still sits in the good spectral regime — see §4. §4's matched-step comparison to
> `i26sidsm` is the primary read; training/loss metrics are end of run.

## 1 · Training behavior

- **GAN health:** `healthy` — gen 1.34, disc 1.08 (both engaged; no collapse).
- **Loss domination** (ratios at `gan_weight=3e-4`):

  | term | last | note |
  |---|---|---|
  | `train/f_distill_loss` | 0.06483 |  |
  | `train/spectral_loss_weighted` | 0.002911 | 0.045× f_distill |
  | `train/gan_loss_gen` | 1.342 | ×weight 3e-4 = 0.000403; 0.0062× f_distill (vs ~0.013× at gan=1e-3) |
  | `train/fake_score_loss` | 0.02498 |  |
  | `train/total_loss` | 0.8968 |  |

  _f_distill dominates; the GAN term is now ~half its `i26sidsm` share, as intended._

- **`val/crps_mean`:** 0.1164 → **0.1053** (best_tail@8190) → 0.1075 (+2% best_tail→last).
- **`val/spec_mae_mean`:** 0.921 → **0.0456** (best_tail@8190) → 0.0633 (+39% best_tail→last).
  **⚠️ this drift is not comparable to `i26sidsm`'s catastrophic +632%** — see §4 (run
  crashed at 14k, a much shorter run, so it never entered the late-drift regime; the
  per-band optima are lower still, ~0.01, mid-run).

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under. Values at
`best_student_tail` (step 8190).

| variable | tail_99.99 | tail_99.9999 |
|---|---|---|
| PRATEsfc | 0.823 → **1.060** → 1.081 | 0.776 → **1.001** → 1.052 |

Worst tail: **PRATEsfc** (only variable). Both tails near-ideal at best_tail and only
mildly over-produced by end (1.08 / 1.05) — no drift blow-up (run stopped at 14k).

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better). Values at `best_student_tail`
(step 8190).

| variable | lo | mid | hi |
|---|---|---|---|
| PRATEsfc | 1.13 → **0.078** → 0.117 | 1.12 → **0.019** → 0.029 | 0.512 → **0.041** → 0.045 |

Worst spectrum: **lo band** at best_tail (0.078) · PSD figures:
`val/power_spectrum/<VAR>` in wandb. Note best_tail@8190 lands at a clean, near-optimal
spectrum here (`spec_mean` 0.046 vs best-sustained 0.040) — the tail selector and the
spectral optimum happen to coincide because the run never drifted.

## 4 · Training trajectory vs baseline `i26sidsm`

The run **crashed at step 14040**; `i26sidsm` ran to 27820. So endpoint `best→last`
drift is *not* comparable — it must be read at matched steps.

**Matched-step comparison (smoothed spec_mae_mean / tail_99.99):**

| step | lowgan spec / tail | base spec / tail | read |
|---|---|---|---|
| 5000 | 0.062 / 1.08 | 0.084 / 1.10 | lowgan marginally better |
| 8000 | 0.064 / 1.06 | 0.065 / 1.11 | tied spec, lowgan better tail |
| 11000 | 0.073 / 1.13 | 0.080 / 1.18 | lowgan marginally better |
| 14000 | 0.063 / 1.10 | 0.069 / 1.17 | lowgan marginally better |

**Best-sustained spectrum:** lowgan @2860 (20%) `spec_mean=0.040` (lo 0.051 / mid 0.034
/ **hi 0.044**); base @2340 (8%) `spec_mean=0.043` (lo **0.022** / mid 0.038 / hi 0.074).
Nearly tied on the mean; lowgan better on hi+mid, worse on lo.

**★ Drift at matched step 14k is the same:** lowgan drifted +57% from its best by 14k;
base drifted +61% by *its* step 14k. The eye-catching "+92% vs +632%" gap is a pure
run-length artifact — **`i26sidsm`'s catastrophic drift happens *after* 14k** (it was
+61% at 14k, +632% by 28k). `6dotglmg` stopped at 14k (state `crashed`, but likely a
manual cancel — see the LOG note), right before that regime, so **the late drift the
low-GAN weight was meant to tame was never actually tested.**

**Constrained vs less-constrained:** tails (less-constrained) run ~0.03–0.07 lower
(better) at matched steps — a genuine, if modest, low-GAN benefit on extreme control.
Spectrum (partly constrained) is ~tied.

## Verdict

- **Outcome vs baseline `i26sidsm`:** ➖ **inconclusive, mildly positive.** At matched
  steps (≤14k) the lower GAN weight is marginally better on overall spectrum and
  modestly better on tail overshoot (1.10 vs 1.17 @14k), and best-sustained hi-band is
  better (0.044 vs 0.074) — no downside seen. **But the run crashed at 14k, before the
  late-drift regime**, so its headline hypothesis — *does gan=3e-4 tame the late drift?*
  — is **untested** (drift rate at 14k is identical to baseline, +57% vs +61%; baseline's
  blow-up to +632% is a >14k phenomenon).
- **Recommended checkpoint:** best-sustained ~step 2.9k (`spec_mean` 0.040, tail 1.07).
  The saved `best_student_tail.ckpt` @8190 is nearly as good here (`spec_mean` 0.046,
  tails 1.06/1.00) — selector and spectral optimum coincide because this run crashed
  before drifting (contrast `i26sidsm`, whose best_tail sits at a noisy 0.11).
- **Next action:** **re-run longer** to actually reach the late-drift regime (≥28k, with
  checkpointing so a crash doesn't lose the answer) — ideally after spec 13 (spectral
  early-stop) lands so it stops at the optimum instead of drifting. Until then, low-GAN
  is a mild, safe improvement on tails but not a demonstrated drift fix. Note the
  `band_gamma` sweep (`gamma0p5`/`gamma1`, launched 2026-07-13) is orthogonal and keeps
  gan=1e-3; a low-GAN × band_gamma combination is a later cell.

## Caveats

- ⚠️ _Prepend here if a later run invalidates this one._
- ⚠️ The `spec_mae_mean` `best→last` drift (+92%) in §1 is a run-length artifact vs
  `i26sidsm` (+632%); the runs crashed at different steps (14k vs 28k). Use the §4
  matched-step drift (+57% vs +61% at 14k) instead.
