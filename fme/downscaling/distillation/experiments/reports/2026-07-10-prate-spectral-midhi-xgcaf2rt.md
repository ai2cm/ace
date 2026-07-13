# Run report — `ace-downscaling-distillation-fdistill-with-val-prate-spectral-midhi`

_Hypothesis: the spectral loss currently weights every wavenumber flat
(`band_gamma=0`, `min_wavenumber=0`), so a third of its budget lands on the **low**
band that f-distill / distribution-matching already reproduces well. The **mid** band
(the coarse→fine transition, where a bicubic-interpolation artifact from the 100 km
inputs is suspected) carries the largest `spec_mae`. Redirect the low-third budget
onto mid+high by excluding wavenumbers below `nw//3 = 85` (`min_wavenumber=85`),
keeping `band_gamma=0` so **mid and high are weighted equally**. Expectation: mid-band
`spec_mae` drops without giving back high-k texture, vs the flat-weighting win
`i26sidsm`._

## Artifacts

| | |
|---|---|
| Experiment name | `ace-downscaling-distillation-fdistill-with-val-prate-spectral-midhi` |
| wandb run | `xgcaf2rt` — https://wandb.ai/ai2cm/fastgen/runs/xgcaf2rt |
| Beaker experiment | `01KX6T1BM73VETZF53TWBHSEFE` — https://beaker.org/ex/01KX6T1BM73VETZF53TWBHSEFE |
| Commit | `e7679c0` — https://github.com/ai2cm/ace/commit/e7679c0a9583bc42ee07d7eacf8e8db619c120d0 |
| State / last step | `canceled` @ ~`52650` (stopped 2026-07-13; useful optimum was ~2.6k) |

## Config

- **Method / teacher:** fdistill — default single-model teacher (`best_histogram_tail.ckpt`)
- **Launch command:**
  ```
  conda run -n fme bash configs/experiments/2026-05-18-distillation-with-val/run.sh \
      fdistill --suffix prate-spectral-midhi \
      --spectral-weight 1e-2 --spectral-min-wavenumber 85
  ```
- **Key knobs:** `spectral_weight=1e-2`, `band_gamma=0` (flat over included band),
  **`min_wavenumber=85`** (= `nw//3`; drops the eval "lo" third), `gan_weight=1e-3`
  (default). Only `min_wavenumber` changes vs the baseline.
- **Baseline for comparison:** `i26sidsm` — identical config except
  `min_wavenumber=0` (flat over all 257 wavenumbers, low included).
- **Band geometry:** student fine output is 512 wide → `nw = 512//2+1 = 257`
  wavenumbers. Eval thirds: lo `[0,85)` · mid `[85,170)` · hi `[170,257)`.

## 1 · Training behavior

- **GAN health:** `healthy` — gen 1.2, disc 1.18 (both engaged; no collapse).
- **Loss domination:**

  | term | last | note |
  |---|---|---|
  | `train/f_distill_loss` | 0.07346 |  |
  | `train/spectral_loss_weighted` | 0.001409 | 0.019× f_distill (smaller than baseline's 0.029× — fewer wavenumbers in the loss) |
  | `train/gan_loss_gen` | 1.204 | ×weight 0.001 = 0.0012; 0.016× f_distill |
  | `train/fake_score_loss` | 0.02411 |  |
  | `train/total_loss` | 0.9782 |  |

  _f_distill_loss dominates the generator objective (expected)._

- **`val/crps_mean`:** flat — improved 9% first→best, then drifted +13% best→last.
- **`val/spec_mae_mean`:** degrading late (checkpoint-selection trap) — improved 98%
  first→best, then drifted +691% best→last. Same late-drift signature as `i26sidsm`.

## 2 · Tail behavior

Ratio to teacher; ~1.0 ideal, >1 over-produces extremes, <1 under.

| variable | tail_99.99 | tail_99.9999 |
|---|---|---|
| PRATEsfc | 0.816 → 1.01@4% → 2.19 | 0.77 → 0.977@3% → 2.19 |

Worst tail: **PRATEsfc**.

## 3 · Power spectrum

`spec_mae` per band (relative error; lower better).

| variable | lo | mid | hi |
|---|---|---|---|
| PRATEsfc | 1.17 → 0.0206@4% → 0.408 | 1.14 → 0.0103@4% → 0.0785 | 0.501 → 0.00366@9% → 0.0497 |

Worst spectrum: **PRATEsfc** · PSD figures: `val/power_spectrum/<VAR>` in wandb.

### ★ Comparison at the SELECTED checkpoints vs baseline `i26sidsm`

The right comparison is at the checkpoints the callback actually saves —
`best_student.ckpt` = argmin `val/crps_mean`; `best_student_tail.ckpt` = argmin
`|val/tail_99.9999_PRATEsfc − 1|` (top-percentile `_tail_deviation_score`,
single-var here). `spec_mae` per band (lo `[0,85)` / mid `[85,170)` / hi
`[170,257)`) at those steps:

| checkpoint | run | step (frac) | spec lo/mid/hi | spec_mean | crps | tail6 |
|---|---|---|---|---|---|---|
| `best_student.ckpt` (CRPS-min) | **midhi** | 2730 (5%) | 0.029 / 0.066 / 0.091 | **0.062** | 0.1047 | 1.06 |
| `best_student.ckpt` (CRPS-min) | base | 1170 (4%) | 0.43 / 0.58 / 0.38 | 0.47 | 0.1045 | 0.94 |
| `best_student_tail.ckpt` (tail-min) | midhi | 1690 (3%) | 0.29 / 0.37 / 0.24 | 0.30 | 0.1054 | 0.98 |
| `best_student_tail.ckpt` (tail-min) | **base** | 7930 (29%) | 0.060 / 0.12 / 0.14 | **0.11** | 0.1050 | 1.00 |

**★ The selectors are decoupled from spectral quality, so the "winner" flips by
which selector you read — neither flip is a real effect of the loss change.**
`val/crps_mean` is flat to ~1% across the whole useful region (midhi min 0.1047 vs
base 0.1045), so its argmin step is essentially noise-determined, while `spec_mae`
swings 5–10× across those steps. Hence at CRPS-min midhi looks 7× better (its CRPS
bottomed at a spectrally-converged step) and at tail-min base looks 3× better (its
tail-min landed mid-run @29%, midhi's landed un-converged @3%).

The fair, noise- and step-controlled quantity is each run's **best *sustained*
spectrum** (rolling-median `spec_mae`, ~650-step window) — which lands at nearly the
same frac for both runs:

| | step (frac) | spec mid / hi / mean |
|---|---|---|
| **midhi** | 2600 (5%) | 0.023 / 0.065 / **0.044** |
| base | 2340 (8%) | 0.038 / 0.074 / **0.043** |

**Essentially tied** (mean 0.044 vs 0.043), midhi marginally better on mid+hi.

## Verdict

- **Outcome vs baseline `i26sidsm`:** ➖ **roughly neutral / inconclusive** (NOT the
  "degrade" originally logged — see Caveats). At each run's best-sustained spectrum
  the two are tied (`spec_mae_mean` 0.044 vs 0.043), with midhi marginally lower on
  mid (0.023 vs 0.038) and hi (0.065 vs 0.074) — a *weak* signal in the hypothesis's
  favor, well within noise. There is no clear win and no clear loss from dropping the
  low third. `crps` and tails are unchanged. Not worth the lost low-band coverage on
  this evidence; **flat all-band weighting (`i26sidsm`) stays the default** as the
  simpler config, but the band-cut is not harmful.
- **The louder finding is about checkpoint selection, not the loss.** Neither
  `best_student.ckpt` (CRPS) nor `best_student_tail.ckpt` (tail) reliably lands on the
  spectral optimum: CRPS is flat to ~1% so its argmin is noise, and tail-min landed at
  3% (midhi) vs 29% (base). Each run *can* deliver a ~0.06-`spec_mean` checkpoint, but
  via a *different* saved file (midhi ← CRPS ckpt; base ← tail ckpt). Any
  checkpoint-to-checkpoint spectral comparison between runs is dominated by which
  selector happened to coincide with the spectral optimum.
- **Recommended checkpoint:** for this run, `best_student.ckpt` @2730 (`spec_mean`
  0.062) — it happens to sit at midhi's spectral optimum. Its `best_student_tail.ckpt`
  @1690 is spectrally poor (0.30) and should **not** be used.
- **Next action:** (1) Job **canceled** (2026-07-13, by user). (2) Do not pursue
  `min_wavenumber` band-cutting further — neutral, not worth the added knob. (3) A
  genuine mid *bump* (non-monotonic band weight on top of the full band) remains the
  open idea for the mid band; needs a `SpectralMatchingLoss` change. (4) **★ Add a
  spectral-aware early-stopping / checkpoint-selection criterion** — see the new
  planned item in `LOG.md`. The current CRPS/tail selectors miss the spectral optimum
  *and* let runs drift for tens of thousands of wasted steps (this run ran to 52k; its
  useful optimum was ~2.6k).

## Caveats

- ⚠️ _Prepend here if a later run invalidates this one._
- **⚠️ Correction (2026-07-13):** an earlier version of this report logged a
  **❌ degrade** verdict based on a 25th-percentile-over-fixed-window `spec_mae`
  comparison. That was an artifact: `xgcaf2rt` ran to 52k and drifts late, so its p25
  over a fixed 15k window was inflated by more post-peak drift than the baseline's
  shorter (27.8k) run contained — it compared drift exposure, not the achievable
  optimum. Re-doing the comparison **at the selected checkpoints** and at the
  **best-sustained spectrum** (both step-controlled) shows the two are tied. Verdict
  corrected to ➖ neutral.
- The per-step validation `spec_mae` is very noisy (single-snapshot spikes of 5–10×);
  all spectral comparisons here use either the saved-checkpoint step or a rolling
  median, never a raw single-point min.
