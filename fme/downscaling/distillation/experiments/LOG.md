# Distillation experiment log

Central planning + outcomes log for distilled downscaling students. Process:
[`WORKFLOW.md`](WORKFLOW.md). Per-run reports: [`reports/`](reports/).

> Pre-2026-07 MoE-distillation history is frozen in
> [`../MOE_DISTILLATION_STATUS.md`](../MOE_DISTILLATION_STATUS.md).

---

## ⚡ At a glance  <!-- keep this current: the daily check-in view -->

_Last updated: 2026-07-29._

### 🔴 In flight — check for updates, finish write-ups when done

- **f-distill step-count sweep** (launched 2026-07-23) — native **1-step**
  (`01KY8F8DJG89CNVV89257V0B72`) + native **4-step** (`01KY8F8M3BD7NG1QQDNQJJ8KVW`),
  spectral W=1e-2, **early-stop patience=10** (spec-13, first sweep to use it). Baseline =
  2-step `i26sidsm`. Finds the quality-vs-NFE knee. First runs launched after plumbing the
  early-stop flags through the launcher ([`1440599`](https://github.com/ai2cm/ace/commit/144059904)).
  ([write-up](reports/2026-07-13-fdistill-step-count-sweep-TBD.md))

_Recently closed:_
- **Single-GPU CONUS eval** `6eff6ig5` (hirov1) + `y543b0gf` (spectral) → ✅ **the multi-rank
  tail histogram was measuring winter only; ground-truth extremes understated 8–21%**
  (2026-07-29). `ContiguousDistributedSampler` gives rank 0 the *first* quarter of the record
  (Jan–early Apr), so it is a seasonal bias, not a smaller sample. CRPS/PSD verdicts survive
  (≤0.3%), but hirov1's extreme tail goes 0.936→0.974 and the student's @99.99
  over-production is now clearly its weakest metric (+9.5%). Cost 4.02× wall clock →
  **fix the reduction** rather than repeat this.
  ([report](reports/2026-07-28-hirov1-vs-spectral-conus-1gpu.md))
- `2yhjonz9` (band_gamma=0.5) + `34rg7wii` (band_gamma=1) → ➕ **mild positive; monotonic
  response curve** (2026-07-14). The hi-k tilt works as designed — best-sustained hi
  `spec_mae` improves 0.074→0.066→0.050 and overall mean 0.043→0.038→0.035 across
  γ=0/0.5/1 — at a monotonic lo cost (0.022→0.024→0.037). Small gains; γ=1 best on mean.
  ([γ0.5](reports/2026-07-13-prate-spectral-gamma0p5-2yhjonz9.md) ·
  [γ1](reports/2026-07-13-prate-spectral-gamma1-34rg7wii.md)).
- `p337gcg9` Lo-only ablation → ✅ **Hi is needed — for extreme precip only** (2026-07-13,
  [report](reports/2026-07-13-lo-only-from-noise200-ablation-p337gcg9.md)).

### 🟢 Next up — likely-good experiments (queued, not launched)

0. **★ Two measured f-distill defects the current selectors are blind to** (raised
   2026-07-30, see the ★ RESEARCH TASK entries below). Both are ~8–30× the teacher's error
   and both sit where the spectral loss tuning actively removed weight:
   **(a)** 200–400 mm/day precip density **+25% to +52%** (teacher ~2%), invisible because
   `tail@99.9999` reads 0.995; **(b)** zonal PSD **+48% at k≈95 (~70 km)**, in the **lo**
   third that `min_wavenumber=85` zeroed and `band_gamma=1` down-weighted ~12×.
   Cheapest next probes: `--disc-feature-depth 1|2` (current disc cell ≈180 km vs a 70 km
   defect) and a DMD2-vs-f-distill histogram comparison.
1. **Native step-count sweep** — 1-step (task #3) + 4-step (task #2) f-distill vs the
   2-step `i26sidsm`; find the quality-vs-NFE knee.
   ([write-up](reports/2026-07-13-fdistill-step-count-sweep-TBD.md))
2. **Spectral-aware early stop + `best_student_spec` selector** — spec 13 / task #1;
   also fixes the checkpoint-selection trap that confounds every comparison.
3. **Port the tuned spectral loss → multi-variable MoE** with per-var `variable_weights`
   (up-weight the worst-spectrum vars; winds/PRMSL were the MoE weak spots).
4. **Multi-scale (multi-head) discriminator** — spec 12 flagship E2; the big untested GAN
   texture lever for the winds hi-k gap single taps never closed.
5. **Non-monotonic band *bump*** (SpectralMatchingLoss code change) — motivated by the
   closed `band_gamma` sweep: monotonic tilt improves hi but makes **lo** the limiting
   band at γ=1, so the better lever is a bump that lifts a target band **without**
   down-weighting the rest (rather than pushing γ higher toward the neutral `xgcaf2rt`
   regime). **Aim it at lo, not hi:** the measured defect is at k≈21 of 257 (~70 km), and
   γ>0 down-weights exactly there — see the ★ RESEARCH TASK on the k≈95 bump.
6. **Longer reduce-GAN re-run** (`gan=3e-4`) — `6dotglmg` stopped at 14k, before the
   late-drift regime it was meant to test; re-run with checkpointing.

---

## Run registry

Every launched run gets a row. `verdict`: ✅ win · ➕ mild positive · ➖ flat · ❌ degrade
· ⏳ running · ⚠️ invalid. Regenerate a row with `check_runs.py --registry-row <id>`.

> **Note on `state`:** a wandb/beaker state of **`crashed` usually means the run was
> *manually cancelled***, not that it hit a genuine error — these are experiment arms
> we stop once we've seen enough (or that get preempted). Don't over-interpret
> "crashed" as a failure; check the last step and metrics. A real error shows a
> traceback in the beaker logs.

| wandb | date | experiment name | beaker | commit | method / knobs | state | verdict | report |
|---|---|---|---|---|---|---|---|---|
| `xklvoz0n` | 2026-07-23 | …-prate-1step | `01KY8F8DJG89CNVV89257V0B72` | [`26a887f`](https://github.com/ai2cm/ace/commit/26a887f9c) | fdistill, spectral W=1e-2, **student-steps=1**, early-stop patience=10 | running | ⏳ running | [report](reports/2026-07-13-fdistill-step-count-sweep-TBD.md) |
| `850hcj6i` | 2026-07-23 | …-prate-4step | `01KY8F8M3BD7NG1QQDNQJJ8KVW` | [`26a887f`](https://github.com/ai2cm/ace/commit/26a887f9c) | fdistill, spectral W=1e-2, **student-steps=4**, early-stop patience=10 | running | ⏳ running | [report](reports/2026-07-13-fdistill-step-count-sweep-TBD.md) |
| `f7z93y0a` | 2026-07-07 | …-prate-baseline | `01KWX5CVJQ2BP53VH95WKPVPED` | [`26868ca`](https://github.com/ai2cm/ace/commit/26868ca) | fdistill, no spectral (reference) | crashed@29510 | ➖ baseline | [report](reports/2026-07-07-prate-baseline-f7z93y0a.md) |
| `i26sidsm` | 2026-07-08 | …-prate-spectral-fix | `01KX00N9SE3ZVQFHQJ54XS0TAP` | [`e29f797`](https://github.com/ai2cm/ace/commit/e29f797) | fdistill, spectral W=1e-2, gan=1e-3 | crashed@27820 | ✅ win (mid-train ckpt) | [report](reports/2026-07-08-prate-spectral-fix-i26sidsm.md) |
| `6dotglmg` | 2026-07-09 | …-prate-spectral-lowgan-fix | `01KX4DRYQ0RSQEWRY5F6QBP9BY` | [`e29f797`](https://github.com/ai2cm/ace/commit/e29f797) | fdistill, spectral W=1e-2, **gan=3e-4** | crashed@14040 | ➖ inconclusive (mild tail gain; crashed before late-drift regime) | [report](reports/2026-07-09-prate-spectral-lowgan-fix-6dotglmg.md) |
| `xgcaf2rt` | 2026-07-10 | …-prate-spectral-midhi | `01KX6T1BM73VETZF53TWBHSEFE` | [`e7679c0`](https://github.com/ai2cm/ace/commit/e7679c0a9583bc42ee07d7eacf8e8db619c120d0) | fdistill, spectral W=1e-2, **min_wavenumber=85** (drop lo third, flat mid+hi) | canceled@52k | ➖ neutral (tied at best-sustained spectrum; `best_student.ckpt`@2730) | [report](reports/2026-07-10-prate-spectral-midhi-xgcaf2rt.md) |
| `2yhjonz9` | 2026-07-13 | …-prate-spectral-gamma0p5 | `01KXEN0NJ81G7R1SF1F4ZFZV2R` | [`06aee7f`](https://github.com/ai2cm/ace/commit/06aee7f9c) | fdistill, spectral W=1e-2, **band_gamma=0.5** (gentle hi tilt; lo≈0.61× hi≈1.37×) | canceled@18850 | ➕ mild positive (best-sustained hi 0.066 / mean 0.038 vs flat 0.074 / 0.043; small lo cost) | [report](reports/2026-07-13-prate-spectral-gamma0p5-2yhjonz9.md) |
| `34rg7wii` | 2026-07-13 | …-prate-spectral-gamma1 | `01KXEN0PH05655AQD3FWJRSCXQ` | [`06aee7f`](https://github.com/ai2cm/ace/commit/06aee7f9c) | fdistill, spectral W=1e-2, **band_gamma=1** (linear hi tilt; lo≈0.33× hi≈1.7×) | canceled@17680 | ➕ mild positive (best-sustained hi 0.050 / mean 0.035 — best of sweep; lo cost +68%) | [report](reports/2026-07-13-prate-spectral-gamma1-34rg7wii.md) |
| `s4abc6ba` | 2026-07-07 | …-prate-spectral | — | [`ae3979b`](https://github.com/ai2cm/ace/commit/ae3979b) | fdistill, spectral W=1e-2 (**pre-fix target**) | stopped | ⚠️ invalid (wrong target) | — |
| `gpx5574t` | 2026-07-07 | …-prate-spectral-lowgan | `01KWYPADNHC7SK58FMA981XTQV` | [`ae3979b`](https://github.com/ai2cm/ace/commit/ae3979b) | fdistill, spectral+gan=3e-4 (**pre-fix target**) | crashed@3770 | ⚠️ invalid (wrong target) | — |

### Eval-bundle comparisons (project `andrep-downscaling`)

| distilled | teacher | date | region/period | commit | beaker (distilled / teacher) | verdict | report |
|---|---|---|---|---|---|---|---|
| `rmoodemk` | `1r1p6djp` | 2026-07-08 | CONUS 2023, 100km→3km X-SHiELD | [`de3e00c`](https://github.com/ai2cm/ace/commit/de3e00ce2bf8215114a818faae11700afd8005f9) | `01KWZD6YMZSD37XZHDMYB8RFC7` / `01KWZD6WFN4TCSMMC48BTFMN8Q` | see report | [report](reports/2026-07-08-moe-eval-distilled-vs-teacher.md) |
| `x2nyzmzh` (spectral) | `flzvb6tp` (baseline) | 2026-07-13 | CONUS, 100km→3km X-SHiELD AMIP control | [`d6cd8dd`](https://github.com/ai2cm/ace/commit/d6cd8dd261a45aaa999e58cc551c460ee68dc940) | — | ✅ spectral wins: PSD bias 0.46→0.13 (−71%), CRPS −3.5%, tails ~ideal | [report](reports/2026-07-13-prate-eval-baseline-vs-spectral.md) |
| `l6vv7yx0` (spectral) | `fg9byv9y` (baseline) | 2026-07-13 | maritime continent, 100km→3km X-SHiELD AMIP control | [`d6cd8dd`](https://github.com/ai2cm/ace/commit/d6cd8dd261a45aaa999e58cc551c460ee68dc940) | — | ✅ spectral wins: PSD bias 0.60→0.13 (−78%), CRPS −2.6%, tails closer to 1 | [report](reports/2026-07-13-prate-eval-baseline-vs-spectral.md) |
| `y543b0gf` (spectral `i26sidsm`, 2-step) | `6eff6ig5` (hirov1, full diffusion) | 2026-07-28 | CONUS, 100km→3km X-SHiELD AMIP — **single GPU** | [`8b9edba`](https://github.com/ai2cm/ace/commit/8b9edba3f) | `01KYP41DEZ7DND9YFY77XVYEH7` / `01KYP41B245ZXBNH3QGJEGR74N` | ✅ tails now trustworthy: ground-truth percentile was understated **8% @99.9999 / 21% @99.99**; ratios self-normalize so past verdicts hold, but hirov1's extreme tail 0.936→**0.974**. Student within 3.3% CRPS / 8.3% PSD of full diffusion, better @99.9999, +9.5% @99.99 | [report](reports/2026-07-28-hirov1-vs-spectral-conus-1gpu.md) |
| `p337gcg9` (Lo-only, 2 NFE) | `rmoodemk` (Hi→Lo bundle, 3 NFE) | 2026-07-13 | CONUS, 100km→3km X-SHiELD AMIP | [`af4d134`](https://github.com/ai2cm/ace/commit/af4d13415dacc38ab34e5ad8bbfa22a51615d611) | `01KXEYCC9HAZ7F1G85E3KRPKFD` | ✅ **Hi needed for extreme precip only**: Lo-only ≈ bundle on CRPS (<0.03%) + PSD (<1%) all 4 vars, but under-produces `tail_99.9999_PRATEsfc` 1.01→0.93 (wind tails unchanged) | [report](reports/2026-07-13-lo-only-from-noise200-ablation-p337gcg9.md) |

### MoE per-expert base models (bundled into `rmoodemk`)

The two per-expert students assembled into the distilled 2-step MoE bundle above.
Full lineage/diagnoses are frozen in
[`../MOE_DISTILLATION_STATUS.md`](../MOE_DISTILLATION_STATUS.md); these rows just
point at the standardized reports.

| wandb | date | expert / role | beaker | commit | verdict | report |
|---|---|---|---|---|---|---|
| `zct08386` | 2026-07-03 | expert 0 · Student-Lo (σ 0.005–200) | `01KWJAFKZ96YBR73F0TETBKC0Q` | [`184fa29`](https://github.com/ai2cm/ace/commit/184fa298b6dadad9ad40252d83e0d697b73d0c84) | ✅ clean (fine-scale carrier) | [report](reports/2026-07-03-baseline-fixed-moe-teacher-expert0-zct08386.md) |
| `4mez4kmn` | 2026-07-06 | expert 1 · Student-Hi (σ 200–2000) | `01KWTXGADFPB4GKVZ33C7ZGJP4` | [`e920ca7`](https://github.com/ai2cm/ace/commit/e920ca7f425be97fbbfbddae7a700b97ac04e536) | ➖ f-distill-only (GAN inert by design) | [report](reports/2026-07-06-hi-1step-moe-teacher-expert1-4mez4kmn.md) |

---

## Active / planned

- ~~**`6dotglmg` (reduce-GAN arm)**~~ — ➖ **done, inconclusive** (reported 2026-07-13):
  crashed@14k, mildly better tails at matched steps but **crashed before the late-drift
  regime** it was meant to test (drift at 14k identical to baseline). Needs a **longer
  re-run** (≥28k, with checkpointing) to actually test the drift hypothesis — ideally
  after spec 13 early-stop. See report.
- ~~**`band_gamma` sweep (launched 2026-07-13)**~~ — ➕ **done, mild positive; monotonic
  response curve** (canceled 2026-07-14, trained enough). `2yhjonz9` (gamma=0.5) +
  `34rg7wii` (gamma=1), f-distill PRATEsfc, W=1e-2 / gan=1e-3 / min_wavenumber=0, only
  `band_gamma` varies. At each run's best-sustained spectrum the tilt behaves exactly as
  designed and **monotonically**: `spec_mae_hi` improves 0.074→0.066→0.050 across
  γ=0/0.5/1, overall mean 0.043→0.038→0.035, at a monotonic lo cost 0.022→0.024→0.037.
  γ=1 is best on mean (−19% vs flat) and hi (−32%) but lo becomes the worst band (+68%).
  Because lo stays constrained (0.33×/0.61×, not zeroed like `xgcaf2rt`), it degrades
  gracefully and the net mean still improves — the opposite of the neutral hard cut.
  Gains are small; both runs show the checkpoint-selection trap (best_tail lands at
  8–10%, spectrally unconverged). See reports + outcomes bullet.
- ~~**`xgcaf2rt` (mid+high band arm)**~~ — ➖ **done, neutral** (checked & canceled
  2026-07-13): the `min_wavenumber=85` cut is tied with flat-band `i26sidsm` at the
  best-sustained spectrum (marginally better mid+hi, within noise). See report +
  outcomes bullet.
- **★ RESEARCH TASK — why does f-distill over-produce 200–400 mm/day precip?** (raised
  2026-07-30.) **Measured** on the 1-GPU CONUS evals with
  `scripts/downscaling/diagnose_eval_histogram_spectrum.py` (mass fraction per magnitude
  band, each source using **its own** dynamic bin edges — see the script's bin-edge warning;
  the script's `--check` reproduces the logged `prediction_frac_of_target` exactly):

  | band (mm/day) | hirov1 `6eff6ig5` | f-distill spectral `y543b0gf` |
  |---|---|---|
  | 50–100 | −17.7% | −7.5% |
  | 100–200 | −0.7% | −4.9% |
  | 200–300 | −1.0% | **+25.4%** |
  | 300–400 | −1.6% | **+51.5%** |
  | 400–600 | −7.9% | **+33.3%** |

  **The signature is a redistribution, not an inflation:** f-distill is *deficient* at
  50–200 mm/day and *excessive* at 200–600, i.e. it moves mass up into the
  moderate-extreme range. The teacher is unbiased to ~2% across 100–600. **Why it went
  unnoticed:** the headline tail selectors are blind to it —
  `prediction_frac_of_target@99.9999` is **0.995** (looks ideal) and @99.99 only **+9.5%**,
  because a steeply-decaying PDF converts a 1.5× density excess into a small *quantile*
  shift. Log-scaled histogram axes hide it too.
  **Investigate:** (a) is this the same defect as the k≈95 spectral bump below — excess
  variance at ~70 km producing too many moderate-extreme cells? Test by checking whether
  the two co-vary across the existing arms (`f7z93y0a` no-spectral, `i26sidsm`,
  `2yhjonz9`/`34rg7wii` γ sweep, `xgcaf2rt`); (b) **does the GAN cause it?** — compare
  against a **DMD2** arm and the GAN-only baseline, since the user's recollection is that
  DMD2 training did not show this; the DMD2 eval config already exists
  (`configs/experiments/2026-05-20-distilled-model-eval/config-dmd2.yaml`, dataset
  `01KRYPVQ3Z5YWQWND9X680GBMD`); (c) is it a **step-count / exposure-bias** effect? The
  1-step and 4-step arms (`xklvoz0n` / `850hcj6i`) are a free test — see
  [[fdistill-step-coupling]]; (d) add a **mid-magnitude density metric** to validation
  (e.g. mass-fraction ratio over fixed physical bands), since no current selector sees this.
- **★ RESEARCH TASK — f-distill power-spectrum excess at k≈95 (~70 km); would the GAN
  lever fix it?** (raised 2026-07-30.) **Measured** on the same evals: f-distill peaks at
  **+48.2% at k=95** of 1153 (`k/k_max`=0.082), a broad bump over k≈50–200 (+30% to +45%);
  hirov1's worst error anywhere below k=384 is **+6.1%**. So this is ~8× the teacher's error
  and specific to the student. Because `k/k_max = 2·Δx/λ`, the fractional position is
  **grid-independent**: 0.082 ⇒ λ ≈ **70 km** (≈24 fine pixels) on any 3 km grid.
  **Why the spectral loss never addressed it — quantitatively.** Training patches are
  512² (`input_shape [1,512,512]`), so the val PSD has **257** wavenumbers and its
  equal-thirds `spec_mae_{lo,mid,hi}` split falls at **k=85 / k=171**. The defect sits at
  k ≈ 0.082·256 ≈ **21** — deep in the **lo** third. Both tuning knobs moved weight *away*
  from it: `min_wavenumber=85` (`xgcaf2rt`) is *exactly* the lo/mid boundary and zeroed the
  defect band entirely, and `band_gamma` weights ∝ `(k/k_max)^γ`, so γ=1 (`34rg7wii`)
  **down-weighted the defect band ~12×** relative to Nyquist. The sweep concluded "γ=1 is
  best on mean but lo becomes the worst band" — that lo cost *is* this defect.
  **On the GAN question:** the discriminator currently taps a **single, deepest** encoder
  level — `feature_index=6, resolution=8, all_res=[512,…,8]`, i.e. `disc_feature_depth=0`,
  so each disc cell covers 64 fine px ≈ **180 km**, ~2.6× coarser than the 70 km defect. So
  the defect scale is plausibly *under-policed* today, and the multi-scale discriminator
  (below) is a reasonable candidate — but note it is motivated in this LOG by the **hi-k**
  winds gap, which is a different scale. **Cheapest decisive test first:** re-run with
  `--disc-feature-depth 1` or `2` (tap `resolution=16`/`32` ⇒ 32/16 px cells ≈ 90/45 km,
  straddling the defect) before building a multi-head disc. Complementary and cheaper still:
  a **non-monotonic band weight** that *lifts* k≈15–40 on the 257-axis instead of tilting
  toward hi — the "non-monotonic bump" item below, but aimed at **lo**, not hi.
- **★ TASK — reduce the tail histograms across ranks** (found 2026-07-28 while setting up
  the single-GPU CONUS eval). `ComparedDynamicTailsHistograms` in `fme/core/histogram.py`
  performs **no cross-rank reduction**, so on any multi-rank run the logged
  `histogram/prediction_frac_of_target/*` tail ratios come from **one rank's shard** —
  on 4 GPUs, a quarter of the samples, and the 99.9999th percentile is exactly where the
  4× smaller sample hurts most. Verified scope: the histogram is the *only* affected
  path — `Mean` / `MeanComparison` both return
  `TensorDictAccumulator.get_distributed_mean()`, so `metrics/crps/*`, `metrics/rmse/*`
  and `power_spectrum/*` were always reduced correctly; the other `Distributed` users in
  the downscaling aggregators are `LossVsNoiseAggregator` (training-only `reduce_sum`)
  and `PairedSampleAggregator` (`gather` for event images). **The shard is contiguous, which
  makes this a seasonal bias rather than just a smaller sample** — the eval loader builds
  with `train=False`, so `_get_sampler` (`fme/downscaling/data/config.py:630-637`) returns
  `ContiguousDistributedSampler`, whose `__iter__` gives rank 0
  `indices[0:N/num_replicas]`: on 4 ranks over CONUS 2023 that is **1 Jan – early Apr only**,
  no summer convection. **Magnitude measured** (2026-07-29 single-GPU re-run): the
  **ground-truth** percentile was understated **8.1% @99.9999** and **21.1% @99.99** — and the
  99.99th moving *more* than the 99.9999th is the seasonal signature (sample-size loss
  predicts the reverse). Tail *ratios* largely self-normalize, so past comparative verdicts
  mostly survive — but not always: hirov1's extreme tail moved 0.936→**0.974**.
  **Consequences:** (a) absolute tails in every multi-rank eval are lower bounds; (b) more
  insidiously, **`best_student_tail.ckpt` and `best_histogram_tail.ckpt` were *selected* on a
  seasonally biased slice**, so every tail-based checkpoint selector in the history above is
  biased, not merely noisier. **Interim workaround:** run evals on one GPU
  (`configs/experiments/2026-07-28-hirov1-vs-spectral-conus-1gpu/`) — but it costs **4×**
  wall clock (12.6 h for hirov1), so this is not standing practice. **Real fix:** add the
  reduction — durable pipeline change → numbered spec under `../specs/` first, and note it
  will shift every tail-selected checkpoint, so it interacts with the spec-13 early-stop
  work below.
- **★ TASK — spectral-aware early stopping / checkpoint selection** (motivated by
  `xgcaf2rt`). Two coupled problems this run exposed: (a) **wasted compute** — it ran
  to 52k steps but its useful spectral optimum was ~2.6k; `val/crps_mean` is flat to
  ~1%, so it gives no stop signal, and the run just drifts (late `spec_mae_mean` +691%,
  `tail_99.99` → 2.2). (b) **selection misses the spectral optimum** — `best_student.ckpt`
  (CRPS-min) and `best_student_tail.ckpt` (tail-min) landed at very different, often
  un-converged fracs (CRPS-min noise-determined; tail-min 3% for midhi vs 29% for base).
  **Proposal:** add a spectral-based early-stop + a `best_student_spec.ckpt` selector to
  `BestStudentCheckpointCallback` — track running-min `spec_mae_mean`, save on
  improvement, and stop after N consecutive vals without spectral improvement (patience).
  This both saves compute and gives a checkpoint that actually sits at the spectral
  optimum (the analyses keep hand-picking mid-training ckpts because no selector does).
  Durable pipeline change → write a numbered spec under `../specs/` first. **Would also
  make every future arm's baseline comparison honest** (all runs stop/select at their
  own spectral optimum instead of an arbitrary flat-CRPS argmin).
- ~~**Lo-only (from-noise@200) ablation: is Student-Hi worth keeping?**~~ — ✅ **DONE
  2026-07-13: Hi is needed, for extreme precip only.** Lo-only from noise@200 (`p337gcg9`)
  matches the full `[Hi→Lo]` bundle (`rmoodemk`) on CRPS (<0.03%) and PSD bias (<1%) across
  all 4 vars incl. PRMSL/winds — **but under-produces the extreme precip tail**
  (`tail_99.9999_PRATEsfc` 1.01→0.93; wind tails unchanged). This *confirms* the MoE design
  rationale: the high-noise regime exists to generate the rare precip extremes (σ=200 can't
  resynthesize them), and Hi helps precip only. **Keep Hi where extreme precip matters;
  Lo-only (2 NFE, no 46M Hi expert) suffices for winds/PRMSL + precip mean/spectrum.**
  Closes the deferred MoE decision (`MOE_DISTILLATION_STATUS.md:117–119, 254`). Config
  `config-lo-only.yaml`, launcher `run-lo-only.sh`, beaker `01KXEYCC9HAZ7F1G85E3KRPKFD`.
  Write-up: [report](reports/2026-07-13-lo-only-from-noise200-ablation-p337gcg9.md).
  _Follow-ups: variable-scoped Hi (precip-only high-σ steps); confirm `tail_99.99`;
  re-confirm on maritime continent (heavier precip tails → Hi should matter more)._
- **★ PLANNED — native f-distill step-count sweep (1 / 2 / 4 step).** Train a native
  **1-step** (task #3) and native **4-step** (task #2) student from scratch
  (`--student-steps 1|4`, spectral W=1e-2), baseline = the 2-step `i26sidsm`; find the
  quality-vs-NFE knee. No warm-start (training is short; a native run at each step count
  is the fair test — a 1-step *eval* of the 2-step model only lower-bounds native-1-step).
  **Mechanism:** f-distill training is *not* step-independent — `student_sample_steps` sets
  the discrete `t_list` nodes `t_student` is drawn from and whether `input_student` is pure
  noise (1-step) or real-data re-noised (N-step interior, teacher-forced → inference
  exposure bias). See [[fdistill-step-coupling]] / `dmd2.py:97–116`. Write-up:
  [report](reports/2026-07-13-fdistill-step-count-sweep-TBD.md).
- **Next (experiments):** port the tuned spectral config to the multi-variable MoE runs
  with per-variable `variable_weights` (up-weight the worst-spectrum variables). Config
  choice for the port: **flat all-band (`i26sidsm`)** is the safe default; **`band_gamma`
  ≈ 0.5–1** gives a small, monotonic hi-k gain (sweep closed 2026-07-14) if hi texture is
  the priority and a modest lo cost is acceptable. The remaining lever beyond the
  monotonic ramp is a **non-monotonic** band weight that lifts hi/mid **without**
  down-weighting lo (needs a `SpectralMatchingLoss` change; `band_gamma` can only tilt
  monotonically today, and at γ=1 lo is already the limiting band).

---

## Outcomes log

_Reverse-chronological; one line per finding, linking the run report._

- **2026-07-29** — ✅ **Multi-rank tail histograms were measuring *winter only*, not a random
  quarter — extremes understated 8–21%.** Single-GPU CONUS re-runs of hirov1 (`6eff6ig5`) and
  the spectral student (`y543b0gf`) against their 4-GPU twins (`j3thqivd` / `x2nyzmzh`)
  isolate the bug. The eval loader uses **`ContiguousDistributedSampler`** (`train=False`), so
  rank 0 held the **first** quarter of the record — **1 Jan – early Apr 2023**, no summer
  convection. The **ground-truth** percentile (same observed data in every run → pure
  artifact, and bit-identical across both models within each rank count, a clean check) was
  understated **8.1% @99.9999** and **21.1% @99.99**; the 99.99th moving *more* is the
  seasonal signature — sample-size loss predicts the reverse. Tail *ratios* mostly
  self-normalize (shared shard cancels), and CRPS/RMSE/PSD agree to **≤0.3%**, confirming
  those were always reduced correctly, so the 2026-07-13 **CRPS/PSD** verdict stands. **What
  does change:** hirov1's extreme tail 0.936→**0.974** (under-produces 2.6%, not 6.4%) — ratio
  robustness is not a property to assume, and that report's tail rows still rest on an
  un-rerun baseline arm (`flzvb6tp`). Head-to-head, the 2-step student is within **3.3% CRPS
  / 8.3% PSD** of full diffusion at ~17× lower per-batch cost and better @99.9999 (0.995 vs
  0.974), but over-produces **@99.99 by 9.5%** — now its clearest weakness. Cost 4.02× wall
  clock (12.6 h for hirov1) → fix the reduction rather than repeat the workaround. See
  [report](reports/2026-07-28-hirov1-vs-spectral-conus-1gpu.md).
- **2026-07-14** — ➕ **`band_gamma` hi-k tilt is a mild, monotonic positive.** The
  `{0, 0.5, 1}` sweep (`i26sidsm` / `2yhjonz9` / `34rg7wii`) shows, at each run's
  best-sustained spectrum, that tilting the spectral budget toward high-k does exactly
  what it targets and monotonically: **hi `spec_mae` 0.074→0.066→0.050**, overall
  **mean 0.043→0.038→0.035**, paid for by **lo 0.022→0.024→0.037**; `crps` and tails
  tied. γ=1 is the best-mean point (−19% vs flat) but makes lo the worst band. Unlike the
  hard cut `xgcaf2rt` (which zeroed lo and came back neutral), the gentle tilt keeps lo
  in the loss so it degrades gracefully and the net mean improves. **Small win; the
  bigger lever is likely a non-monotonic mid/hi *bump* that lifts hi without starving lo
  (needs a `SpectralMatchingLoss` change).** Both arms again show the checkpoint-selection
  trap (best_tail lands at 8–10%, spectrally unconverged → deploy the best-sustained /
  spec-13 checkpoint, not best_tail). Confirm the gain out-of-sample before adopting over
  flat. See [γ0.5](reports/2026-07-13-prate-spectral-gamma0p5-2yhjonz9.md) ·
  [γ1](reports/2026-07-13-prate-spectral-gamma1-34rg7wii.md).
- **2026-07-13** — ➖ **The mid+high band-cut arm `xgcaf2rt` is roughly neutral**
  (corrected — an earlier entry called it ❌ degrade; that was a windowing artifact,
  see the report Caveats). Compared **at the selected checkpoints** and at each run's
  **best-sustained spectrum** (both step-controlled), midhi and flat-band `i26sidsm`
  are tied (`spec_mae_mean` 0.044 vs 0.043), midhi marginally better on mid+hi — a
  weak nod to the hypothesis, within noise. No clear win, no clear loss; flat all-band
  weighting stays the default as the simpler config. **The real finding: CRPS/tail
  checkpoint selection is decoupled from spectral quality** — CRPS is flat to ~1% so
  its argmin is noise, and tail-min landed at 3% (midhi) vs 29% (base); the "winner"
  flips by which selector you read. Motivates a spectral-aware early-stop/selection
  criterion (new planned item). See
  [report](reports/2026-07-10-prate-spectral-midhi-xgcaf2rt.md).
- **2026-07-13** — ✅ **Lo-only ablation: Student-Hi is needed — for extreme precip only.**
  A single-model Student-Lo from noise@200 (`p337gcg9`, 2 NFE) matches the full `[Hi→Lo]`
  bundle (`rmoodemk`, 3 NFE) on CRPS (<0.03%) and power-spectrum bias (<1%) across all 4
  vars incl. PRMSL/winds — **but under-produces the extreme precip tail**
  (`tail_99.9999_PRATEsfc` 1.01→0.93; wind tails unchanged). This *confirms* the MoE design
  rationale: the high-noise regime (σ up to 2000) exists to generate the rare precip
  extremes — a σ=200 start doesn't destroy enough signal to resynthesize them — and Hi
  helps precip only. Keep Hi where extreme precip matters; Lo-only suffices otherwise.
  See [report](reports/2026-07-13-lo-only-from-noise200-ablation-p337gcg9.md).
- **2026-07-13** — ✅ **Held-out eval confirms the spectral loss is a real, generalizing
  win.** On X-SHiELD AMIP control (out-of-sample vs the training val period), 100km→3km,
  the spectral student beats the GAN-only baseline on **power-spectrum bias 3.5–4.5×**
  (CONUS 0.46→0.13, maritime continent 0.60→0.13) with CRPS ~3% better and tails
  near-ideal — no regression, both regions, both bundling `best_student_tail.ckpt` (fair).
  Confirms the training-val `i26sidsm` result transfers out-of-sample; de-risks porting
  the loss to the MoE runs. See
  [report](reports/2026-07-13-prate-eval-baseline-vs-spectral.md).
- **2026-07-13** — ➖ **Reduce-GAN arm `6dotglmg` (gan=3e-4) reported: inconclusive.**
  Marginally better spectrum + tails than `i26sidsm` at matched steps (tail 1.10 vs 1.17
  @14k), no downside — but it **crashed@14k, before the late-drift regime** (baseline
  drifts +61%→+632% only after 14k; at 14k both are ~+60%). The headline "does low-GAN
  tame late drift" question is untested; the +92%-vs-+632% gap was a run-length artifact.
  Re-run longer with checkpointing. See
  [report](reports/2026-07-09-prate-spectral-lowgan-fix-6dotglmg.md).
- **2026-07-09** — Launched the first valid reduce-GAN arm `6dotglmg` (gan=3e-4);
  the earlier `gpx5574t` low-GAN run was invalid (pre-fix target, crashed early).
- **2026-07-08** — ✅ **Corrected spectral-matching loss is a clear win.** Compared
  **checkpoint-matched** (each run at its own `best_student_tail` — the checkpoint
  actually deployed to eval: baseline @2470, `i26sidsm` @7930), `i26sidsm` beats the
  GAN-only baseline **~3–4× on `spec_mae`** (mean 0.11 vs 0.36) while keeping the
  independent metrics tied (`crps_mean` 0.105 vs 0.105; tails both ~ideal ~1.0), without
  fighting distillation (`f_distill_loss` ≈ baseline). Same direction/magnitude as the
  held-out X-SHiELD eval (−71–78% PSD bias). _(An earlier entry cited "5–20×" from
  last-step values; that compared the two runs' drifted end states and overstated it —
  the checkpoint-matched gain is ~3–4×.)_ Late drift persists → true spectral optimum is
  mid-training, missed by CRPS/tail selectors. See
  [report](reports/2026-07-08-prate-spectral-fix-i26sidsm.md).
- **2026-07-07** — ❌ First spectral arms (`s4abc6ba`, `gpx5574t`) were net-harmful
  due to two coupled bugs (matched teacher's x0 *prediction* not a *sample*;
  average-then-spectrum instead of spectrum-then-average). Fixed in `e29f797`.
