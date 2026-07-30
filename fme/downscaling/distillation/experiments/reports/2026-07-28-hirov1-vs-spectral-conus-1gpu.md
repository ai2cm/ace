<!--
Comparison-eval report: hirov1 baseline (full diffusion) vs the spectral-loss
distilled student (i26sidsm), CONUS 2023 held-out X-SHiELD AMIP, run on a SINGLE
GPU so the tail histograms cover the whole sample population. Fill the tables via
`check_runs.py --compare-eval <hirov1_run> <spectral_run> --project
andrep-downscaling` (the tool is a generic A-vs-B diff of two eval-run summaries;
relabel teacher/distilled -> hirov1/spectral).
-->
# Eval comparison — **hirov1** baseline vs **spectral-loss** distilled (CONUS 2023, 100km→3km, single GPU)

_Status: **complete** (launched 2026-07-28, both finished 2026-07-29, exit 0) — hirov1
`6eff6ig5`, spectral `y543b0gf`._

## Why this run

Re-runs the CONUS held-out eval on **one GPU** to get tail statistics over the whole
record. `ComparedDynamicTailsHistograms` (`fme/core/histogram.py`) performs **no cross-rank
reduction**, so every prior multi-rank eval logged
`histogram/prediction_frac_of_target/*` from a **single rank's shard**. One rank makes the
histogram see the full population.

**The shard is contiguous in time, not a random subsample — this is the crux.** The
evaluator builds its loader with `train=False`
(`fme/downscaling/evaluator.py:170,201`), and `PairedDataLoaderConfig._get_sampler`
(`fme/downscaling/data/config.py:630-637`) returns a **`ContiguousDistributedSampler`** in
that case (it has no `shuffle` field to take the striding branch).
`ContiguousDistributedSampler.__iter__` (`fme/downscaling/data/datasets.py:641-655`) assigns
`indices[rank * chunk_size : ...]`, so **rank 0 receives the first quarter of the record** —
368 of 1472 samples, i.e. roughly **1 Jan – early Apr 2023**.

So the 4-GPU tail histograms characterized **CONUS winter/early-spring precipitation only**.
That is a *seasonal* bias, not a smaller sample: it omits summer convective precipitation
entirely. This matters for how the numbers below are read, and it makes the fix higher
priority than "the estimate is noisier."

**Scope of the bug (verified in code, 2026-07-28):** the missing reduction is confined to
the histogram path. `Mean` and `MeanComparison` both return
`TensorDictAccumulator.get_distributed_mean()`, so `metrics/crps/*`, `metrics/rmse/*`,
and `power_spectrum/*` **were** correctly reduced across ranks. The only other `Distributed`
users in the downscaling aggregators are `LossVsNoiseAggregator` (training-only,
`reduce_sum`) and `PairedSampleAggregator` (`gather`, for event sample images).

**Consequences for comparison:**
- **CRPS and power-spectrum bias are comparable** to the earlier 4-GPU runs
  (`flzvb6tp` / `x2nyzmzh`, [report](2026-07-13-prate-eval-baseline-vs-spectral.md)).
- **Tail ratios are not** — the new numbers supersede the old ones rather than
  reproducing them. Expect movement even with an identical model.

Also the first eval of hirov1 against the spectral student directly (the prior CONUS
comparison was spectral vs the *GAN-only distilled* baseline `f7z93y0a`, not hirov1).

## Configuration

`configs/experiments/2026-07-28-hirov1-vs-spectral-conus-1gpu/`, launched with:

```
bash configs/experiments/2026-07-28-hirov1-vs-spectral-conus-1gpu/run.sh all
```

| | hirov1 (baseline) | spectral (distilled) |
|---|---|---|
| checkpoint | `best_histogram_tail.ckpt` | `best_student_tail.ckpt` |
| checkpoint dataset | `01KNM6H3JB1ZNS76HX17AAZRF7:checkpoints` | `01KX00NA0DMZ99S3TKN1RYJKKQ:fastgen/…-prate-spectral-fix/student_checkpoints` |
| sampler | full diffusion (checkpoint default) | fastgen, `num_diffusion_generation_steps: 2` |
| training run | — | `i26sidsm` |

Both: CONUS lat 22–50 / lon 228–300, X-SHiELD AMIP 100km→3km from 2023-01-01, `n_samples: 4`,
patched generation with composite prediction, same 3 events. `batch_size: 4` (global batch;
16 across 4 ranks == 4 on one rank, so per-GPU memory matches the earlier runs — the sampler
shards by rank, so one rank still evaluates the full CONUS set). Not `--preemptible`: ~4×
longer on one GPU with no resume path.

## Artifacts

| role | wandb run | commit | Beaker experiment | checkpoint |
|---|---|---|---|---|
| hirov1 (baseline) | `6eff6ig5` — https://wandb.ai/ai2cm/andrep-downscaling/runs/6eff6ig5 | [`8b9edba`](https://github.com/ai2cm/ace/commit/8b9edba3f) | [`01KYP41B245ZXBNH3QGJEGR74N`](https://beaker.org/ex/01KYP41B245ZXBNH3QGJEGR74N) | `best_histogram_tail.ckpt` |
| spectral (`i26sidsm`) | `y543b0gf` — https://wandb.ai/ai2cm/andrep-downscaling/runs/y543b0gf | [`8b9edba`](https://github.com/ai2cm/ace/commit/8b9edba3f) | [`01KYP41DEZ7DND9YFY77XVYEH7`](https://beaker.org/ex/01KYP41DEZ7DND9YFY77XVYEH7) | `best_student_tail.ckpt` |

Prior 4-GPU counterparts of these exact configurations, used for the artifact check in
§"What the shard-local histogram cost us": hirov1 `j3thqivd`, spectral `x2nyzmzh`
([report](2026-07-13-prate-eval-baseline-vs-spectral.md)).

Both 1-GPU runs exited 0 and processed **368 batches × 4 = 1472 samples** — the full CONUS
2023 set. On 4 ranks at `batch_size: 16` that is 92 batches/rank, so rank 0's histogram saw
368 samples — **exactly 1/4, and contiguously the first 1/4** (see the sampler note above).
Runtime cost of one rank: hirov1 3.13 h → **12.59 h** (4.02×), spectral 0.21 h → **0.76 h**
(3.6×).

Regenerate the head-to-head tables with:

```
conda run -n fme python -m fme.downscaling.distillation.check_runs \
    --compare-eval 6eff6ig5 y543b0gf --project andrep-downscaling
```

## Head-to-head: hirov1 vs spectral student (whole-dataset, 1 GPU)

Single output variable **PRATEsfc**. Δ = spectral − hirov1.

| metric | hirov1 (full diffusion) | spectral (2-step) | Δ | read |
|---|---|---|---|---|
| CRPS (lower better) | **8.082e-6** | 8.350e-6 | +3.3% | hirov1 better |
| RMSE (lower better) | **8.078e-5** | 8.677e-5 | +7.4% | hirov1 better |
| relative CRPS vs bicubic | **0.4786** | 0.5009 | +4.7% | hirov1 better |
| power-spectrum bias (lower better) | **0.1235** | 0.1338 | +8.3% | hirov1 better |
| tail ratio @99.9999 (~1.0) | 0.9737 | **0.9950** | — | **spectral better** (−2.6% vs −0.5% from ideal) |
| tail ratio @99.99 (~1.0) | **1.0032** | 1.0947 | — | hirov1 better (spectral over-produces +9.5%) |

The 2-step distilled student lands within **3.3% CRPS / 8.3% spectrum bias** of the
full-diffusion model at **~17× lower cost per batch** (7.3 s vs 123 s), beats it on the
extreme 99.9999th tail, and over-produces the 99.99th by 9.5%.

## What the shard-local histogram cost us

Comparing each 1-GPU run against its 4-GPU counterpart of the *same configuration and
checkpoint* isolates the histogram bug — the only intended difference is **which** samples
the histogram saw (all of 2023 vs rank 0's contiguous Q1).

**Metrics that were already reduced correctly (control group).** All agree to ≤0.3%,
confirming the code reading that `Mean`/`MeanComparison` reduce via
`get_distributed_mean()`:

| metric | hirov1 4-GPU → 1-GPU | spectral 4-GPU → 1-GPU |
|---|---|---|
| CRPS | 8.0868e-6 → 8.0819e-6 (−0.06%) | 8.3519e-6 → 8.3505e-6 (−0.02%) |
| RMSE | 8.0902e-5 → 8.0780e-5 (−0.15%) | 8.6692e-5 → 8.6774e-5 (+0.10%) |
| power-spectrum bias | 0.12554 → 0.12352 (−1.6%) | 0.13347 → 0.13379 (+0.24%) |

**The histogram (the affected metric).** The cleanest evidence is the *target* percentile,
recovered as `prediction ÷ prediction_frac_of_target`. It is ground truth — the same
observed CONUS precipitation in all four runs — so any movement is pure artifact. It comes
out **bit-identical across the two models within each GPU count** (ratio 1.000000, a good
check on the derivation) and shifts sharply with rank count:

| ground-truth percentile (PRATEsfc) | 4-GPU (Jan–early Apr only) | 1-GPU (full year, 1472) | error |
|---|---|---|---|
| target @99.9999 | 0.0066552 | 0.0071962 | **−8.1% understated** |
| target @99.99 | 0.0025260 | 0.0030595 | **−21.1% understated** |

So every absolute tail number in a prior multi-rank report is low by 8–21%.

**The ordering confirms the seasonal mechanism.** Note that the **99.99th** moved *more*
(−21.1%) than the **99.9999th** (−8.1%). Pure sample-size shrinkage predicts the opposite —
the most extreme quantile is the one that suffers most from having ¼ the draws. A
winter-only shard explains the observed ordering: excluding summer convection depresses the
broad upper distribution (the 99.99th) substantially, while the very most extreme events
(the 99.9999th) include winter atmospheric rivers, which the Q1 shard *does* contain. This
is the anomaly that pointed to the contiguous sampler in the first place.

The **ratios**, though, partly self-normalize — prediction and target percentiles are
computed on the same shard and shift together — which is why the historical *comparative*
verdicts largely survive:

| tail ratio | 4-GPU | 1-GPU | change |
|---|---|---|---|
| spectral @99.9999 | 0.99545 | 0.99495 | −0.05% (unchanged) |
| spectral @99.99 | 1.06599 | 1.09466 | +2.7% |
| hirov1 @99.9999 | 0.93593 | 0.97370 | **+4.0%** |
| hirov1 @99.99 | 0.99966 | 1.00321 | +0.4% |

**One verdict does change materially.** On 4 GPUs hirov1 looked like it under-produced the
99.9999th percentile by **6.4%**; over the full year it under-produces by only **2.6%**. The
winter-only shard made the full-diffusion model's extreme tail look considerably worse than
it is. The spectral student's ratio, already ~1.0, barely moved — its ratio was robust by
luck of sitting at the fixed point, not by construction.

**Comparability caveat:** the 4-GPU and 1-GPU tail numbers describe **different seasons**
(Q1 vs the full year), not merely different sample sizes. Read the Δs above as "what the old
metric was measuring vs what the new one measures," not as a convergence study.

`power_spectrum_of_single_sample_time_mean` also moved for hirov1 (0.0788 → 0.0877, +11%),
but that metric is defined on a *single* stochastic sample per batch and the batch
composition changed (368 batches vs 92/rank), so that is sampling noise, not the bug —
`MeanMapAggregator` uses `Mean` and reduces correctly.

## Figures  <!-- generated separately -->

- Histograms / spectra via `scripts/downscaling/plot_compared_histograms.py`
  + `plot_beaker_histograms.py` on the per-event netCDFs (`fetch_beaker_dataset`).

## Caveats  <!-- pre-filled; keep in the write-up -->

- **Both checkpoints were selected by tail metrics during training validation**
  (`best_histogram_tail` / `best_student_tail`). If that validation ran multi-rank, the
  *selection itself* saw a shard, so neither checkpoint is guaranteed to be its run's true
  tail optimum. Kept as-is to preserve comparability with the prior held-out eval.
- **Checkpoint-selection trap:** `best_student_tail` for `i26sidsm` lands mid-training, which
  is near its spectral optimum — a favorable coincidence, not a principled selector. See the
  LOG's spectral-aware early-stop item (spec 13).
- **Sampler asymmetry is the point, not a confound:** hirov1 runs the full diffusion sampler
  vs 2 fastgen steps. This is a quality-vs-NFE comparison, not an ablation.

## Verdict

- **✅ The single-GPU eval did what it was for, and the bug was worse than assumed.** It is
  not that the histogram saw a random quarter of the samples — the eval loader uses
  `ContiguousDistributedSampler`, so rank 0 saw the **first** quarter: **CONUS
  winter/early-spring only (≈ 1 Jan – early Apr 2023)**, with summer convection entirely
  absent. The recovered ground-truth percentile — the same observed data in every run, so a
  pure artifact measurement — was **understated 8.1% @99.9999 and 21.1% @99.99**. That the
  99.99th moved *more* than the 99.9999th is the signature of seasonal bias rather than
  sample-size loss.
- **These tails supersede the old ones; they are the best available estimate, not a
  converged one.** The 1-GPU histogram is still 300 dynamic bins, now over a wider range, and
  there is no third sample size here to demonstrate convergence.
- **The historical comparative conclusions largely survive.** Tail *ratios* self-normalize
  (numerator and denominator share the shard), so the spectral arm of the 2026-07-13
  comparison barely moved — @99.9999 0.99545 → 0.99495. The control metrics (CRPS, RMSE,
  power-spectrum bias) agree to ≤0.3% with the 4-GPU runs, confirming they were reduced
  correctly all along, so **the CRPS and PSD conclusions of that report need no revision.**
  Its *tail* rows, though, rest partly on an arm not re-run here — the GAN-only baseline
  `flzvb6tp` (ratio 1.009) — and hirov1 proves ratio robustness is not a property to assume.
  Treat those tail rows as approximate pending a re-run rather than as confirmed.
- **One number does change: hirov1's extreme tail.** 0.936 → 0.974 — the full-diffusion model
  under-produces the 99.9999th percentile by 2.6%, not the 6.4% the sharded histogram
  reported. Treat the ratio's robustness as luck rather than a property: it held for the
  spectral student because that student already sat at ~1.0, and it did *not* hold for
  hirov1.
- **Head-to-head, the 2-step student is a strong trade.** Within **3.3% CRPS** and **8.3%
  spectrum bias** of full diffusion at **~17× lower per-batch cost**, and *better* on the
  extreme 99.9999th tail (0.995 vs 0.974). Its one real weakness is the moderate tail: it
  over-produces the 99.99th by **9.5%** (1.095) where hirov1 is nearly exact (1.003) — a
  more visible flaw now that the histogram is trustworthy, and a concrete target for the
  spectral/GAN tuning.
- **Cost of the workaround:** 4.02× wall clock (hirov1 3.13 h → 12.59 h). Acceptable once,
  not as standing practice — which is the argument for actually fixing the reduction.
- **Next actions:**
  1. **Fix the reduction** in `ComparedDynamicTailsHistograms` (spec first — LOG ★ TASK).
     Higher priority than a noise fix: with `ContiguousDistributedSampler`, every multi-rank
     tail metric is a **seasonally biased** estimate, and that includes training-time
     validation — so `best_student_tail.ckpt` / `best_histogram_tail.ckpt` were selected on
     whatever contiguous slice rank 0 happened to hold.
  2. Re-run the **GAN-only baseline** `flzvb6tp` on one GPU to close the one arm of the
     2026-07-13 tail comparison this run left untested (~45 min, it is a 2-step student).
  3. Re-check the **maritime continent** pair (`fg9byv9y` / `l6vv7yx0`) the same way —
     heavier precip tails and a different seasonal cycle mean a different, likely larger,
     shard bias.
  4. Investigate the spectral student's **+9.5% @99.99 over-production** — read as +6.6% on
     the winter-only shard, now clearly its weakest metric.
