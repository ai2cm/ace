<!--
Comparison-eval report: hirov1 baseline (full diffusion) vs the spectral-loss
distilled student (i26sidsm), CONUS 2023 held-out X-SHiELD AMIP, run on a SINGLE
GPU so the tail histograms cover the whole sample population. Fill the tables via
`check_runs.py --compare-eval <hirov1_run> <spectral_run> --project
andrep-downscaling` (the tool is a generic A-vs-B diff of two eval-run summaries;
relabel teacher/distilled -> hirov1/spectral).
-->
# Eval comparison — **hirov1** baseline vs **spectral-loss** distilled (CONUS 2023, 100km→3km, single GPU)

_Status: **running** (launched 2026-07-28) — hirov1 `6eff6ig5`, spectral `y543b0gf`.
Awaiting results; tables below are unfilled._

## Why this run

Re-runs the CONUS held-out eval on **one GPU** to get trustworthy tail statistics.
`ComparedDynamicTailsHistograms` (`fme/core/histogram.py`) performs **no cross-rank
reduction**, so every prior multi-rank eval logged
`histogram/prediction_frac_of_target/*` from a **single rank's shard** — on 4 GPUs, ~1/4
of the CONUS samples, and the extreme tails (99.9999th percentile) are exactly where a
4× smaller sample hurts most. One rank makes the histogram see the full population.

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

Fill the tables with:

```
conda run -n fme python -m fme.downscaling.distillation.check_runs \
    --compare-eval 6eff6ig5 y543b0gf --project andrep-downscaling \
    --out fme/downscaling/distillation/experiments/reports/
```

## CRPS  (`metrics/crps/<VAR>` — lower better)

| variable | hirov1 | spectral | Δ (spectral−hirov1) |
|---|---|---|---|
| ... | | | |

## Tail ratio  (`histogram/prediction_frac_of_target/<pct>th-percentile/<VAR>` — ~1.0 ideal)

_Whole-dataset for the first time; do **not** read these against the 4-GPU numbers._

| variable | hirov1 | spectral |
|---|---|---|
| ... | | |

## Power spectrum bias  (`power_spectrum/mean_abs_norm_bias/<VAR>` — lower better)

| variable | hirov1 | spectral |
|---|---|---|
| ... | | |

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

## Verdict  <!-- HUMAN: fill this in -->

- **Does the 2-step spectral student hold up against hirov1?** per-variable summary.
- **What changed vs the 4-GPU tails?** the point of the run.
- **Next action:** _..._
