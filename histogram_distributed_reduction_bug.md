# Histogram / percentile aggregation is not reduced across ranks (multi-GPU)

## Summary

Under multi-GPU (distributed) runs, the `DynamicHistogram`-backed aggregation
outputs — full-field histograms, the 99.99/99.9999th percentiles derived from
them, and the `histogram_tail_metric` used for `best_histogram_tail.ckpt`
selection — are **computed on each rank independently and never reduced across
ranks**. Only rank 0's values are logged. Because the downscaling data loaders
shard the dataset into **contiguous time-chunks** across ranks
(`ContiguousDistributedSampler`), rank 0 sees only the earliest `1/world_size`
of the evaluation/validation period. For regional data whose extremes are
seasonally clustered, this silently truncates the tail of every logged
histogram and corrupts extreme-tail checkpoint selection.

All the *mean*-based metrics (CRPS, MAE, power spectrum, time-mean maps) are
correctly all-reduced. The bug is isolated to the histogram/percentile path.

## How it was found

Investigating a discrepancy between the wandb evaluation run
`ai2cm/andrep-downscaling/j3thqivd` (entrypoint: `fme.downscaling.evaluator`,
4× B200) and manual histograms of the X-SHiELD 3km target data:

- The logged **target** histogram for `PRATEsfc` topped out at **891.6 mm/day**
  (99.9999th pct ≈ 538 mm/day), but the underlying 3km validation data contains
  an event of **1879.7 mm/day** on 2023-06-24T06:00 at ~23°N, ~288°E, plus
  similar extremes on 2023-07-08 and 2023-07-09.
- The event is unambiguously **inside** the evaluator's crop (lat 22–50,
  lon 228–300) and **inside** the evaluated time subset (all 1473 contiguous
  6-hourly steps of 2023 were loaded; all 92 batches ran).
- `DynamicHistogram` auto-grows its bin edges to the max value it receives and
  does not clip, and the fine target bypasses patch compositing entirely
  (`PatchPredictor.generate_on_batch`: `target = batch.fine.data`). So a top bin
  at 891 means the 1879 value **never reached rank 0's histogram**.

The extremes cluster in ~4 summer timesteps (late June / July 2023), which fall
on ranks 1–2 under contiguous time-sharding. Only rank 0's histogram (Jan 1 –
early April 2023) is logged, so all summer extremes are dropped.

## Root cause

1. **Contiguous time-sharding.** For validation/eval
   (`PairedDataLoaderConfig.build(train=False, ...)`),
   `_get_sampler` returns `ContiguousDistributedSampler`
   (`fme/downscaling/data/datasets.py:613`), which assigns each rank a
   **contiguous** block of the time-ordered dataset:
   `chunk_size = len // num_replicas; indices[rank*chunk : ...]`.
   With 4 ranks and 1473 samples, rank 0 = indices 0–367 = 2023-01-01 through
   ~2023-04-03. (`local_batch_size(16) = 16//4 = 4`; 368/4 = 92 batches, matching
   the run log exactly.)

2. **No cross-rank reduction of the histogram.** `fme/core/histogram.py`
   (`DynamicHistogram`, `DynamicHistogramAggregator`,
   `ComparedDynamicTailsHistograms`, `ComparedDynamicHistograms`) contains **no**
   distributed code. In the downscaling aggregators, the histogram adapters and
   `GenerationAggregator.get_wandb()/get_summary()` log rank 0's local numpy
   counts directly. `Evaluator.run` (`fme/downscaling/evaluator.py:74-76`) calls
   `aggregator.get_wandb()` then `wandb.log(...)` with no gather/all_reduce.

This is consistent with the codebase contract that `fme/core` aggregation
primitives are single-process and the **caller** performs the reduction — which
every mean-based downscaling/ACE aggregator does (`dist.reduce_mean` /
`reduce_sum` / `gather`), but the histogram path does not.

## Scope of impact

### Affected

- **Downscaling evaluator diagnostics** (`fme.downscaling.evaluator`,
  multi-GPU): logged `histogram/*`, `histogram/99.99th-percentile/*`,
  `histogram/99.9999th-percentile/*`, `histogram/prediction_frac_of_target/*`,
  and `evaluator_maps_and_metrics.nc` histogram vars reflect only rank 0's
  contiguous time-chunk.
- **Downscaling training-time checkpoint selection**
  (`fme/downscaling/train.py`): `valid_one_epoch` builds a per-rank
  `GenerationAggregator`; `get_summary().histogram_tail_metric`
  (`histogram/prediction_frac_of_target/99.9999th-percentile`) is computed on
  rank 0's chunk, and `save_best_checkpoint` (runs on `dist.is_root()`) selects
  `best_histogram_tail.ckpt` from it. **The one selection metric that targets
  the tail is computed blind to most of the tail.** Both the
  `validation_aggregator` (`Aggregator`, `main.py:934`) and the
  `generation_aggregator` carry unreduced `ComparedDynamicTailsHistograms`.
- **Downscaling no-target generation diagnostics**
  (`fme/downscaling/aggregators/no_target.py`, `DynamicHistogramAggregator`):
  used in the contiguous dataset-generation / no-target inference path — same
  per-rank gap (diagnostic only; the generated dataset itself is written per
  contiguous chunk and is unaffected).

### NOT affected (correctly reduced)

| Metric | Accumulator | Reduction |
|---|---|---|
| CRPS (`metrics/relative_crps_bicubic`, `best_ckpt` metric) | `MeanComparison` → `TensorDictAccumulator` | `dist.reduce_mean` |
| MAE / positional comparisons | `MeanComparison` | `dist.reduce_mean` |
| Zonal power spectrum (`ZonalPowerSpectrum{Aggregator,Comparison}`) | `Mean` → `TensorDictAccumulator` | `dist.reduce_mean` |
| Time-mean / main `Aggregator` maps | running sums | `dist.reduce_sum` (`main.py:176-177`) |
| **Histogram + percentiles + `histogram_tail_metric`** | `DynamicHistogram` (numpy counts) | **none** |

- `best_checkpoint` (best CRPS) selection is a correctly reduced full-set mean.
  Note: CRPS is a mean skill score and is **not** correlated with extreme-tail
  fidelity, so it does not compensate for the broken tail metric.
- Single-GPU runs are unaffected (`_get_sampler` returns `None`; rank 0 sees
  everything).
- Downscaling **event evaluators** load a specific window on every rank, so
  they are unaffected.

### Other components using the core histogram

- **ACE inference** (`fme/ace/aggregator/inference/histogram.py`,
  `HistogramAggregator` → `ComparedDynamicHistograms`): has the **same
  code-level gap** — `get_logs` does not reduce across ranks. **However, the
  practical impact is expected to be minor**, because ACE inference shards
  **initial conditions / ensemble members**, not time
  (`fme/ace/data_loading/inference.py:297`:
  `i_member % total_data_parallel_ranks != data_parallel_rank`). Each rank still
  processes the full forecast time range and full global spatial extent for its
  ICs, and global fields contain tail events at essentially every timestep
  (e.g. tropical convection). So rank 0's histogram is a representative — if
  slightly noisier (1/world_size of the ensemble) — sample of the true
  distribution rather than a systematically truncated one. It is still
  technically incorrect (the logged histogram should be the all-reduced
  ensemble) and worth fixing for exactness, but it does not produce the
  qualitative truncation seen in regional downscaling.
- The failure is specific to **regional** data where extremes are both spatially
  localized and seasonally clustered, so a contiguous time-chunk can miss them
  entirely.

## Proposed fix

Add cross-rank reduction of the histogram bin counts, invoked from the
downscaling aggregator layer (keeping the "core provides the primitive, caller
invokes" split):

1. **Core primitive.** Add `reduce_across_ranks(dist)` to `DynamicHistogram`
   (and a pass-through on `DynamicHistogramAggregator` /
   `ComparedDynamicTailsHistograms` / `ComparedDynamicHistograms`). Because each
   rank's `bin_edges` grew independently to its own local `[min, max]`, the
   counts cannot be summed directly — **edges must be reconciled first**:
   gather each rank's `[vmin, vmax]` (and bin size), establish common edges
   spanning the global range, re-bin/interpolate each rank's counts onto the
   common edges, then `all_reduce(SUM)` the integer counts.
2. **Invocation.** Call the reduction in
   `GenerationAggregator.get_wandb()/get_summary()` (and the no-target adapter)
   before producing logs/percentiles, so both `Evaluator.run` and
   `valid_one_epoch`/`save_best_checkpoint` see the full-set histogram.
3. **ACE.** Apply the same reduction in `HistogramAggregator.get_logs` for
   correctness (low urgency given IC-sharding).

## Testing

- Parallel/regression test asserting the distributed histogram equals the
  single-rank histogram on identical data (generate single-rank baseline via
  `python -m pytest`, verify under `torchrun` per the repo's spatial-parallel
  testing convention). Cover the mismatched-edges case (ranks with disjoint
  value ranges) explicitly.
- A targeted test that a contiguously-sharded, temporally-clustered extreme is
  present in the reduced histogram tail (guards the exact regression found here).

## Reference

- wandb run: `ai2cm/andrep-downscaling/j3thqivd`
  (beaker `01KT2JXJYXFQQ47GHXQBJ67NQ3`), commit `d46c1379f5`.
- Key files: `fme/core/histogram.py`,
  `fme/downscaling/data/datasets.py:613` (`ContiguousDistributedSampler`),
  `fme/downscaling/evaluator.py:74-76`,
  `fme/downscaling/train.py:257,322-328`,
  `fme/downscaling/aggregators/generation.py:435`,
  `fme/downscaling/aggregators/{main.py,adapters.py,no_target.py}`,
  `fme/ace/aggregator/inference/histogram.py`.
