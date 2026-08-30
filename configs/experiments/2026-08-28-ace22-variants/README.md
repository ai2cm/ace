# ACE2.2 variant campaign (2026-08-28)

Two training variants of the 6-hourly ACE2.2-ERA5 recipe
(`../2026-08-12-aimip-1deg-6hourly/`, evaluated as AIMIP `v20260825`), plus a probe of
the existing stage-1 checkpoint.

The goal is to work out which of ACE2.2's ~7 differences from ACE2.1 are load-bearing.
Differences that turn out to be immaterial can be reverted, shortening the description
of the experiment; differences that matter stay, now justified by measurement.

## Runs

| | what | seeds |
|---|---|---|
| **P0** | inference probe of the *existing* ACE2.2 stage-1 checkpoint — no training | — |
| **P1** | ACE2.2 recipe with the ACE2.1 train/val split and out-of-sample checkpoint selection | 3 |
| **P2** | P1, plus near-surface fields reverted to secondary-decoder diagnostics | 1 |

P1 is a test, not a foregone conclusion: if the shorter, cooler 1979–2008 span costs
forced-response skill, the ACE2.2 split stays and the difference is documented rather
than removed. Its 3 seeds also supply the noise floor — ACE2.1 is a best-of-4-seed
model and ACE2.2 a single seed, so no existing comparison between them has an error bar.

P0 has run (2026-08-28) and its answer was negative: the stage-1 checkpoint is not a
usable stand-in for a trained model. It carries a *larger* uniform-SST response than the
final model, but its time-mean bias is far worse on every near-surface field and its
global-mean temperature is several times over-dispersed year to year. **Both variants
must be evaluated after all three stages** — there is no stage-1 shortcut for either.

P2 is judged against that spread. `TMP2m`/`Q2m`/`UGRD10m`/`VGRD10m` move out of the
prognostic set and into the stage-3 secondary decoder (27 → 31 names), matching ACE2.1.
Note the ace repo's ACE2S-ERA5 baseline instead makes these *primary* diagnostics; ACE2.1
is the comparison that matters here, hence the choice. Because they then only exist after
stage 3, P2 cannot be evaluated at stage 1.

## Differences from the 2026-08-12 experiment

Both variants, all three stages:

- **Train** 1979–2008 in 4 concat entries (was 1979–2013 in 6). 1994 returns to training.
  ERA5 production-stream stitch boundaries remain on subset boundaries — that is an
  ACE2.2 protocol choice ACE2.1 did not make, and it is kept deliberately.
- **Validate** 2009–2014 (was 1994 + 2014).
- **Checkpoint selection** by `5year_outsample`: 8 ICs through 2009, 7300 steps, ending
  2014-11-14 — out-of-sample, mirroring ACE2.1. Replaces `10year_insample`, whose 1995
  ICs sat inside the training window.
- **No inline-inference entry touches 2015–2024.** `10year` (2015 ICs) and
  `weather_2024` are dropped; `long_46year` becomes `long_36year`, ending in 2014.
- **`weather_2014` added** alongside `weather_1994`. Under the harmonized split 1994 is
  a training year, so the pair is an in-sample/out-of-sample weather comparison — a
  widening gap between them is a direct overfitting signal. Both are weight 0.

P2 only: the four near-surface names are removed from `in_names`/`out_names` and from
`corrector.force_positive_names` (`Q2m`), and added to stage 3's
`secondary_diagnostic_names`.

Unchanged: architecture, CRPS/energy-score loss, `global_mean_removal`, normalization
statistics, batch size, epoch counts, and the three-stage structure.

## Running

```bash
cd configs/experiments/2026-08-28-ace22-variants
bash run-train.sh                  # 4 stage-1 jobs on titan, 4 GPUs each
CLUSTER=jupiter bash run-train.sh  # the same 4 jobs on jupiter, 8 GPUs each
```

`CLUSTER` selects the target cluster and the rank count together, and they must stay
coupled. The configs' `batch_size: 8` is a *global* batch that fme divides among
data-parallel ranks, so it is identical on both clusters — only the per-GPU share
changes, 2 samples on titan and 1 on jupiter. The launcher refuses a rank count that
does not divide the config's batch sizes.

Both targets submit to the `ai2/ace` workspace at `high` priority with
`--min-runtime 8h`. The `--preemptible`/`--not-preemptible` flags are deprecated and,
since the 2026-08-28 scheduler change, priority no longer protects a run from
preemption — it only orders contention inside our own budget, hence `high` rather than
`urgent`.

Seeds come from the launcher via `--override seed=N`, as in ACE2.1's `run-ace-train.sh`;
the configs themselves all say `seed: 0`.

Each stage after the first needs a donor result dataset. The launcher passes it per
chain as a fourth argument to `run_training`, substituting it into the config's `# arg:`
header, so one config serves all three seeds; it refuses to submit if the placeholder
survives. Uncomment the block for the stage you are launching — the completed stages
stay commented so a rerun cannot resubmit them.

## Status

Stage 1 complete (2026-08-29), all four exit 0 at 40/40 epochs. `best_inference_error`:
rs2 0.04743, rs0 0.05786, rs1 0.06422 — a ±15% seed spread. P2's stage-1 figure is not
comparable, since it scores a smaller output set.

**All three P1 seeds continue through stages 2 and 3**, rather than the ACE2.1 protocol's
best-of-N single chain. The decision rules need a seed spread on the final metrics, and
the P0 probe showed stage-1 behaviour does not predict those, so a stage-1 spread cannot
stand in. The three chains additionally bound the selection advantage ACE2.1 gained from
best-of-4 seeds and ACE2.2, a single seed, did not: for a downstream metric that bonus is
at most about one standard deviation of the seed spread on that metric.

Comparing P1 and P2 at stage 1 is possible on the shared variables — P2's logged metrics
are a strict subset of P1's — but only as a diagnostic. At matched epoch, P2 is ~2%
*better* on single-step validation and ~2x worse on free-running rollout metrics,
concentrated in moisture and the TOA energy budget. Rollout stability is exactly what
stage 2 exists to fix, so this is a prediction to test after stage 2, not a verdict.
