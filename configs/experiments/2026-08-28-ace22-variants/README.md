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

Stages 2 and 3 are commented out in `run-train.sh`. They run only for the selected
stage-1 seed, and each config's `# arg:` header must first be filled with the donor
result dataset id — the launcher refuses to submit while the placeholder is present.
