# Handoff: fix the enso_coefficient cross-rank deadlock

**Branch**: `troya/fix-enso-coefficient-rank-deadlock` (currently just this
note on `origin/main`). **Goal**: implement the fix, test it, and prepare a
PR to ai2cm/ace main. **Remove this file before the PR is marked ready.**

## The bug

`EnsoCoefficientEvaluatorAggregator._get_coefficients()`
(`fme/ace/aggregator/inference/enso/enso_coefficient.py`, lines 236-243 at
`origin/main` = 986948a12) enters the cross-rank reduction only when the
rank has local data:

```python
if target_coefficients_all:
    reduced_target_coefficients = reduce_data(dist, target_coefficients_all)
...
if gen_coefficients_all:
    reduced_gen_coefficients = reduce_data(dist, gen_coefficients_all)
```

`reduce_data` calls `dist.reduce_mean` — a collective. Whether a rank has
data is **data-dependent**: each sample gets an index series only if its
inference window overlaps the hardcoded observed Nino3.4 index (CMIP6 AMIP,
1940-01 through 2021-01, `historical_index.py`) by at least
`OVERLAP_THRESHOLD = 0.9` (`get_sample_index_series`). When some ranks'
samples clear the threshold and others' don't, the active ranks block in
the all-reduce waiting for ranks that never call it, until the NCCL
collective timeout kills the job.

## How it manifested (real incident, 2026-09-09)

A 4-GPU ocean fine-tune on a real-world-dated dataset ran inline inference
from four initial conditions (Jan 2012/2013/2014/2015, ~8.6-year windows),
one sample per rank. Window overlap with the 1940-2021 index: ~100%, ~93%,
~81%, ~70% — so exactly two ranks were active. Epoch-1 summaries completed
in seconds; the two active ranks then sat in `reduce_mean` for the full
collective timeout (120 minutes) and the job died at the next barrier
(inside `wandb.log`), with a traceback pointing nowhere near this
aggregator. Beaker experiment `troya/samudra-enso-w1-hybridufsft-fabc`.

Why it was never seen before: on simulation-calendar datasets (e.g. model
years 02xx) **no** sample overlaps 1940-2021, every rank is uniformly
inert, and no rank enters the collective. Only runs with dates that
*partially* straddle the index's endpoints (any real-world run extending
past early 2021, or starting before 1940) can split the ranks.

## Required behavior after the fix

1. No deadlock for any mix of active/inert ranks (0%, partial, 100%).
2. Coefficients averaged only over ranks/samples that actually have them —
   inert ranks must not bias the mean toward zero.
3. When no rank has data, `get_logs` returns `{}` on every rank (current
   inert-everywhere behavior preserved).
4. Single-process behavior unchanged.

## Suggested design (adapt as you see fit)

Make participation unconditional and weight the reduction:

- In `record_batch`, track variable names and spatial shape on every rank
  regardless of active status (the batch's data dict is identical across
  ranks), so an inert rank can construct zero tensors with the right keys
  and shapes.
- In `_get_coefficients`, every rank always calls the collective(s):
  reduce `sum(coefficients * w_local)` and `sum(w_local)` where `w_local`
  is the rank's active-sample count (or 0), then divide on the root; if the
  reduced weight is zero, return `None`/`{}` as today. Two unconditional
  `reduce_sum`-style collectives (or one on a stacked tensor with the
  weight appended) replace the two conditional `reduce_mean`s. Note
  `reduce_data` currently does an unweighted `reduce_mean` across ranks
  even when ranks hold different numbers of active samples — the weighted
  form also fixes that mild pre-existing bias.
- Keep collective ORDER and COUNT identical on all ranks (target first,
  then gen — or one fused call).

## Testing

- Extend `fme/ace/aggregator/inference/enso/test_enso_coefficient.py`.
- The deadlock itself can't run under pytest, but the participation logic
  can: monkeypatch the `Distributed` singleton's reduce methods with a fake
  that RECORDS how many times each rank-side code path would invoke a
  collective, and assert the invocation sequence is identical for an
  aggregator whose samples are all-None vs all-active (build two
  aggregators with initial times inside vs outside 1940-2021 and identical
  `n_forward_timesteps`). A precedent for faking the distributed layer
  lives in `test_dynamic_index.py` (a gathered-tensor monkeypatch test).
- Also assert the weighted mean: two "ranks" simulated by two aggregator
  instances (or direct calls to the new reduction helper) with different
  active counts produce the correctly weighted result.
- Run the whole enso test dir plus `fme/ace/aggregator/inference` tests.

## Mechanics

- Conda env `fme`; run pytest with
  `/home/troya/miniconda3/envs/fme/bin/python -m pytest ...`.
- Pre-commit hooks (ruff, ruff-format, mypy) REFORMAT files and silently
  ABORT the commit — after every `git commit`, verify with
  `git log --oneline -1` and `git status --short`, and re-add/recommit if
  a hook modified files.
- Draft a PR description (bug, incident, fix, tests); do NOT open the PR —
  the branch owner does. End the description with:

🤖 Generated with [Claude Code](https://claude.com/claude-code)

- Delete HANDOFF.md in the final commit.
