# Stochastic-ACE bake-off, 6h-step subset

The daily-step bake-off (`../stochastic-ace-bakeoff/`, reports#51) left every
arm with a substantial small-scale precipitation power deficit; these arms
rerun the protocol at a 6h step to test whether the deficit is a
daily-timestep artifact. Six arms: the daily bake-off's arms 1, 2, 3, 6, and 7
unchanged, plus a new arm 9 = arm 7 + γ=0.5 whitening — the cell the daily
bake-off left untested (whitening applied to the spectral-power term alone;
with `energy_score_weight: 0` the shared whitening operator touches only that
term). Runs are launched by Jeremy via `run-train.sh` — this directory only
defines the configs.

## Arms

Same knob table as the daily bake-off (weights `crps` / `es` / `sp`):

| config | crps | es | sp | total-energy corrector | whitening |
|---|---|---|---|---|---|
| `arm1-90-10-ec.yaml` (base) | 0.9 | 0.1 | 0 | `constant_temperature` | none |
| `arm2-90-10-noec.yaml` | 0.9 | 0.1 | 0 | off | none |
| `arm3-50-50-ec.yaml` | 0.5 | 0.5 | 0 | `constant_temperature` | none |
| `arm6-80-10-sp10-ec.yaml` | 0.8 | 0.1 | 0.1 | `constant_temperature` | none |
| `arm7-90-0-sp10-ec.yaml` | 0.9 | 0.0 | 0.1 | `constant_temperature` | none |
| `arm9-90-0-sp10-ec-whiten-g0.5.yaml` | 0.9 | 0.0 | 0.1 | `constant_temperature` | `per_sample`, γ=0.5 |

## What changes from the daily configs (and nothing else)

Each config is its daily counterpart with the mechanical edits below plus one
recipe change — `global_mean_removal` removed (arm 9 additionally adds the
`energy_score_whitening` block, in arm 8's syntax):

- **Data**: the daily wrapper directory
  `2026-03-19-era5-1deg-8layer-daily-1940-2025.zarr/` → `data_path:
  /climate-default/`, with the unchanged `file_pattern` selecting the 6-hourly
  store `2026-03-19-era5-1deg-8layer-1940-2025.zarr` directly (the 6-hourly
  store has no wrapper directory; this is the form the prior 6h stochastic
  runs trained with). Same variables, same ACE2 train/val/inference split,
  same 06Z IC timestamps.
- **Normalization**: the daily 1990–2019 stats → the 6-hourly 1990–2019 stats
  (beaker dataset `andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019`,
  mounted at `/statsdata` via the config's `# arg:` header). Residual scaling
  is timestep-dependent, so the daily stats cannot be reused.
- **Horizons**: inference step counts ×4 at fixed lead time — 10-year runs
  3652 → 14608, the ACE2-comparable 5-year 1826 → 7304, the 5-day weather
  evals 5 → 20 steps, and the day-5 `step_means`/`ensembles` step index
  5 → 20. `forward_steps_in_memory` is a memory/IO chunk size, not a horizon,
  and is unchanged everywhere.
- **No global-mean removal**: the daily arms' `global_mean_removal` block
  (`kind: shared`, `append_as_input: true`) is dropped — it was not meant to
  be part of this recipe (Jeremy, PR review 2026-07-29).

Held fixed deliberately (the timestep and the global-mean-removal drop above
are the only differences vs the daily arms): model (`NoiseConditionedSFNO`, fg16/sr0.125), 1-step
training, `seed: 0`, `max_epochs: 80`, batch size, LR, EMA decay 0.999, and
the inline-inference cadence (every 2 epochs). Consequences to be aware of at
review: 80 epochs at 6h is ~4× the optimizer steps (and ~4× the wall time) of
a daily arm, per-epoch EMA/LR-schedule shapes differ in step terms, and the
per-inference cost is also ~4×.
