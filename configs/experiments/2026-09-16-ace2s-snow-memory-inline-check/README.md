# ACE2S snow-memory inline check

Evaluator runs that exercise the new `anomaly_memory` inference aggregator on the checkpoints the
offline snow-memory prototype scored, so the inline numbers can be compared against the prototype
before the aggregator is used in training runs.

The prototype (`explore2/brianh/2026-06-09-land-atm-coupling/snow-memory-prototype`) established
that prognostic snow restores the anomaly memory of upward shortwave that the snow-free control
lacks, and that per-cell accumulation reduced to regional scalars is the right shape for an inline
metric. This round is the acceptance test of that metric's implementation: same checkpoints, same
initial conditions, same rollout length as the prototype's saved rollouts.

## Arms

| arm | checkpoint dataset | best-inference epoch |
|---|---|---|
| cm4-control | `01KZC1J3R3EW9YVM6HPNSNNNCY` | 43 |
| cm4-masked-naive | `01KZVBJZ8KHR9E84CEF0NF95ES` | 46 |
| era5-control | `01KYX6AQTSXD3N23HP128TJYTC` | 34 |
| era5-masked-naive | `01KZVBA39HPP7ZNZ8FXD2HG9DR` | 39 |
| cm4-masked-log1p | `01KZVBJZJ1E81EAF5BQJEVN8AH` | 48 |
| era5-masked-log1p | `01KZWD52T8QEQW3V08F9ENBE06` | 30 |

The masked checkpoints store the per-field `transforms` stepper field that exists only on
`exp/ace2s-snow-prognostic-daily`, so this directory lives on a side branch of that branch
(`exp/ace2s-snow-memory-inline-check`) with the aggregator commits cherry-picked onto it. The two
control arms were run once from `feature/anomaly-memory-aggregator` (main-tip code, commit
9571f3580) before the treatment loads failed there; those runs are kept, and only the four masked
arms were launched from this branch (`./run-ace-evaluator.sh masked`). The aggregator code is
identical on both branches.

## Initial conditions

Four ICs per arm, 1825 forward steps (~5 yr), as in the prototype rollouts: CM4 0311, 0321, 0331,
0341 (Jan 1, all held out); ERA5 1998, 2003, 2008, 2013 (Jan 1).

## Metric settings

`anomaly_memory` on `USWRFsfc`, `TMP2m`, `surface_temperature` and, where the arm predicts them,
`surface_snow_amount_masked` and `surface_snow_area_fraction_masked`. Lags 0, 1, 3, 7, 14 and
30 days; scalars logged at lags 7 and 14; maps at lag 7. Leading times are restricted to the
extended cold season per hemisphere (Nov-May north, May-Nov south). Regions are the seven snow
boxes of the prototype plus one box covering everything north of 40S (`north_of_40S`). Region names use underscores because
they become W&B key segments.

Outputs: W&B scalars `anomaly_memory/{prediction,target,gap}/<var>-<region>-lag<L>` and
`anomaly_memory/variance_ratio/<var>-<region>`, one target/prediction map image per variable, and
`anomaly_memory_diagnostics.nc` with the full per-cell curves under `experiment_dir`.

## Acceptance

Regional lag-7 correlations for `USWRFsfc` and snow amount within about 0.02 of the prototype's
pooled-climatology reference, the control-to-treatment gap reproduced, and the lag-7 map matching
the prototype's map in shape. The reference numbers are produced by
`snow-memory-prototype/inline_reference.py`, which recomputes the prototype statistics with the
aggregator's exact settings (pooled climatology, these months, raw upward shortwave).

## Launch

```bash
./run-ace-evaluator.sh                    # all six, ai2/jupiter
CLUSTER=ai2/titan ./run-ace-evaluator.sh  # elsewhere
./run-ace-evaluator.sh cm4                # substring filter on the job name
```

Each invocation validates its config with `fme.ace.validate_config` before submitting. Writing
to GCS requires `log_to_file: false`.
