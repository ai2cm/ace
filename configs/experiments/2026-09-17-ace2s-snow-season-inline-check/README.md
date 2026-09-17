# ACE2S snow-season inline check

Evaluator runs that exercise the new `snow_season` inference aggregator on the four masked-snow
checkpoints the snow-memory prototype scored, so its inline numbers can be compared against the
water-year traces in the prototype report before the aggregator is used in training runs.

The prototype's water-year addendum (`explore2/brianh/2026-06-09-land-atm-coupling/snow-memory-prototype`,
`phenology.py`) showed that region-mean SWE traces through the water year expose snowpack defects
that neither time-mean bias nor anomaly memory ranks. This round is the acceptance test of the
inline implementation: same checkpoints, initial conditions and rollout length as the prototype's
saved rollouts. The controls are not run; they have no snow channel.

## Arms

| arm | checkpoint dataset | best-inference epoch |
|---|---|---|
| cm4-masked-naive | `01KZVBJZ8KHR9E84CEF0NF95ES` | 46 |
| cm4-masked-log1p | `01KZVBJZJ1E81EAF5BQJEVN8AH` | 48 |
| era5-masked-naive | `01KZVBA39HPP7ZNZ8FXD2HG9DR` | 39 |
| era5-masked-log1p | `01KZWD52T8QEQW3V08F9ENBE06` | 30 |

These checkpoints store the per-field `transforms` stepper field that exists only on
`exp/ace2s-snow-prognostic-daily`, so the configs live on this side branch of it with the
aggregator commits cherry-picked on. The arms are test material for the metric, not preferred
configurations.

## Initial conditions

Four per arm, 1825 forward steps (~5 yr), as in the prototype rollouts: CM4 0311, 0321, 0331,
0341 (Jan 1); ERA5 1998, 2003, 2008, 2013 (Jan 1). Each gives four complete water years.

## Metric settings

`snow_season` on `surface_snow_amount_masked` over the seven snow boxes of the prototype, with the
water year starting 1 October for northern boxes and 1 April for the Andes. Outputs: W&B scalars
`snow_season/{prediction,target,gap}/<region>/<quantity>` for peak, midwinter mean, melt-out day
and summer floor, one `snow_season/traces` figure of every water year for target and prediction,
and `snow_season_diagnostics.nc` with the full traces under `experiment_dir`
(`gs://vcm-ml-intermediate/2026-09-17-ace2s-snow-season-inline-check/<arm>`).

## Acceptance

Target-side scalars equal the offline values in the prototype's `snow_season_reference.md`, which
recomputes them from the saved rollouts with the aggregator's own phenology functions and the
same full-box area weighting; prediction-side values within the interannual spread; the traces
figure matches the report's regional water-year figure panel by panel.

## Launch

```bash
./run-ace-evaluator.sh                    # all four, ai2/jupiter
CLUSTER=ai2/titan ./run-ace-evaluator.sh  # elsewhere
./run-ace-evaluator.sh cm4                # substring filter on the job name
```

Each invocation validates its config with `fme.ace.validate_config` before submitting.
