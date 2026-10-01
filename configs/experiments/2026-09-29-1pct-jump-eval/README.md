# 1pctCO2 per-step evaluation around the sea-ice drop

Every 4deg salinity-corrector run evaluated with
`../2026-09-25-1pct-yearly-eval` (and the no-corrector control) holds
3.4-4.3e12 m^3 more sea ice than the target through years 2-40 and sheds the
excess within about a year: between 0041 and 0042 in the corrected runs,
mostly over 0038-0040 in the no-corrector run. The deep salinity bias steps at
the same time, and the uncorrected run's largest monthly salt-content jumps
cluster in years 39-41. This config writes every 5-day step for 15 years
around the drop, so the order of events (ice, salinity, temperature, the
forcing) can be read off directly instead of from monthly means.

| config | status tag | initial condition | steps | output |
| --- | --- | --- | --- | --- |
| `evaluator-config-1pct_jump_4deg.yaml` | `run_1pct_jump_4deg` | 0034-01-01, from the target | 1095 (to 0049-01-01) | every step, all variables, predictions and target (the target file carries the forcing); corrector deltas every step; monthly means. About 5 GB. |

Starting from the target, the run begins without the long rollouts' ice
excess. If it still shows a drop or a salinity step around 0040-0042, the
inputs drive it; if it does not, the drop belongs to the state the long
rollouts reached by then (and the 139-year runs' monthly output is what shows
it).

## Running

Same mechanism as the yearly eval (`job_runner/evaluate.sh --config-dir`):
append a row with the status tag to the training experiment's
`experiments.txt`, then

```bash
bash job_runner/evaluate.sh configs/experiments/2026-09-28-test-no-sal-correct . \
    --config-dir configs/experiments/2026-09-29-1pct-jump-eval
```

Row for the no-corrector control, the cleanest case (no corrector to confound
the salinity step):

```
cm4_1pct_46to125_piC_156to235-4deg-ocean_no_sal_correct|rs0|01M3M32Q30ZKAN66PA2F58QMRJ|run_1pct_jump_4deg|best_inference_ckpt|normal|--min-runtime 2h
```

The salinity-corrector runs, and the control checkpoint with the salt
correction added at evaluation, need the salt corrector and are evaluated from
`exp/2026-09-30-salt-corrector-pr1533`.
