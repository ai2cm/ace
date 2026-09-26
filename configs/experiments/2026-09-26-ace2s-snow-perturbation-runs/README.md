# ACE2S perturbed-snow sensitivity runs, CM4, 1-deg daily

Does the fine-tuned CM4 snow treatment (`2026-09-17-ace2s-snow-inline-metrics-train`,
cm4-masked-naive fine-tune, best-inference epoch 13, dataset `01M38TTH492G0WT71YFAEH0V58`) use its
snow input non-physically for remote variables? Six 30-day inference runs from custom
initial-condition files that share everything but the snow amount, eight members each on 15 January
of holdout years 0311, 0315, ..., 0339 (the holdout-rollout years). The response of each case is
its difference from the control at matched dates and lead times.

| case | perturbation of `surface_snow_amount_masked` in the initial condition |
|---|---|
| control | none |
| null | none; 0.1 K white noise on land skin temperature instead (the noise floor for "remote") |
| siberia-removed | zero in the Siberia box (50-68N, 60-140E) |
| plains-removed | zero in the Great Plains box (32-49N, 255-265E) |
| plains-plus50 | +50 mm in the Great Plains box (bare to snow-covered) |
| nh-removed | zero on every snow-mask cell north of 20N |

Box edges are tapered with a 5-degree cosine ramp so the spectral model does not ring on a step.
Snow cover and albedo are model outputs and follow the snow amount. The files are built by
`build_initial_conditions.py` in the explore2 round `2026-09-26-snow-perturbation-runs` and stored
as the Beaker dataset `brianhenn/2026-09-26-ace2s-snow-perturbation-ics-v2`, mounted at `/ics`.

A physical response is confined to the box and its downstream neighbourhood at day 1, spreads over
days, and never exceeds the noise floor in the other hemisphere within 30 days. A remote response
above the noise floor at day 1, or remote growth not preceded by local and near-field growth, is
the non-physical signature. The analysis lives in the explore2 round.

## Output

`gs://vcm-ml-intermediate/2026-09-26-ace2s-snow-perturbation-runs/<case>/`: full daily prediction
fields for all eight members and 30 lead times, as `predictions.zarr` (netCDF prediction files
cannot be written to GCS).

## Launch

```
cd configs/experiments/2026-09-26-ace2s-snow-perturbation-runs
./run-ace-inference.sh            # all six; W&B group ace2s-snow-inline-metrics
./run-ace-inference.sh plains     # subset by substring
```
