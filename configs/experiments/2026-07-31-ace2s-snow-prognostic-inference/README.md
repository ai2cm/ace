# ACE2S snow-prognostic inference: standard CM4 40-year evaluation

Offline evaluation of the CM4 arm of the snow-prognostic experiment
(`configs/experiments/2026-07-14-ace2s-snow-prognostic-training/`), using the standard CM4
evaluator recipe: one initial condition, 58,300 forward steps (~39.9 yr at 6h) over the holdout
period, with the full offline diagnostic suite (time-mean and zonal-mean maps, power spectra,
ENSO/IPO indices, histograms). Training only ever exercised the *inline* 20-year rollout over the
pre-validation years, so this is the first long rollout over held-out time.

The ERA5 arm was stopped early — its training was going less well than CM4's — so only CM4 is
evaluated here.

## Run

```bash
./run-inference.sh
```

One single-GPU beaker job, `ace2s-snowprog-cm4-1deg-6h-evaluator`, logging to W&B project `ace`,
group `ace2s-snow-prognostic`, job type `inference`. Prior 58,300-step CM4 evaluations at
`forward_steps_in_memory: 50` took 1-7 h on one GPU.

## Inputs

| | value |
|---|---|
| checkpoint | beaker `01KXPJ21JHMQ7YHJ0GTZZ65APC:training_checkpoints/best_inference_ckpt.tar` |
| checkpoint source | beaker experiment `01KXPE9HE4BC5M7D2DGFRME5JF` (`ace2s-snowprog-cm4-1deg-6h-pretrain`) |
| data | `/climate-default/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr` (Weka) |
| initial condition | `0311-01-01T06:00:00` (index 233,600; the validation/holdout boundary) |

The CM4 split is train < `0306`, validation `0306`-`0311`, holdout `0311`-`0351`, so the rollout
covers 233,600 -> 291,900 of the zarr's 292,000 steps — essentially the whole holdout.

`cm4.yaml` is checkpoint-agnostic and needs no snow-specific settings: `surface_snow_amount` is
prognostic, so the loader supplies it as part of the initial condition, and the output-only
`surface_snow_area_fraction` reference comes from the same store. Note this uses the land zarr
alone — not the `merge:` with the interpolated-SST atmosphere zarr that
`configs/baselines/cm4-piControl/uncoupled-atmos/evaluator-config.yaml` adds — because training
and the inline rollouts used the land zarr alone, and eval forcing must match training.

Only the 1-step pretrain exists; the multi-step finetune stage of the recipe was never launched.

## Interpreting the results

**There is no control 40-year evaluation to compare against.** The `ace2s-cm4-picontrol` W&B group
contains training runs only, so nothing here is a control-relative result.

Any control 40-year eval belongs on the control training branch,
`exp/ace2s-cm4-piControl-train` — `configs/experiments/2026-06-05-ace2s-cm4-picontrol/run-ace-evaluator.sh`
already has the seed loop for it. Not to be launched from this directory. Two candidate baseline
checkpoints, for reference:

| baseline | beaker dataset | note |
|---|---|---|
| control 1-step pretrain | `01KTPTS6C23P8SWB9RBFWB09BE` | stage-matched, but stopped at epoch 11 |
| deployed finetuned control (rs0) | `01KTYXNSJX90Y5E2CQ6SV8K37D` | production ACE2S CM4, includes multi-step finetune |

Comparison across arms happens in W&B on the shared metric keys, so the runs need matching
`n_forward_steps`/IC/aggregator settings — not co-located configs.

Expect noticeable climate drift. The training run's inline 20-year
`inference/time_mean/rmse/PRESsfc` bottoms out near 200 Pa around epoch 26, degrades to
1100-1400 Pa over epochs 31-39, and ends at 544 Pa at epoch 50, whereas the control pretrain was
at 66 Pa by epoch 11 and still falling. `best_inference_ckpt.tar` (written 2026-07-22, mid-run,
~epoch 27) is selected on the inline long-rollout metric, so it captures the best-climate epoch —
but the drift is still substantially larger than a control's.

Focus on the snow channels (`surface_snow_amount`, `surface_snow_area_fraction`: time-mean bias
maps, zonal-mean error Hovmollers, histograms) to see whether prognostic snow stays bounded over
40 years, and on near-surface T/Q plus the surface energy fluxes for regressions attributable to
the added prognostic state.
