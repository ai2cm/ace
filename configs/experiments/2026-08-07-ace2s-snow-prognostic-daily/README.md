# ACE2S snow-prognostic training, 1-deg daily

Gives ACE2S **prognostic snow** at 1-deg daily cadence: snow water equivalent
(`surface_snow_amount`) becomes a predicted, fed-back state variable and snow-covered area
(`surface_snow_area_fraction`) a predicted diagnostic. This is the daily counterpart to the
1-deg 6-hourly runs on `exp/ace2s-snow-prognostic-train`; 6-hourly training proved too slow and
4-deg too coarse to resolve the land/snow signal.

A variable is prognostic iff it is in both `in_names` and `out_names`, diagnostic iff it is in
`out_names` only (`fme/core/step/single_module.py`). Adding an input channel changes the encoder
input dims, so this model trains from scratch. Both snow fields are added to
`corrector.force_positive_names` since neither can be negative.

## Control and comparability

The control is the CM4-piControl daily 1-step pretrain on `exp/ace2s-cm4-piControl-train`
(`configs/experiments/2026-06-05-ace2s-cm4-picontrol/ace-train-config-1-step-pretrain-daily.yaml`,
W&B group `ace2s-cm4-picontrol-daily`). The treatment config here is a copy of it, and the diff
against it is only:

- `surface_snow_amount` added to `in_names`
- `surface_snow_amount` and `surface_snow_area_fraction` added to `out_names`
- both snow names added to `corrector.force_positive_names`
- `save_per_epoch_diagnostics: true` (writes netCDF diagnostics; does not affect training)

Everything else — stats mount, dataset paths, IC timestamps, `max_epochs`, the 20-year inference
rollout, aggregator, optimizer, batch sizes, EMA, SFNO builder, `clip_frozen_precipitation` — is
byte-identical, so the two runs' W&B keys line up directly.

The branch base is `main` at `355d51757` plus a merge of `feature/one-step-r2-metric` (the
`skill_map` one-step R²/RMSE aggregator, still unmerged on main). That is the same code and the
same `latest_deps_only_image.txt` the control is training on. Later main commits were deliberately
not merged: #1402 bumps to python 3.12 / torch 2.10 and changes the image, which would put the
treatment on a different torch than the control.

## Short-lead skill is evaluated offline, not inline

The 6-hourly configs carried a second inline `inference-short-lead` entry. It is not used here.
Measured on the 6-hourly CM4 pretrain, it cost ~2% of epoch wall clock — cheap, but the daily
control has no such entry, so an inline probe would be unpaired and unusable for the cross-arm
comparison it exists for. Run `fme.ace.evaluator` against both arms' `best_inference_ckpt.tar`
after training instead: it costs no training time, works against the control as it already is, and
with `enable_time_series=True` yields the continuous lead-time curve inline inference cannot log.

## ERA5 arm

Pending data. No ERA5 daily store contains snow — the daily coarsening config's `snapshot_names`
omits it. PR #1388 adds `surface_snow_amount` / `surface_snow_area_fraction` (and
`soil_temperature_0..3`) and repoints the output at a new dated store superseding the 07-24 one.
The ERA5 config is authored once that store, its stats, and its weka copy are confirmed.

## Inputs

| | value |
|---|---|
| CM4 train/inference data | `/climate-default/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily.zarr` (weka) |
| CM4 stats | beaker `brianhenn/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-stats` → `/statsdata` |

The stats dataset is mounted via the config's `# arg:` header, which the launcher extracts. All
three stats files carry both snow variables, including `surface_snow_amount` in
`scaling-residual.nc` (required — it is prognostic, so it is residual-scaled).

## Launch

```bash
./run-ace-train.sh                    # ai2/jupiter, 8 GPUs
CLUSTER=ai2/titan ./run-ace-train.sh  # ai2/titan, 4 GPUs
```

`train_loader.batch_size` is a global batch split across ranks, so the GPU count sets per-rank
activation memory without changing the math: 4 ranks fit titan's 180 GiB B200s, jupiter's 80 GiB
H100s need 8. That equivalence relies on no channel being sample-masked, since `LossOutput._reduce`
normalizes a masked channel per rank — the CM4 snow fields are NaN-free, so it holds.
