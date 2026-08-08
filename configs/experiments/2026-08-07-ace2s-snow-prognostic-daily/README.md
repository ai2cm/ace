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

Two models, one config each, both compared against the corresponding daily control:

- **CM4-piControl** — control is the daily 1-step pretrain on `exp/ace2s-cm4-piControl-train`
  (`configs/experiments/2026-06-05-ace2s-cm4-picontrol/ace-train-config-1-step-pretrain-daily.yaml`,
  W&B group `ace2s-cm4-picontrol-daily`).
- **ERA5** — control is the daily 1-step pretrain on `config/ace2s-era5-daily-baseline`
  (`configs/baselines/era5/ace-train-config-1-step-pretrain-daily.yaml`, PR #1400).

Each treatment config is a copy of its control, and the diff is only:

- `surface_snow_amount` added to `in_names`
- `surface_snow_amount` and `surface_snow_area_fraction` added to `out_names`
- both snow names added to `corrector.force_positive_names`
- `save_per_epoch_diagnostics: true` (writes netCDF diagnostics; does not affect training)

Everything else — IC timestamps, `max_epochs`, the long inference rollout, aggregator, optimizer,
batch sizes, EMA, SFNO builder, `clip_frozen_precipitation` — is byte-identical, so the runs' W&B
keys line up directly. `surface_snow_thickness` exists in the ERA5 daily store but is deliberately
not used by either model, so the two arms stay comparable.

The ERA5 treatment additionally reads a **different store than its control**: the control trained on
`2026-07-24-…`, which contains no snow, so the treatment uses the `2026-08-07-…` store created to
add it. This was verified not to affect comparability — the two stores' time coordinates are
identical, shared variables are bitwise equal, and all 87 shared variables in all four
`combined/*.nc` stats files are bitwise identical, with only the 16 new land fields added. The
existing ERA5 control therefore remains a valid baseline.

One asymmetry to expect on ERA5: its control run completed 2026-07-31 on a branch predating the
`skill_map` aggregator, so it logs no skill-map panels. The ERA5 treatment will log them one-sided.
CM4 is unaffected. Cross-arm comparison rests on per-channel validation loss and the long inference,
which every run has.

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

## Inputs

| | value |
|---|---|
| CM4 train/inference data | `/climate-default/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily.zarr` (weka) |
| CM4 stats | beaker `brianhenn/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-stats` |
| ERA5 train/inference data | `/climate-default/2026-08-07-era5-1deg-8layer-daily-1940-2025.zarr` (weka) |
| ERA5 stats | beaker `andrep/2026-08-07-era5-1deg-8layer-daily-stats-1990-2019` |

Stats datasets are mounted via each config's `# arg:` header, which the launcher extracts. For both
models all three stats files carry both snow variables, including `surface_snow_amount` in
`scaling-residual.nc` (required — it is prognostic, so it is residual-scaled).

The ERA5 store is copied to weka from
`gs://vcm-ml-intermediate/2026-08-07-era5-1deg-8layer-daily-1940-2025/2026-08-07-era5-1deg-8layer-daily-1940-2025.zarr`
with `scripts/data_process/gcs_to_weka.sh`, keeping the same name. Until that copy exists the ERA5
job fails at startup: `XarrayDataConfig` globs `data_path`/`file_pattern` and hard-errors on an
empty match.

The coarsening run originally wrote that zarr as `2026-03-19-era5-1deg-8layer-1940-2025.zarr` — the
output is named after the `runs` key, which must stay the 6-hourly source since the same key
resolves the coarsening input — and it was renamed by hand afterwards, as the 07-24 store was. So a
re-run of `make era5_1deg_daily_dataset` reproduces the confusing name and needs the rename again.

## Launch

```bash
./run-ace-train.sh                    # ai2/jupiter, 8 GPUs; both models
CLUSTER=ai2/titan ./run-ace-train.sh  # ai2/titan, 4 GPUs
```

Comment out either `run_training` call to launch one model. `train_loader.batch_size` is a global
batch split across ranks, so the GPU count sets per-rank activation memory without changing the
math: 4 ranks fit titan's 180 GiB B200s, jupiter's 80 GiB H100s need 8. Both configs use
`batch_size: 8`, so the profile applies to each. That equivalence relies on no channel being
sample-masked, since `LossOutput._reduce` normalizes a masked channel per rank — the snow fields are
NaN-free in both stores, so it holds.
