# Datasets still to produce for the slab-ocean (SOM) experiments

`generate_som_configs.py` reproduces the ACE experiments of
[ai2cm/ace2s-shield-plus-paper](https://github.com/ai2cm/ace2s-shield-plus-paper/tree/main/ACE-experiments/inference)
on the 4deg daily SHiELD-SOM data. Three of the paper's inputs have no 4deg
daily counterpart yet. Their configs are generated against placeholder paths
(`/climate-default/TBD-...`) so the machinery is complete; `submit_som_jobs.py`
refuses the dependent kinds until the placeholder is replaced.

To bring one online: produce the dataset, put its real name into
`MISSING_DATASETS` in `generate_som_configs.py`, regenerate, commit, submit.

## D1 — daily 4deg abrupt-CO2 runs

- Placeholder: `/climate-default/TBD-vertically-resolved-4deg-daily-c96-shield-som-abrupt-co2-increase-fme-dataset/abrupt-{2x,3x,4x}CO2.zarr`
- Needed by: `abrupt-10yr-eval` (model vs SHiELD's abrupt run, initialized
  from its 2020-01-01 state), `abrupt-data-only` (that run scored against
  itself). Until then `abrupt-10yr` runs the same experiment as free inference
  with no reference.
- Source: 6-hourly 4deg already processed —
  `gs://vcm-ml-intermediate/2024-08-14-vertically-resolved-4deg-c96-shield-som-abrupt-co2-increase-fme-dataset/`
  (14612 steps from 2020-01-01T06; has every model variable plus
  `prescribed_qflux` / `prescribed_mixed_layer_depth`).
- How: add a daily `time_coarsen` block to
  `scripts/data_process/configs/shield-som-abrupt-co2-increase-c96-4deg-8layer.yaml`
  mirroring `shield-som-ensemble-c96-4deg-8layer.yaml`. Cheapest of the three.

## D2 — daily 4deg SOM spin-up year

- Placeholder: `/climate-default/TBD-vertically-resolved-4deg-daily-c96-shield-som-ensemble-spin-up-fme-dataset/{climate}-spin-up-ic_000N.zarr`
  (1x/2x/4xCO2 `ic_0005`, 3xCO2 `ic_0002` — the members the paper uses).
- Needed by: `eq` — the paper's spin-up stage starts from the 2030-01-01
  state of this dataset and runs one year before the scored ten-year main run.
  `eq-nospinup` covers the experiment without it, starting from the member's
  2031 state.
- Source: raw 45x90 regrids exist —
  `gs://vcm-ml-raw-flexible-retention/2024-07-03-C96-SHiELD-SOM/regridded-zarrs/gaussian_grid_45_by_90/{1x,2x,3x,4x}CO2-spin-up-ic_000N`.
- How: clone `scripts/data_process/configs/shield-som-spin-up-c96-1deg-8layer.yaml`
  to 4deg (gaussian_grid_45_by_90 inputs, daily `time_coarsen`). The configs
  assume the daily time labels sit at 06Z like the main dataset's.

## D3 — daily 4deg abrupt-4xCO2 36-member ensemble

- Placeholder: `/climate-default/TBD-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset/abrupt4xCO2-ic_00NN.zarr`
- Needed by: `abrupt-ens-data-only` — SHiELD's own 90-day abrupt-4xCO2
  spread, the target for the model's `abrupt-ens` runs.
- Source: 1deg only —
  `gs://vcm-ml-raw-flexible-retention/2025-02-03-C96-SHiELD-SOM-abrupt-4xCO2-ensemble/regridded-zarrs/gaussian_grid_180_by_360/abrupt-4xCO2-ic_00NN`.
  No 45x90 regrid exists.
- How: regrid to gaussian_grid_45_by_90 (as for the 2024-07-03 SOM runs), then
  clone `scripts/data_process/configs/shield-som-abrupt4xCO2-ensemble-c96-1deg-8layer.yaml`
  to 4deg with a daily `time_coarsen`. Heaviest of the three.

## Not missing

- 1000-year forcing: the 1xCO2 member tiled via `n_repeats` (all SOM forcing
  but CO2 is climatological), as the paper does with its Hugging Face forcing.
- Everything for `eq-nospinup`, `eq-1000yr`, `data-only`, `abrupt-10yr`,
  `abrupt-ens`, `7day`: the 2026-06-08 4deg daily SOM ensemble.
