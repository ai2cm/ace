# Datasets still to produce for the slab-ocean (SOM) experiments

`generate_paper_configs.py` reproduces the ACE experiments of
[ai2cm/ace2s-shield-plus-paper](https://github.com/ai2cm/ace2s-shield-plus-paper/tree/main/ACE-experiments/inference)
on the 4deg daily SHiELD-SOM data. Three of the paper's inputs have no 4deg
daily counterpart yet. Their configs are generated anyway so the machinery is
complete — against the real name once a processing config fixes it, a
placeholder (`/climate-default/TBD-...`) before that — and `submit_paper_jobs.py`
refuses the dependent kinds until the entry is marked `available` in
`MISSING_DATASETS` in `generate_paper_configs.py`.

To bring one online: produce the dataset, copy it to weka, set the real name
and `available=True` in `MISSING_DATASETS`, regenerate, commit, submit.

## D1 — daily 4deg abrupt-CO2 runs — IN PROGRESS

- Path (fixed): `/climate-default/2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-abrupt-co2-increase-fme-dataset/abrupt-{2x,3x,4x}CO2.zarr`
- Needed by: `abrupt-10yr-eval` (model with slab ocean vs SHiELD's abrupt
  run, initialized from its 2020-01-01 state), `abrupt-10yr-eval-sst` (same,
  but SST and sea ice prescribed from SHiELD's run — the atmospheric response
  given SHiELD's own surface warming), `abrupt-data-only` (the run scored
  against itself). Until then `abrupt-10yr` runs the slab experiment as free
  inference with no reference.
- Source: raw 45x90 regrids
  `gs://vcm-ml-raw-flexible-retention/2024-07-03-C96-SHiELD-SOM/regridded-zarrs/gaussian_grid_45_by_90/abrupt-{2x,3x,4x}CO2`
  (14612 six-hourly steps from 2020-01-01T06). A 2024-08-14 six-hourly 4deg
  processing exists but predates `total_frozen_precipitation_rate` and
  `PRMSL`; the dataset is recomputed with the current pipeline instead.
- How: `scripts/data_process/configs/shield-som-abrupt-co2-increase-c96-4deg-8layer.yaml`
  now writes the 2026-09-16 six-hourly store and a daily `time_coarsen`
  (same variable lists as the SOM ensemble). On the argo VM:
  `make shield_som_abrupt_co2_increase_c96_dataset RESOLUTION=4deg` (3 pods),
  then `python scripts/data_process/copy_zarrs_to_weka.py gs://vcm-ml-intermediate/2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-abrupt-co2-increase-fme-dataset`,
  then set `available=True` on `MISSING_DATASETS["abrupt"]`.

## D2 — daily 4deg SOM spin-up year — READY TO PROCESS

- Path (fixed): `/climate-default/2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-ensemble-spin-up-fme-dataset/{climate}-spin-up-ic_000N.zarr`
  (1x/2x/4xCO2 `ic_0005`, 3xCO2 `ic_0002` — the members the paper uses).
- Needed by: `eq` — the paper's spin-up stage starts from the 2030-01-01
  state of this dataset and runs one year before the scored ten-year main run.
  `eq-nospinup` covers the experiment without it, starting from the member's
  2031 state.
- Source: raw 45x90 regrids exist for every member —
  `gs://vcm-ml-raw-flexible-retention/2024-07-03-C96-SHiELD-SOM/regridded-zarrs/gaussian_grid_45_by_90/{climate}-spin-up-ic_000N`
  (1460 six-hourly steps from 2030-01-01T06). Only 1deg processed versions
  exist in `gs://vcm-ml-intermediate` (2024-08-15, 2026-06-08).
- How: `scripts/data_process/configs/shield-som-spin-up-c96-4deg-8layer.yaml`
  (clone of the 1deg config with 45x90 inputs and the daily `time_coarsen`
  block). On a machine with argo access:
  `make shield_som_c96_spin_up_dataset RESOLUTION=4deg` (4 pods), then
  `python scripts/data_process/copy_zarrs_to_weka.py gs://vcm-ml-intermediate/2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-ensemble-spin-up-fme-dataset`,
  then set `available=True` on `MISSING_DATASETS["spin-up"]`. The configs
  assume the daily time labels sit at 06Z like the main dataset's (six-hourly
  source starts at 06Z, so they do).

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
