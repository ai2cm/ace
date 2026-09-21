# Datasets for the paper-replication experiments

`generate_paper_configs.py` reproduces the ACE experiments of
[ai2cm/ace2s-shield-plus-paper](https://github.com/ai2cm/ace2s-shield-plus-paper/tree/main/ACE-experiments/inference)
on the 4deg daily SHiELD data. Inputs the paper needs which have no 4deg daily
counterpart are tracked in `MISSING_DATASETS` in `generate_paper_configs.py`.
Their configs are generated anyway so the machinery is complete — against the
real name once a processing config fixes it, a placeholder
(`/climate-default/TBD-...`) before that — and `submit_paper_jobs.py` refuses
the dependent kinds until the entry is marked `available`.

To bring one online: produce the dataset, copy it to weka with
`scripts/data_process/copy_zarrs_to_weka.py` (one gantry job per zarr; needs
`gantry` on `PATH`, e.g. the `fme` env's `bin`), set the real name and
`available=True` in `MISSING_DATASETS`, regenerate, commit, submit.

## Still missing

### 3xCO2 `ic_0003-0005` of the SOM ensemble — RAW REGRID LANDED, PROCESSING PENDING

- Not a `MISSING_DATASETS` entry: the 2026-06-08 SOM ensemble store simply has
  only `ic_0001-2` for 3xCO2, so `SOM_MEMBERS["3xCO2"]` lists two members and
  `som-eq-dataCO2-10yr-sstdata-dataonly` runs 17 jobs instead of 20.
- Source: pulled from tape and regridded by Spencer, landed 2026-09-21 at
  `gs://vcm-ml-raw-flexible-retention/2024-07-03-C96-SHiELD-SOM/regridded-zarrs/gaussian_grid_45_by_90/3xCO2-ic_000{3,4,5}`.
- How: a copy of `shield-som-ensemble-c96-4deg-8layer.yaml` listing only the
  three members (already present there, commented out) with the *existing*
  2026-06-08 output directories, run with `--dataset --time-coarsen` and no
  `--stats`; copy the three daily zarrs to weka; extend `SOM_MEMBERS["3xCO2"]`;
  regenerate `som-eq-dataCO2-10yr-sstdata-dataonly`. Step-by-step in the
  plan's "Instructions for the next agent".

### D4 — daily 4deg increasing-CO2 (2%/yr) — NOT PLANNED

- Needed by the paper's `2pct*` scripts. A 6-hourly 4deg processing from the
  old pipeline exists (`2024-07-16-…`); no daily version and no kind written.

## Landed (2026-09-21)

### D3 — daily 4deg abrupt-4xCO2 36-member ensemble

- Path: `/climate-default/2026-09-21-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset/abrupt4xCO2-ic_00NN.zarr`,
  90 daily labels each, 2031-01-02T00 .. 2031-04-01T00 (so
  `n_forward_steps` = 89, `ABRUPT_ENSEMBLE_N_STEPS - 1`, unchanged). Same 99
  arrays as the SOM ensemble daily store.
- Unblocked: `somabruptens-abrupt-4xCO2-ens-sstdata-dataonly` (36 jobs) and
  `somabruptens-abrupt-4xCO2-ens-sstprescribed-eval` (36 per training run).
- Produced by argo `compute-fme-dataset-ensemble-kf56p` from
  `scripts/data_process/configs/shield-som-abrupt4xCO2-ensemble-c96-4deg-8layer.yaml`
  (Spencer's 45x90 regrid of
  `2025-02-03-C96-SHiELD-SOM-abrupt-4xCO2-ensemble`, landed 2026-09-21). The
  first attempt died on every pod with "argument list too long": the per-run
  argo pod template stored the scripts and config twice, fixed in
  `compute_dataset_argo_workflow.yaml`. Six-hourly sibling store
  `2026-09-21-vertically-resolved-4deg-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset`
  on GCS, not copied to weka. Copied to weka by 36 gantry jobs
  (`gcs-to-weka-abrupt4xCO2-ic_00NN.zarr`, all exit 0).

## Landed (2026-09-16)

### D1 — daily 4deg abrupt-CO2 runs

- Path: `/climate-default/2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-abrupt-co2-increase-fme-dataset/abrupt-{2x,3x,4x}CO2.zarr`,
  3653 daily labels each, 2020-01-02T00 .. 2030-01-01T00 (a day's mean is
  labelled at the following 00Z, as in every 4deg daily store).
- Unblocked: `somabrupt-abrupt-4xCO2-10yr-sstslab-eval`, `somabrupt-abrupt-4xCO2-10yr-sstprescribed-eval`,
  `somabrupt-abrupt-4xCO2-10yr-sstdata-dataonly`. The 2x/3x stores are unused (paper is 4x only).
- Produced by argo `compute-fme-dataset-ensemble-xwpb9` from
  `scripts/data_process/configs/shield-som-abrupt-co2-increase-c96-4deg-8layer.yaml`
  (raw 45x90 regrids of `2024-07-03-C96-SHiELD-SOM`, current pipeline; the
  2024-08-14 six-hourly 4deg store lacks `total_frozen_precipitation_rate` and
  `PRMSL`). A six-hourly sibling store
  `2026-09-16-vertically-resolved-4deg-c96-shield-som-abrupt-co2-increase-fme-dataset`
  sits next to it on GCS, not copied to weka.

### D2 — daily 4deg SOM spin-up year

- Path: `/climate-default/2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-ensemble-spin-up-fme-dataset/{1xCO2,2xCO2,4xCO2}-spin-up-ic_0005.zarr`,
  `3xCO2-spin-up-ic_0002.zarr`; 365 daily labels each, 2030-01-02T00 ..
  2031-01-01T00. The member stores begin 2031-01-02T00, so the two-stage `eq`
  main stage reads its forcing through a weka directory of ordered symlinks
  (`generate_paper_configs.EQ_FORCING_DIR`, created by a one-off gantry job).
- Unblocked: `som-eq-dataCO2-10yr-sstslab-inference`.
- Produced by argo `compute-fme-dataset-ensemble-26c2p` from
  `scripts/data_process/configs/shield-som-spin-up-c96-4deg-8layer.yaml`.
  Six-hourly sibling on GCS as for D1.

## Never missing

- Prescribed-SST inputs: AMIP `ic_0002`
  (`2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset`),
  `AMIP-p4K.zarr` / `AMIP-p2K.zarr`
  (`2026-07-09-vertically-resolved-c96-4deg-daily-shield-amip-{p4k,p2k}-dataset`),
  ramped `ic_0003`
  (`2026-06-08-vertically-resolved-c96-shield-ramped-climSST-random-CO2-ensemble-fme-dataset-4deg-daily`).
  The AMIP ensemble and ramped roots are the training configs' own. The p2k
  and p4k stores were copied in July via
  `scripts/data_process/amip_p2k_p4k_transfer.yaml` (`copy_zarrs_to_weka.py
  --mapping`); weka is not mounted on the submitting machine, so re-run the
  mapping if a job fails on a missing path.
- 1000-year forcing: the 1xCO2 member tiled via `n_repeats` (all SOM forcing
  but CO2 is climatological), as the paper does with its Hugging Face forcing.
- Every other `som-*` kind: the 2026-06-08 4deg daily SOM ensemble.
- ERA5 kinds: `/climate-default/2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr`,
  the training configs' own ERA5 store (00Z labels).
