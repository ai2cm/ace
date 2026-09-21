# Plan: replicating the ACE2S-SHiELD+ paper experiments on the FM runs

Living plan for reproducing the ACE experiments of
[ai2cm/ace2s-shield-plus-paper](https://github.com/ai2cm/ace2s-shield-plus-paper)
("Disentangling the effects of sea surface temperature and CO2 in global
machine learned weather-climate emulators", arXiv 2606.07928) on the 4deg daily
FM / c96 / era5 training runs of this directory, and transferring the abrupt
4xCO2 experiments to ERA5 with prescribed SST. Machinery lives in
`generate_paper_configs.py` (config generator, one "kind" per experiment type),
`submit_paper_jobs.py` (gantry submission), `run-ace-evaluator.sh`,
`run-ace-inference.sh`, `run-ace-som-two-stage.sh`; dataset gaps in
`MISSING_DATASETS.md`; vocabulary in `CONTEXT.md` ("Paper-replication
experiments").

Resume with a fresh agent: read this file, `MISSING_DATASETS.md`, and the
docstrings of `generate_paper_configs.py` / `submit_paper_jobs.py`, then check
the status section below against `argo list`, `beaker`, and the GCS paths
(weka is not mounted on the submitting machine).

## Status (2026-09-21)

- **Submitted 2026-09-17, all 25 non-D3 kinds for the A1/A2/A3 cells**
  (`--arm a1 a2 a3`, workspace `ai2/ace`, clusters jupiter + titan, priority
  normal, wandb group `ace2-fm-paper-2026-06-26`): 3347 experiments, exactly
  the expected count per kind. **1807 succeeded, 0 failed at run time**, on
  every `nc-sfno` and `nc-swin-v2` cell in the fm, c96 and era5 regimes.
- **Blocked: the 1540 jobs on `nc-swin-v2.1` cells** (19 SHiELD-eligible and
  13 ERA5-eligible runs). Their training result datasets hold `best_ckpt.tar`,
  `ckpt.tar` and EMA checkpoints but no
  `training_checkpoints/best_inference_ckpt.tar`, so Beaker could not create
  the container ("path does not exist"). The trainings did finish (150
  epochs, exit 0, e.g. `ace2-fm-nc-swin-v2.1-fm-a3` =
  https://beaker.org/ex/01M2HEKMW51M14JRD0BXZCFNZ1, dataset
  `01M2HEKMWCCH2MJ2J9NX3PX8WX`); the file is missing because **every inline
  weighted inference of every v2.1 run scored NaN** ("Inference error: nan"
  at all 15 inference epochs of fm-a3, all 9 of c96-a1
  https://beaker.org/ex/01M2HEH9DRW3HM2C0QYKVPQFXD, all 75 of era5-a1
  https://beaker.org/ex/01M2HEJB25ZYGRBTDZP24T3KZK), so
  `best_inference_error` stayed `inf` and fme never wrote the checkpoint.
  fm-a3 also skipped 21 non-finite training losses. This is an nc-swin-v2.1
  model problem, not a submission problem; the paper experiments deleted
  2026-09-21 can be resubmitted with `--skip-if-in-beaker` once a v2.1
  checkpoint exists (or pointed at `best_ckpt.tar` if that is what is wanted:
  `submit_paper_jobs.CHECKPOINT_PATH`).
- **D3 on weka (2026-09-21)**: processed by argo `compute-fme-dataset-ensemble-kf56p`
  (36 members, 90 daily labels each, 2031-01-02T00 .. 2031-04-01T00) and copied
  by 36 gantry jobs, all exit 0; `MISSING_DATASETS["abrupt-ensemble"]` points at
  it. The first argo attempt (`tbbqd`) failed on every pod with "argument list
  too long", fixed in `compute_dataset_argo_workflow.yaml` (scripts and config
  were stored twice in the per-run pod template). The two D3 kinds are next to
  submit. The three extra `som-eq-dataCO2-10yr-sstdata-dataonly` jobs for 3xCO2
  `ic_0003-0005` wait on argo `compute-fme-dataset-ensemble-wtnfg`.
- Fixes made during the campaign, all on `exp/alexeyfm` and pushed: start
  times moved to the stores' real 00Z day-2 labels (every daily store labels a
  day's mean at the following 00Z; the first submission's 164 jobs died on
  this); the two-stage `eq` main stage reads forcing through a weka symlink
  directory (`EQ_FORCING_DIR`, created by gantry job `01M2PG8TRA76QYRJ9BPN6EBREQ`)
  because spin-up and member stores live in different directories;
  `run-ace-som-two-stage.sh` given its executable bit; the submitter validates
  against this checkout (`PYTHONPATH`), skips jobs already in Beaker
  (`--skip-if-in-beaker`), and survives a failed gantry call; the Beaker
  listing helper now counts a never-started job as failed. Lesson: do not
  commit locally while a submission driver is running unless commit and push
  are back to back, since gantry refuses a HEAD that is not on the remote.
- Data: D1 and D2 on weka (argo `xwpb9` / `26c2p`, copied 2026-09-16). Raw
  45x90 regrids landed 2026-09-21: D3 at
  `gs://vcm-ml-raw-flexible-retention/2025-02-03-C96-SHiELD-SOM-abrupt-4xCO2-ensemble/regridded-zarrs/gaussian_grid_45_by_90/`
  (36 members `abrupt-4xCO2-ic_00NN`) and the 3xCO2 `ic_0003-0005`
  equilibrium members at
  `gs://vcm-ml-raw-flexible-retention/2024-07-03-C96-SHiELD-SOM/regridded-zarrs/gaussian_grid_45_by_90/`
  (pulled from tape). Not yet verified from this machine (gsutil needs
  `gcloud auth login --update-adc`).

## Kind naming

```
{data}-{experiment}-{co2}-{shape}-{ocean}-{mode}
```

Six slots, always all present, no token containing `-`.

| Slot | Tokens | Meaning |
|---|---|---|
| data | `som` `somabrupt` `somabruptens` `amip` `amipp4k` `amipp2k` `ramped` `era5` | the store supplying SST, sea ice, initial state and reference; also the config's label (`somabrupt*` → `som`, `amipp*` → `amip`). `som` = SOM equilibrium members; `somabrupt` = SHiELD's 10-yr abrupt-4xCO2 run (D1); `somabruptens` = SHiELD's 36-member abrupt ensemble (D3) |
| experiment | `eq` `eqnospinup` `abrupt` `control` | `abrupt` = the CO2-step experiment; `control` = forcing exactly as stored |
| co2 | `4xCO2` `dataCO2` | `4xCO2`: the run is at 4× control (overwritten on a control store, read from an abrupt store); `dataCO2`: as stored |
| shape | `10yr` `1000yr` `43yr` `42yr` `5yr` `ens` (36 monthly ICs × 90 d) `7day` (36 ICs × 7 d) | run length and initial conditions |
| ocean | `sstslab` `sstprescribed` `sstdata` | see below |
| mode | `inference` (free, no target) `eval` (evaluator vs the data store) `dataonly` (store vs itself) | |

Ocean, i.e. where SST comes from at inference (training always reads it from
the data; no training run ever ran a slab):

| | `sstslab` | `sstprescribed` | `sstdata` |
|---|---|---|---|
| SST at step 1 | from the store | from the store | store |
| SST at steps 2…N | mixed-layer ocean integrating the model's own fluxes plus the SOM store's q-flux and depth (`stepper_override`, `interpolate: false`) | from the store, every step | store (no model) |
| Measures | model error incl. its own surface response | atmospheric error given the store's SST | the reference's own diagnostics |

The prescribed-SST 4xCO2 kinds come in pairs differing only in the data slot:
`som-abrupt-4xCO2-…-sstprescribed-eval` (control store's SST while CO2 is
overwritten to 4x: the direct atmospheric response, no SHiELD counterpart)
versus `somabrupt-…` / `somabruptens-…` (SST from SHiELD's own 4xCO2 run,
CO2 from the store: the response given SHiELD's surface warming, scorable).
ERA5 has no slab fields, so ERA5 kinds are `sstprescribed` only.

Job names are `{run}-{kind}[-{climate}][-ic{n}]` for per-run kinds and
`{kind}[-{member}]` for data-only kinds. Config files are
`run_configs/ace-paper-{kind}-config-4deg[-{parts}].yaml`.

## Experiment inventory

Legend: ✅ runnable (data on weka) · 🚧 blocked on Spencer's regrid ·
❌ not planned. "Paper" names the script in `ACE-experiments/inference`; "ours"
marks experiments the paper does not have.

### Slab-ocean (SHiELD-SOM)

| Paper script | Our kind | Data (4deg daily) | Window | Jobs / run | Label | Output | Status |
|---|---|---|---|---|---|---|---|
| `run-ace-equilibrium-climate-inference.sh` (spin-up 2030 → 10 yr main, 4 climates × 5 ICs) | `som-eq-dataCO2-10yr-sstslab-inference` | D2 spin-up member, then SOM paper member | 2030-01-02T00 + ic stagger, `364 - offset` steps; restart at 2031-01-01T00, 3652 steps on spin-up + member symlink dir | 20 (two-stage) | `som` | daily `PRATEsfc` zarr, 1x/3x | ✅ |
| — single-stage variant (ours) | `som-eqnospinup-dataCO2-10yr-sstslab-inference` | SOM paper member | 2031-01-02T00 + ic stagger, `3652 - offset` steps | 20 | `som` | daily `PRATEsfc` zarr, 1x/3x | ✅ |
| `run-ace-1000-year-equilibrium-climate-inference.sh` | `som-eq-dataCO2-1000yr-sstslab-inference` | SOM 1x member tiled ×101, CO2 → climate | 2032-01-01T00, 365250 steps | 4 | `som` | none | ✅ (long) |
| `run-ace-data-only-equilibrium-climate-evaluator.sh` | `som-eq-dataCO2-10yr-sstdata-dataonly` | every SOM member vs itself (3x has `ic_0001-2` only) | 2031-01-02T00, 3652 steps | 17 total; +3 🚧 | `som` | daily `PRATEsfc` zarr, 1x/3x | ✅ |
| — free variant (ours) | `som-abrupt-4xCO2-10yr-sstslab-inference` | SOM 1x member, CO2 → 4x | 2031-01-02T00, 3652 steps | 1 | `som` | monthly netCDF | ✅ |
| `run-ace-abrupt-4xCO2-evaluator.sh` | `somabrupt-abrupt-4xCO2-10yr-sstslab-eval` | D1 `abrupt-4xCO2` | 2020-01-02T00, 3651 steps | 1 | `som` | monthly netCDF | ✅ |
| `run-ace-abrupt-4xCO2-data-only-evaluator.sh` | `somabrupt-abrupt-4xCO2-10yr-sstdata-dataonly` | D1 vs itself | same | 1 total | `som` | monthly netCDF | ✅ |
| `run-ace-abrupt-4xCO2-ensemble-evaluator.sh` | `som-abrupt-4xCO2-ens-sstslab-eval` | SOM 1x member, CO2 → 4x | 2nd of each month 2031-01 … 2033-12 at 00Z, 90 steps | 1 | `som` | none | ✅ |
| `run-ace-abrupt-4xCO2-ensemble-data-only-evaluator.sh` | `somabruptens-abrupt-4xCO2-ens-sstdata-dataonly` | D3 member vs itself | 89 steps from each member's start | 36 total (`--ens-member`) | `som` | none | 🚧 |
| `run-seven-day-1xCO2-and-abrupt-4xCO2-inference-ensemble.sh` | `som-control-dataCO2-7day-sstslab-inference`, `som-abrupt-4xCO2-7day-sstslab-inference` | SOM 1x member; CO2 as is / → 4x | same 36 starts, 7 steps | 1 + 1 | `som` | none | ✅ |
| `run-ace-2pctCO2-*.sh` | — | D4 increasing-CO2 daily | — | — | — | — | ❌ |

D1 also holds `abrupt-2xCO2` and `abrupt-3xCO2`; no kind uses them (paper is
4x only).

### Prescribed-SST (SHiELD)

Aggregator `log_zonal_mean_images: false` (paper) and `forward_steps_in_memory:
40` for the long runs; the 36-IC ensemble kinds use the paper's ensemble
evaluator settings (default aggregator, `forward_steps_in_memory: 1`).

| Paper script | Our kind | Data (4deg daily) | Window | Jobs / run | Label | Output | Status |
|---|---|---|---|---|---|---|---|
| — control for the equilibrium runs (ours) | `som-eq-dataCO2-10yr-sstprescribed-eval` | SOM paper member per climate | 2031-01-02T00 + ic stagger, `3652 - offset` steps, 5 ICs | 20 | `som` | daily `PRATEsfc` zarr, 1x/3x | ✅ |
| — abrupt evaluator with SHiELD's SST instead of the slab (ours) | `somabrupt-abrupt-4xCO2-10yr-sstprescribed-eval` | D1 `abrupt-4xCO2`, its SST and CO2 | 2020-01-02T00, 3651 steps | 1 | `som` | monthly netCDF | ✅ |
| — CO2 step with SST held at 1x (ours) | `som-abrupt-4xCO2-10yr-sstprescribed-eval` | SOM 1x member SST, CO2 → 4x, vs 1x member | 2031-01-02T00, 3652 steps | 1 | `som` | monthly netCDF | ✅ |
| — ensemble with SST from SHiELD's 4xCO2 members (ours) | `somabruptens-abrupt-4xCO2-ens-sstprescribed-eval` | D3 member per job, SST/sea ice/CO2 from it | 89 steps from each member's start | 36 (`--ens-member`) | `som` | none | 🚧 |
| — ensemble CO2 step with SST held at 1x (ours) | `som-abrupt-4xCO2-ens-sstprescribed-eval` | SOM 1x member SST, CO2 → 4x, vs 1x member | same 36 starts, 90 steps | 1 | `som` | none | ✅ |
| — control ensemble (ours) | `som-control-dataCO2-ens-sstprescribed-eval` | SOM 1x member as is | same | 1 | `som` | none | ✅ |
| `run-ace-split-amip-ensemble-inference.sh`, `…-single-member-split-amip-inference-daily-PRATEsfc.sh` | `amip-control-dataCO2-43yr-sstprescribed-eval` | AMIP `ic_0002` (held out) | 1979-01-02T00, 15689 steps (to 2021-12-16) | 1 | `amip` | daily `PRATEsfc` zarr | ✅ |
| `run-ace-split-amip-plus-4K-inference.sh` (+ daily PRATEsfc) | `amipp4k-control-dataCO2-43yr-sstprescribed-eval`, `amipp2k-control-dataCO2-43yr-sstprescribed-eval` | `AMIP-p4K.zarr`, `AMIP-p2K.zarr`, IC from own 1979 state | same | 1 + 1 | `amip` | daily `PRATEsfc` zarr | ✅ |
| `run-ace-amip-split-data-only-evaluator.sh`, `run-ace-amip-variant-data-only-evaluator.sh` | `amip-control-dataCO2-42yr-sstdata-dataonly`, `amipp4k-…`, `amipp2k-…` | `ic_0002`, `AMIP-p4K`, `AMIP-p2K` vs themselves | 1980-01-01T00, 15325 steps | 1 + 1 + 1 total | `amip` | daily `PRATEsfc` zarr | ✅ |
| `run-ace-random-CO2-evaluator.sh` | `ramped-control-dataCO2-5yr-sstprescribed-eval` | ramped `ic_0003` (held out), 1x/2x/4x | 2019-10-02T00, 1918 steps | 3 | `ramped` | none | ✅ |
| `run-ace-amip-constant-CO2-inference.sh` | — | `AMIP-constant-CO2.zarr` | — | — | — | — | ❌ eval suites' `*_constant_co2` entries cover it |

### Prescribed-SST (ERA5)

The paper's abrupt-4xCO2 experiments transferred to ERA5 (Spencer, 2026-09-16:
"for ERA5 for now we'll focus on prescribed SST cases"). ERA5 has no slab
fields. "4x" = 4 × ERA5's own global-mean CO2 on 2015-01-01 (3.986e-4 →
1.5945e-3, volume mixing ratio), held constant for the whole run and all 36
members, as the paper's 4x is 4 × the SOM control's constant. Reference is
ERA5 itself, whose CO2 rises ~26 ppm over the window, so the response is
measured against a drifting baseline (SHiELD's is flat). Runs on fm and era5
cells only; c96 cells never saw ERA5.

| Paper script | Our kind | Data (4deg daily) | Window | Jobs / run | Label | Output | Status |
|---|---|---|---|---|---|---|---|
| `run-ace-abrupt-4xCO2-evaluator.sh` transferred | `era5-abrupt-4xCO2-10yr-sstprescribed-eval` | ERA5 1940–2025, observed SST, CO2 → 1.5945e-3 | 2015-01-01T00, 3652 steps (the eval suites' `10year` window, so the control run exists already) | 1 | `era5` | monthly netCDF | ✅ |
| `run-ace-abrupt-4xCO2-ensemble-evaluator.sh` transferred | `era5-abrupt-4xCO2-ens-sstprescribed-eval` | same | 36 ICs, 1st of each month 2015-01 … 2017-12, 90 steps | 1 | `era5` | none | ✅ |
| — control ensemble (ours; mirrors `som-control-dataCO2-ens-sstprescribed-eval`) | `era5-control-dataCO2-ens-sstprescribed-eval` | ERA5, CO2 as observed | same | 1 | `era5` | none | ✅ |

### Figure 8 (and 10) under each ocean

90-day global-mean response to abrupt 4xCO2, ensemble mean over 36 monthly
ICs, metrics from wandb `inference/mean/weighted_mean_{gen,target}/*` as in
the paper notebook `figures-08-10.ipynb`.

| Line | slab (paper) | SHiELD prescribed | ERA5 prescribed |
|---|---|---|---|
| ACE 4xCO2 | `som-abrupt-4xCO2-ens-sstslab-eval` | `somabruptens-abrupt-4xCO2-ens-sstprescribed-eval` (SHiELD's SST, D3) and `som-abrupt-4xCO2-ens-sstprescribed-eval` (1x SST) | `era5-abrupt-4xCO2-ens-sstprescribed-eval` |
| SHiELD / ERA5 4xCO2 | `somabruptens-abrupt-4xCO2-ens-sstdata-dataonly` (D3) | target of `somabruptens-abrupt-4xCO2-ens-sstprescribed-eval` | none exists |
| 1xCO2 dashed | target of the slab job | target of `som-control-dataCO2-ens-sstprescribed-eval` | target of `era5-control-dataCO2-ens-sstprescribed-eval` |

Ensemble means over the 36 per-member jobs are taken at analysis time; the
single-job kinds already log the ensemble mean.

## Datasets

See `MISSING_DATASETS.md` for full detail.

| | Path (under `/climate-default/`) | Unblocks | Status |
|---|---|---|---|
| D1 | `2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-abrupt-co2-increase-fme-dataset/abrupt-{2x,3x,4x}CO2.zarr` | `som-abrupt-4xCO2-10yr-{slab-eval,sst-eval,data-only}` | ✅ on weka |
| D2 | `2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-ensemble-spin-up-fme-dataset/{climate}-spin-up-ic_000N.zarr` | `som-eq-dataCO2-10yr-sstslab-inference` | ✅ on weka |
| D3 | `2026-09-21-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset/abrupt4xCO2-ic_00NN.zarr` | `somabruptens-abrupt-4xCO2-ens-sstdata-dataonly`, `somabruptens-abrupt-4xCO2-ens-sstprescribed-eval` | ✅ on weka (2026-09-21) |
| 3x members | new-dated SOM ensemble store with 3xCO2 `ic_0003-0005` | 3 more `som-eq-dataCO2-10yr-sstdata-dataonly` jobs | 🚧 Spencer regrid; config not written |
| D4 | daily increasing-CO2 | `2pct*` | ❌ not planned |
| AMIP `ic_0002`, `AMIP-p4K`, `AMIP-p2K`, ramped `ic_0003`, ERA5 | see `generate_paper_configs.AMIP_VARIANTS` / `RAMPED_DATASET` / `ERA5_DATASET` | prescribed-SST kinds | ✅ (p2k/p4k copied in July via `amip_p2k_p4k_transfer.yaml`, not re-verified) |

## How to run

From this directory in the `fme` env, configs committed and pushed (gantry
clones HEAD; `--dry-run` skips that check and the `validate_config` pass).
`--kind` is required. Each kind runs on the regimes that trained on its data
(SHiELD-data kinds: fm + c96; ERA5 kinds: fm + era5). `--arm a1 a2 a3`
restricts to the norm-ablation cells; `--arch`, `--regime`, `--run` narrow
further; `--climate` applies to `som-eq-*` and `ramped-control-dataCO2-5yr-sstprescribed-eval`,
`--ic` to the staggered-IC kinds, `--ens-member` to the per-member D3 kinds;
`--skip-if-in-wandb` fills in only what has no finished wandb run.

```bash
# Regenerate after editing the generator (all kinds, or --kind ...)
python generate_paper_configs.py

# Preview job expansion without submitting
python submit_paper_jobs.py --kind som-eq-dataCO2-10yr-sstprescribed-eval amip-control-dataCO2-43yr-sstprescribed-eval --arm a1 --dry-run

# Prescribed-SST first (see decisions log): SHiELD ...
python submit_paper_jobs.py --kind somabrupt-abrupt-4xCO2-10yr-sstprescribed-eval som-abrupt-4xCO2-10yr-sstprescribed-eval --arm a1 a2 a3
python submit_paper_jobs.py --kind som-control-dataCO2-ens-sstprescribed-eval som-abrupt-4xCO2-ens-sstprescribed-eval --arm a1 a2 a3
python submit_paper_jobs.py --kind som-eq-dataCO2-10yr-sstprescribed-eval ramped-control-dataCO2-5yr-sstprescribed-eval --arm a1 a2 a3
python submit_paper_jobs.py --kind amip-control-dataCO2-43yr-sstprescribed-eval amipp4k-control-dataCO2-43yr-sstprescribed-eval amipp2k-control-dataCO2-43yr-sstprescribed-eval --arm a1 a2 a3
# ... and ERA5
python submit_paper_jobs.py --kind era5-control-dataCO2-ens-sstprescribed-eval era5-abrupt-4xCO2-ens-sstprescribed-eval era5-abrupt-4xCO2-10yr-sstprescribed-eval --arm a1 a2 a3
# Reference rows (one checkpoint, --data-only-run; default ace2-fm-nc-swin-v2-fm-a1)
python submit_paper_jobs.py --kind som-eq-dataCO2-10yr-sstdata-dataonly somabrupt-abrupt-4xCO2-10yr-sstdata-dataonly amip-control-dataCO2-42yr-sstdata-dataonly amipp4k-control-dataCO2-42yr-sstdata-dataonly amipp2k-control-dataCO2-42yr-sstdata-dataonly
# Slab-ocean: cheap first, watch the 1xCO2 SST drift before the long ones
python submit_paper_jobs.py --kind som-eqnospinup-dataCO2-10yr-sstslab-inference --arm a1 --climate 1xCO2 --ic 1
python submit_paper_jobs.py --kind som-abrupt-4xCO2-10yr-sstslab-inference somabrupt-abrupt-4xCO2-10yr-sstslab-eval som-abrupt-4xCO2-ens-sstslab-eval --arm a1 a2 a3
python submit_paper_jobs.py --kind som-control-dataCO2-7day-sstslab-inference som-abrupt-4xCO2-7day-sstslab-inference --arm a1 a2 a3
python submit_paper_jobs.py --kind som-eqnospinup-dataCO2-10yr-sstslab-inference som-eq-dataCO2-10yr-sstslab-inference --arm a1 a2 a3 --skip-if-in-wandb
python submit_paper_jobs.py --kind som-eq-dataCO2-1000yr-sstslab-inference --arm a1 --arch nc-swin-v2
# After D3
python submit_paper_jobs.py --kind somabruptens-abrupt-4xCO2-ens-sstdata-dataonly
python submit_paper_jobs.py --kind somabruptens-abrupt-4xCO2-ens-sstprescribed-eval --arm a1 --ens-member 1 2 3   # 36/run
```

Per-run job counts on SHiELD data (53 fm/c96 runs, 41 with `--arm`, after
the nc-swin-v2.1 cells landed): slab-ocean 49 (`eq` 20, `eqnospinup` 20,
`1000yr` 4, abrupt 10yr 2, ens 1, 7day 2), prescribed 30 (`eq` sst 20, abrupt
10yr 2, ens 2, amip 3, ramped 3) plus 36 `somabruptens-abrupt-4xCO2-ens-sstprescribed-eval` after
D3. On ERA5 (39 fm/era5 runs, 29 with `--arm`): 3. Data-only 21 now, +36 with
D3. Everything ≈ 4300 jobs now, ≈ 6300 with D3; the arms alone
≈ 3300. Submit by kind and arm; run the per-member D3 kind on a subset
(`--run`, `--ens-member`).

The slab is forward-Euler at a daily step, untested here (paper was 6-hourly):
run one `som-eqnospinup-dataCO2-10yr-sstslab-inference` 1xCO2 job and check SST drift
before submitting the rest of the slab kinds.

## Instructions for the next agent

Both remaining datasets landed on GCS as raw 45x90 regrids on 2026-09-21. The
work is: process each into a 4deg daily fme store on argo, copy to weka, point
the generator at it, submit. Do the D3 ensemble first (it feeds figure 8), the
3xCO2 members second. Run everything from `configs/experiments/2026-06-26-fm`
in the `fme` env (`/Users/alexeyy/mamba/envs/fme/bin/python`; `gantry`,
`beaker`, `argo`, `gsutil` need `PATH=/Users/alexeyy/mamba/envs/fme/bin:/Users/alexeyy/.local/bin:$PATH`).
If `gsutil` says "Reauthentication required", have the user run
`gcloud auth login --update-adc`. Never commit while a submission is running
unless you push immediately after (gantry refuses a HEAD not on the remote).

### A. D3: 36-member abrupt-4xCO2 ensemble

1. Verify the upload: `gsutil ls gs://vcm-ml-raw-flexible-retention/2025-02-03-C96-SHiELD-SOM-abrupt-4xCO2-ensemble/regridded-zarrs/gaussian_grid_45_by_90/`
   should list 36 `abrupt-4xCO2-ic_00NN/` directories, each with
   `fluxes_2d.zarr`, `full_state.zarr`, `ocean_forcing.zarr` etc. like the
   1deg ones.
2. Write `scripts/data_process/configs/shield-som-abrupt4xCO2-ensemble-c96-4deg-8layer.yaml`
   by cloning the 1deg file next to it: `gaussian_grid_180_by_360` →
   `gaussian_grid_45_by_90` in every run path; `data_output_directory` →
   `gs://vcm-ml-intermediate/2026-09-21-vertically-resolved-4deg-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset`;
   stats directories renamed the same way and every run listed under
   `stats.exclude_runs` (no stats are needed; D1's 4deg config does this);
   append the `time_coarsen` block from
   `shield-som-abrupt-co2-increase-c96-4deg-8layer.yaml` verbatim except its
   two output directories, which become
   `…/2026-09-21-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset`
   (+ `-stats`). Keep the run keys `abrupt4xCO2-ic_00NN`: the generator reads
   `abrupt4xCO2-{member}.zarr`.
3. Add a Makefile target next to `shield_som_abrupt_co2_increase_c96_dataset`:
   `shield_som_abrupt_4xco2_ensemble_c96_dataset` running
   `./compute_dataset.sh --dataset $(if $(filter 4deg,$(RESOLUTION)),--time-coarsen) --config configs/shield-som-abrupt4xCO2-ensemble-c96-$(RESOLUTION)-$(LAYERS).yaml`.
   Commit + push (data_process files).
4. `cd scripts/data_process && make shield_som_abrupt_4xco2_ensemble_c96_dataset RESOLUTION=4deg`
   (argo cluster `gke_vcm-ml_us-central1-c_ml-cluster-dev`, needs
   `brew install python-yq`); 36 pods, expect ~1 h. Watch with `argo list`.
5. When Succeeded: check one member's daily time axis (copy `time/` with
   `gsutil -m cp -r`, open with `zarr`, decode with `cftime`): expect labels at
   00Z starting the day after the run start, ~90 of them. If the count is not
   90 or 91, adjust `ABRUPT_ENSEMBLE_N_STEPS - 1` in
   `_abrupt_ensemble_member_configs` so `n_forward_steps` = labels − 1.
6. Copy to weka: `PATH=… python scripts/data_process/copy_zarrs_to_weka.py gs://vcm-ml-intermediate/2026-09-21-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset`
   (36 gantry jobs in `ai2/climate-titan`); confirm exit 0 with
   `beaker experiment get <id> --format json`.
7. In `generate_paper_configs.py`, `MISSING_DATASETS["abrupt-ensemble"]`: set
   `path` to `/climate-default/2026-09-21-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset`
   and `available=True`; `python generate_paper_configs.py`; validate one config
   with `PYTHONPATH=<repo root> python -m fme.ace.validate_config --config_type evaluator run_configs/ace-paper-somabruptens-abrupt-4xCO2-ens-sstprescribed-eval-config-4deg-ic_0001.yaml`;
   update `MISSING_DATASETS.md` and this file; commit; push.
8. Submit:
   ```bash
   python submit_paper_jobs.py --kind somabruptens-abrupt-4xCO2-ens-sstdata-dataonly --skip-if-in-beaker
   python submit_paper_jobs.py --kind somabruptens-abrupt-4xCO2-ens-sstprescribed-eval --arm a1 a2 a3 --skip-if-in-beaker
   ```
   36 + 36 × 22 = 828 jobs (`nc-swin-v2.1` cells will fail to start until they
   have `best_inference_ckpt.tar`; pass `--arch nc-sfno nc-swin-v2` to skip
   them). Run the submitter detached (`nohup … &`) and log to a file; it takes
   ~1 h.
9. Monitor with a 20-minute cron: list `ai2/ace` via
   `_beaker_listing.fetch_experiments_by_name("ai2/ace", "-sst")`, keep only
   names containing a kind from `generate_paper_configs.KINDS`, tally by
   `status`; read logs of any `failed`, resubmit infrastructure failures with
   `--run <run> --ens-member N`, record anything else here.

### B. 3xCO2 equilibrium members ic_0003-0005

1. Verify: `gsutil ls gs://vcm-ml-raw-flexible-retention/2024-07-03-C96-SHiELD-SOM/regridded-zarrs/gaussian_grid_45_by_90/`
   now lists `3xCO2-ic_0003/`, `3xCO2-ic_0004/`, `3xCO2-ic_0005/`.
2. Write `scripts/data_process/configs/shield-som-ensemble-3xco2-members-c96-4deg-8layer.yaml`:
   a copy of `shield-som-ensemble-c96-4deg-8layer.yaml` whose `runs` contain
   only the three new members (they are already there, commented out) and
   whose output directories are **unchanged** (the existing
   `2026-06-08-…-4deg-c96-shield-som-ensemble-fme-dataset` and its `-daily-`
   sibling), so the three zarrs land next to the 17 existing ones and
   `SOM_DATASET` needs no change. List the three under `stats.exclude_runs`;
   do not pass `--stats` (the training stats must not change). Makefile
   target `shield_som_ensemble_3xco2_members_c96_dataset` running
   `./compute_dataset.sh --dataset --time-coarsen --config configs/shield-som-ensemble-3xco2-members-c96-$(RESOLUTION)-$(LAYERS).yaml`.
   Commit + push; run with `RESOLUTION=4deg`; 3 pods.
3. Copy the three daily zarrs to weka with `copy_zarrs_to_weka.py <daily dir>/3xCO2-ic_000{3,4,5}.zarr`
   (pass the three `.zarr` paths, not the directory, or the 17 existing zarrs
   are recopied).
4. `SOM_MEMBERS["3xCO2"]` → `("ic_0001", …, "ic_0005")` in
   `generate_paper_configs.py`; regenerate (three new
   `som-eq-dataCO2-10yr-sstdata-dataonly` configs appear, nothing else
   changes: `CLIMATES["3xCO2"].member` stays `ic_0002`, the paper's choice);
   update `MISSING_DATASETS.md` ("3x members" section) and this file; commit;
   push.
5. `python submit_paper_jobs.py --kind som-eq-dataCO2-10yr-sstdata-dataonly --climate 3xCO2 --skip-if-in-beaker`
   — `--skip-if-in-beaker` does not see data-only names (no `ace2-fm-` prefix),
   so restrict to the new members by checking the two existing 3xCO2 jobs are
   in Beaker and passing nothing else; if the submitter offers no member
   filter, add `--member` or submit the three configs by hand with
   `run-ace-evaluator.sh`.

### C. Afterwards

- Update the status section: counts per kind, D3 and 3x rows in the datasets
  table to ✅, and remove the 🚧 legend entries if nothing is blocked.
- The `nc-swin-v2.1` cells stay blocked until their trainings produce
  `best_inference_ckpt.tar` (see status); re-running the seven submit
  commands in "How to run" with `--skip-if-in-beaker` fills them in once it
  exists.

## Decisions log

- Priority (2026-09-16, per Spencer): SHiELD-like runs in both modes as in the
  paper; ERA5 prescribed-SST only for now. Prescribed-SST kinds go first: no
  training run ever ran a slab, so a slab result conflates coupling error with
  model error, and prescribed-SST evaluators give step-wise metrics against a
  valid target.
- Models: each kind runs on the regimes that trained on its data
  (`submit_paper_jobs.REGIME_GRIDS`): c96 → SHiELD data, era5 → ERA5 data,
  fm → both; the hand-written ERA5 runs count as era5. `--arm` narrows to
  norm-ablation cells. c96 fine-tune runs (ERA5 under label `amip`) are a
  separate experiment family and excluded.
- Checkpoint: `best_inference_ckpt.tar` only.
- Naming scheme (see "Kind naming"), adopted 2026-09-16 and revised
  2026-09-17 before any job was submitted: six slots always present, no token
  containing `-`; the ocean slot names only the SST source (`sstslab` /
  `sstprescribed` / `sstdata`), and "SST held at control while CO2 steps" is
  carried by data + co2 slots, not a special ocean token. Abrupt kinds are 4xCO2 only, as in the paper;
  the earlier 2x/3x abrupt configs were dropped (D1 keeps the stores). The
  paper's seven-day script became two kinds (control / 4xCO2).
- Slab `interpolate: false` (training default), not the paper's `true`.
  Prescribed-SST kinds carry no `stepper_override` at all.
- Paper CO2 constants for overrides (1x 0.00036343, 2x 0.00072686,
  3x 0.00109029, 4x 0.0014537, volume mixing ratio); equilibrium runs read CO2
  from the member. ERA5 4x = 4 × the store's value on 2015-01-01
  (`ERA5_CO2_4X` = 1.5945e-3), one constant for the run and all ensemble
  members, applied from the first step; the initial atmosphere is ERA5's 1x
  state, so the step is abrupt exactly as in the paper.
- ERA5 window 2015–2024: the eval suites' `10year` window, so the 10-year
  control run exists for every run; ensemble ICs are its first three years,
  as the paper's are the first three years of the SOM decade.
- Members: 1x/2x/4x `ic_0005`, 3x `ic_0002` (paper). "Held out" means held
  out of the A-cells: they trained on SOM `ic_0001` of 1x/2x/4x, AMIP
  `ic_0001`, ramped `.zarr` + `ic_0002`. The hand-written `fm-random-v1/v3`
  and `fm-0.x-v1` runs trained on 3xCO2 `ic_0001-2`, AMIP `ic_0002` and ramped
  `ic_0003` as well, so those members are in-sample for them.
- Staggered ICs are separate jobs, one day apart (paper: 6 h).
  `som-eq-dataCO2-10yr-sstprescribed-eval` follows the `eq` IC protocol so it is a like-for-like
  control.
- Prescribed-SST runs are evaluators, not the paper's inference + data-only
  pair: under prescribed SST the reference is a valid step-by-step target, so
  the evaluator gives the comparison directly. Data-only rows are still
  produced for the paper-format reference diagnostics.
- AMIP as one run, not the paper's three chained stages (spin-up /
  train-validate / test via restart files): the trajectory is identical, the
  evaluator writes no restart, and the windows are an analysis cut. The
  paper's 5 AMIP ICs are five identical start times, i.e. one run for a
  deterministic model.
- Labels follow the data: `som`, `amip`, `ramped`, `era5`. `AMIP-p4K`/`p2K`
  never appeared in training; `amip` is the closest label and what the SST
  sweep (`submit_sst_jobs.py`, forcing SST +2/+4 K) implies.
- Figure 8 under prescribed SST: `somabruptens-abrupt-4xCO2-ens-sstprescribed-eval` needs
  SHiELD's own 4xCO2 SST, hence one job per D3 member (36/run, 89 steps, CO2
  from the member) rather than the slab version's one 36-IC job. The
  `som-…-sstprescribed` ensembles (control SST, CO2 → 4x) isolate the direct
  CO2 response and have no SHiELD counterpart. No aggregator override for the ensemble kinds, as the paper's
  ensemble evaluator config.
- `amip-constant-co2` skipped: eval suites already run AMIP `ic_0001` with
  constant CO2. `amip-control-dataCO2-43yr-sstprescribed-eval` on `ic_0002` is kept because the suites never
  touch a held-out AMIP member.
- D1 recomputed from raw with the current pipeline rather than coarsening the
  2024-08-14 six-hourly store (which lacks `total_frozen_precipitation_rate`
  and `PRMSL` and would fail the SOM `time_coarsen` name lists).
- Outputs: paper-exact (daily PRATEsfc zarr for 1x/3x equilibrium, data-only
  and the AMIP runs; monthly netCDF of 11 surface vars for the 10-year abrupt
  runs; nothing for the ensembles and random-CO2), all to `/results`.
- Generator/submitter renamed from `*_som_*` to `*_paper_*`, config prefix
  `ace-paper-`, wandb group `ace2-fm-paper-2026-06-26`, before any job was
  submitted. `run-ace-som-two-stage.sh` keeps its name (SOM-specific).
- Job volume is the main risk: `--kind` is required; everything is ≈ 4300
  jobs now, ≈ 6300 with D3 (see "How to run"). Sibling submitters now skip
  already-submitted jobs via the Beaker listing (`drop_jobs_in_beaker`);
  this one still uses `--skip-if-in-wandb`. Switch before the first bulk submit.
