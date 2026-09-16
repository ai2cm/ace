# Plan: replicating the ACE2S-SHiELD+ paper experiments on the FM runs

Living plan for reproducing the ACE experiments of
[ai2cm/ace2s-shield-plus-paper](https://github.com/ai2cm/ace2s-shield-plus-paper)
("Disentangling the effects of sea surface temperature and CO2 in global
machine learned weather-climate emulators", arXiv 2606.07928) on the 4deg daily
FM / c96 training runs of this directory. Machinery lives in
`generate_paper_configs.py` (config generator, one "kind" per experiment type),
`submit_paper_jobs.py` (gantry submission), `run-ace-evaluator.sh`,
`run-ace-inference.sh`, `run-ace-som-two-stage.sh`; dataset gaps in
`MISSING_DATASETS.md`; vocabulary in `CONTEXT.md` ("Paper-replication
experiments").

Resume with a fresh agent: read this file, `MISSING_DATASETS.md`, and the
docstrings of `generate_paper_configs.py` / `submit_paper_jobs.py`, then check
the status section below against `argo list`, `beaker`, and the GCS paths
(weka is not mounted on the submitting machine).

## Status (2026-09-16, end of day)

- All kinds are written: 11 slab-ocean, 9 prescribed-SST (see inventory).
  Every kind but the two on D3 (`abrupt-ens-data-only`,
  `abrupt-ens-eval-sst`) is submittable. No experiment jobs submitted yet;
  wandb group `ace2-fm-paper-2026-06-26` is empty.
- Commits on `exp/alexeyfm` since the plan was first written: `593aa955e`
  (rename to paper-wide names), `792e0788c` (prescribed-SST kinds),
  `62f3366e6` (D1/D2 marked available), `f7259c4fd` (docs), `c4159803c`
  (figure-8 prescribed-SST ensemble kinds), plus this docs commit. Pushed.
- Argo (`gke_vcm-ml_us-central1-c_ml-cluster-dev`): `xwpb9` (D1) and
  `26c2p` (D2) succeeded. Seven `gcs-to-weka-*` gantry copy jobs
  (`01M2P5FQ…` … `01M2P5GA…`, workspace `ai2/climate-titan`) finished with
  exit 0; D1 and D2 are on weka.
- Spencer's 45x90 regrids (Gaea/Snakemake): the 36-member abrupt-4xCO2
  ensemble (D3) and 3xCO2 `ic_0003-0005`. Nothing in
  `gs://vcm-ml-raw-flexible-retention` yet.

## Experiment inventory

Legend: ✅ runnable (data on weka) · 🚧 blocked on Spencer's regrid ·
❌ not planned.

### Slab-ocean (SST-interactive)

| Paper script | Our kind | Data (4deg daily) | Jobs / run | Status |
|---|---|---|---|---|
| `run-ace-equilibrium-climate-inference.sh` (spin-up 2030 → 10 yr main, 4 climates x 5 ICs) | `eq` | SOM ensemble + D2 spin-up | 20 (two-stage) | ✅ |
| — single-stage variant (ours) | `eq-nospinup` | SOM ensemble | 20 | ✅ |
| `run-ace-1000-year-equilibrium-climate-inference.sh` | `eq-1000yr` | SOM 1x member tiled | 4 | ✅ (long) |
| `run-ace-data-only-equilibrium-climate-evaluator.sh` | `data-only` | SOM ensemble (3x has `ic_0001-2` only) | 17 total; +3 🚧 | ✅ |
| `run-ace-abrupt-4xCO2-evaluator.sh` | `abrupt-10yr-eval` (2x/3x/4x) | D1 | 3 | ✅ |
| — prescribed-SST variant (ours) | `abrupt-10yr-eval-sst` | D1 | 3 | ✅ |
| — free inference variant (ours) | `abrupt-10yr` | SOM 1x member | 3 | ✅ |
| `run-ace-abrupt-4xCO2-data-only-evaluator.sh` | `abrupt-data-only` | D1 | 3 total | ✅ |
| `run-ace-abrupt-4xCO2-ensemble-evaluator.sh` (36 monthly ICs x 90 d) | `abrupt-ens` | SOM 1x member | 1 | ✅ |
| `run-ace-abrupt-4xCO2-ensemble-data-only-evaluator.sh` | `abrupt-ens-data-only` | D3 | 36 total | 🚧 |
| `run-seven-day-1xCO2-and-abrupt-4xCO2-inference-ensemble.sh` | `7day` | SOM 1x member | 2 | ✅ |
| `run-ace-2pctCO2-inference.sh`, `run-ace-deterministic-2pctCO2-ensemble.sh`, `run-ace-2pctCO2-data-only-evaluator.sh` | `2pct*` | D4 increasing-CO2 daily | — | ❌ |

Paper figures 8 and 10 (90-day abrupt-4xCO2 response) need `abrupt-ens`
(ACE lines and the SHiELD 1xCO2 dashed line, from the same job) and
`abrupt-ens-data-only` (SHiELD 4xCO2 line, D3).

### Prescribed-SST (designed and added 2026-09-16)

All evaluators with the training-time ocean, no `stepper_override`, aggregator
`log_zonal_mean_images: false` (paper), `forward_steps_in_memory: 40`.

| Paper script | Our kind | Data (4deg daily) | Window | Jobs / run | Label | Output |
|---|---|---|---|---|---|---|
| — (ours; control for `eq`) | `eq-eval-sst` | SOM paper member per climate | 2031-01-01T06 + ic stagger, `3652 - offset` steps, 5 ICs | 20 | `som` | daily `PRATEsfc` zarr, 1x/3x |
| `run-ace-split-amip-ensemble-inference.sh`, `run-ace-single-member-split-amip-inference-daily-PRATEsfc.sh` | `amip-eval` | AMIP `ic_0002` (held out) | 1979-01-01T06, 15689 steps (to 2021-12-15) | 1 | `amip` | daily `PRATEsfc` zarr |
| `run-ace-split-amip-plus-4K-inference.sh` (+ daily PRATEsfc) | `amip-p4k`, `amip-p2k` | `AMIP-p4K.zarr`, `AMIP-p2K.zarr`, IC from own 1979 state | same | 1 each | `amip` | daily `PRATEsfc` zarr |
| `run-ace-amip-split-data-only-evaluator.sh`, `run-ace-amip-variant-data-only-evaluator.sh` | `amip-data-only` | `ic_0002`, `AMIP-p4K`, `AMIP-p2K` vs themselves | 1980-01-01T06, 15324 steps | 3 total | `amip` | daily `PRATEsfc` zarr |
| `run-ace-random-CO2-evaluator.sh` | `random-co2-eval` | ramped `ic_0003` (held out), 1x/2x/4x | 2019-10-01T06, 1918 steps | 3 | `ramped` | none |
| `run-ace-abrupt-4xCO2-ensemble-evaluator.sh` under prescribed SST (ours) | `abrupt-ens-eval-sst` | D3 member `abrupt4xCO2-ic_00NN` per job, SST/sea ice/CO2 from it | 89 steps from each member's start | 36 (`--ens-member`) | `som` | none |
| — control ensemble (ours) | `control-ens-eval-sst` | SOM 1x member, 36 monthly ICs, no CO2 override | 2031-01 … 2033-12 starts, 90 steps | 1 | `som` | none |
| — CO2 step at fixed SST (ours) | `abrupt-ens-fixed-sst` | SOM 1x member SST, CO2 → 4x, vs 1x member | same | 1 | `som` | none |
| `run-ace-amip-constant-CO2-inference.sh` | — | `AMIP-constant-CO2.zarr` | | | | skipped: eval suites' `*_constant_co2` entries cover it |

Job names: `{run}-som-eq-eval-sst-{climate}-ic{n}`, `{run}-amip-{ic2,p4k,p2k}-eval`,
`{run}-ramped-{climate}-eval`, `amip-{ic2,p4k,p2k}-data-only`,
`{run}-som-abrupt-4xCO2-ens-eval-sst-ic{n}`, `{run}-som-control-ens-eval-sst`,
`{run}-som-abrupt-4xCO2-ens-fixed-sst`.

Figure 8 (and 10) under prescribed SST: `abrupt-ens-eval-sst` gives the ACE
4xCO2 lines and, as its target, the SHiELD 4xCO2 line (so
`abrupt-ens-data-only` is not needed for this version); `control-ens-eval-sst`
gives the SHiELD 1xCO2 dashed line as its target plus ACE's control drift;
`abrupt-ens-fixed-sst` adds the direct-CO2 line with no SHiELD counterpart.
Ensemble means over the 36 `abrupt-ens-eval-sst` jobs are taken at analysis
time (the single-job kinds already log the ensemble mean). Metrics come from
wandb `inference/mean/weighted_mean_{gen,target}/*` as in the paper notebook
`figures-08-10.ipynb`.

## Datasets

See `MISSING_DATASETS.md` for full detail.

| | Path (under `/climate-default/`) | Unblocks | Status |
|---|---|---|---|
| D1 | `2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-abrupt-co2-increase-fme-dataset/abrupt-{2x,3x,4x}CO2.zarr` | `abrupt-10yr-eval`, `abrupt-10yr-eval-sst`, `abrupt-data-only` | ✅ on weka |
| D2 | `2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-ensemble-spin-up-fme-dataset/{climate}-spin-up-ic_000N.zarr` | `eq` | ✅ on weka |
| D3 | `TBD-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset/abrupt4xCO2-ic_00NN.zarr` | `abrupt-ens-data-only`, `abrupt-ens-eval-sst` | 🚧 Spencer regrid; processing config not written |
| 3x members | new-dated SOM ensemble store with 3xCO2 `ic_0003-0005` | 3 more `data-only` jobs | 🚧 Spencer regrid; config not written |
| D4 | daily increasing-CO2 | `2pct*` | ❌ not planned |
| AMIP `ic_0002`, `AMIP-p4K`, `AMIP-p2K`, ramped `ic_0003` | see `generate_paper_configs.AMIP_VARIANTS` / `RAMPED_DATASET` | prescribed-SST kinds | ✅ (p2k/p4k copied in July via `amip_p2k_p4k_transfer.yaml`, not re-verified) |

## How to run

From this directory in the `fme` env, configs committed and pushed (gantry
clones HEAD; `--dry-run` skips that check and the `validate_config` pass).
`--kind` is required. `--arm a1 a2 a3` restricts to the norm-ablation cells;
`--arch`, `--regime`, `--run`, `--climate`, `--ic` narrow further;
`--skip-if-in-wandb` fills in only what has no finished wandb run.

```bash
# Regenerate after editing the generator (all kinds, or --kind ...)
python generate_paper_configs.py

# Preview job expansion without submitting
python submit_paper_jobs.py --kind eq-eval-sst amip-eval --arm a1 --dry-run

# Slab-ocean, cheap first: single-stage equilibrium, abrupt, ensembles
python submit_paper_jobs.py --kind eq-nospinup abrupt-10yr abrupt-ens 7day --arm a1 a2 a3
# Reference rows (one checkpoint, --data-only-run; default ace2-fm-nc-swin-v2-fm-a1)
python submit_paper_jobs.py --kind data-only abrupt-data-only amip-data-only
# Abrupt evaluators against SHiELD's abrupt runs (D1)
python submit_paper_jobs.py --kind abrupt-10yr-eval abrupt-10yr-eval-sst --arm a1 a2 a3
# Two-stage equilibrium with the D2 spin-up year
python submit_paper_jobs.py --kind eq --arm a1 a2 a3 --skip-if-in-wandb
# Prescribed-SST
python submit_paper_jobs.py --kind eq-eval-sst random-co2-eval --arm a1 a2 a3
python submit_paper_jobs.py --kind amip-eval amip-p4k amip-p2k --arm a1 a2 a3
# Figure-8 ensembles under prescribed SST (single-job kinds now; the D3 kind later)
python submit_paper_jobs.py --kind control-ens-eval-sst abrupt-ens-fixed-sst --arm a1 a2 a3
python submit_paper_jobs.py --kind abrupt-ens-eval-sst --arm a1 --ens-member 1 2 3   # after D3; 36/run
# 1000-year runs: long; one arch at a time
python submit_paper_jobs.py --kind eq-1000yr --arm a1 --arch nc-swin-v2
```

Per-run job counts: slab-ocean 56 (`eq` 20, `eq-nospinup` 20, `eq-1000yr` 4,
`abrupt-10yr*` 9, `abrupt-ens` 1, `7day` 2), prescribed-SST 28 now
(`eq-eval-sst` 20, `amip-*` 3, `random-co2-eval` 3, `control-ens-eval-sst` 1,
`abrupt-ens-fixed-sst` 1) + 36 `abrupt-ens-eval-sst` when D3 lands; data-only
23 now, +36 with D3. The submit script sees 33 fm/c96 runs (22 with
`--arm a1 a2 a3`), so everything is ~4000 jobs, ~2700 for the arms alone, of
which `abrupt-ens-eval-sst` is ~800: submit by kind and arm, and run that one
on a subset (`--run`, `--ens-member`).

Watch the first `eq-nospinup 1xCO2` run for SST drift: the slab is
forward-Euler at a daily step, untested here (paper was 6-hourly).

## Next steps

1. Submit in the order above, cheap kinds first; check the `eq-nospinup`
   1xCO2 SST drift before submitting `eq`, `eq-1000yr`.
2. When Spencer's 45x90 stores land: write the D3 processing config (clone
   `shield-som-abrupt4xCO2-ensemble-c96-1deg-8layer.yaml` to 4deg + daily
   `time_coarsen`, as `shield-som-abrupt-co2-increase-c96-4deg-8layer.yaml`
   did for D1) and the 3xCO2 `ic_0003-0005` processing (new-dated store; ask
   Spencer whether he runs the argo step). Then name the store in
   `MISSING_DATASETS["abrupt-ensemble"]`, flip `available`, extend
   `SOM_MEMBERS["3xCO2"]`, regenerate, submit `abrupt-ens-data-only`,
   `abrupt-ens-eval-sst` and the three new `data-only` jobs.
3. Analysis: the AMIP windows (discard 1979; 1980-2011 train/validate;
   2012-2020 test) are cut from the single `amip-eval` runs at analysis time.

## Decisions log

- Models: fm and c96 regimes (`submit_paper_jobs.REGIMES`); `--arm` narrows to
  norm-ablation cells. era5-regime cells are mechanically able to run every
  kind (all runs have SST in `in_names`/`out_names` and predict the surface
  fluxes) but are excluded by default as fully out-of-distribution; add
  `"era5"` to `REGIMES` to include them. Hand-written `nc-sfno-c96-v1/v2`
  never saw SOM data.
- Checkpoint: `best_inference_ckpt.tar` only.
- Slab `interpolate: false` (training default), not the paper's `true`. Same
  for the prescribed-SST kinds, which carry no `stepper_override` at all.
- Paper CO2 constants for overrides (1x 0.00036343, 2x 0.00072686,
  3x 0.00109029, 4x 0.0014537); equilibrium runs read CO2 from the member.
- Members: 1x/2x/4x `ic_0005`, 3x `ic_0002` (paper). "Held out" means held
  out of the A-cells: they trained on SOM `ic_0001` of 1x/2x/4x, AMIP
  `ic_0001`, ramped `.zarr` + `ic_0002`. The hand-written `fm-random-v1/v3`
  and `fm-0.x-v1` runs trained on 3xCO2 `ic_0001-2`, AMIP `ic_0002` and ramped
  `ic_0003` as well, so those members are in-sample for them.
- Staggered ICs are separate jobs, one day apart (paper: 6 h). `eq-eval-sst`
  follows the `eq` IC protocol so it is a like-for-like control.
- Prescribed-SST runs are evaluators, not the paper's inference + data-only
  pair: under prescribed SST the SHiELD member is a valid step-by-step
  target, so the evaluator gives the comparison directly. Data-only rows are
  still produced for the paper-format reference diagnostics.
- AMIP as one run, not the paper's three chained stages (spin-up /
  train-validate / test via restart files): the trajectory is identical, the
  evaluator writes no restart, and the windows are an analysis cut. The
  paper's 5 AMIP ICs are five identical start times, i.e. one run for a
  deterministic model.
- Labels follow the data: `som`, `amip`, `ramped`. `AMIP-p4K`/`p2K` never
  appeared in training; `amip` is the closest label and what the SST sweep
  (`submit_sst_jobs.py`, forcing SST +2/+4 K) implies.
- Figure 8 under prescribed SST (2026-09-16): `abrupt-ens-eval-sst` needs
  SHiELD's own 4xCO2 SST, hence one job per D3 member (36/run, 89 steps, CO2
  from the member) rather than the slab version's one 36-IC job.
  `control-ens-eval-sst` (1x member as is) supplies the 1xCO2 target line;
  `abrupt-ens-fixed-sst` (1x SST, CO2 → 4x) isolates the direct CO2 response
  and has no SHiELD counterpart. No aggregator override, as the paper's
  ensemble evaluator config.
- `amip-constant-co2` skipped: eval suites already run AMIP `ic_0001` with
  constant CO2. `amip-eval` on `ic_0002` is kept because the suites never
  touch a held-out AMIP member.
- D1 recomputed from raw with the current pipeline rather than coarsening the
  2024-08-14 six-hourly store (which lacks `total_frozen_precipitation_rate`
  and `PRMSL` and would fail the SOM `time_coarsen` name lists).
- Outputs: paper-exact (daily PRATEsfc zarr for 1x/3x equilibrium, data-only
  and the AMIP runs; monthly netCDF of 11 surface vars for abrupt; nothing for
  the ensembles and random-CO2), all to `/results`.
- Naming: generator/submitter renamed from `*_som_*` to `*_paper_*`, config
  prefix `ace-paper-`, wandb group `ace2-fm-paper-2026-06-26`, before any job
  was submitted under the old names. `run-ace-som-two-stage.sh` keeps its
  name (SOM-specific).
- Job volume is the main risk: `--kind` is required; all kinds x all runs is
  ~2700 jobs (see "How to run").
