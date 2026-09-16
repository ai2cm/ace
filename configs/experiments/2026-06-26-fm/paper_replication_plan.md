# Plan: replicating the ACE2S-SHiELD+ paper experiments on the FM runs

Living plan for reproducing the ACE experiments of
[ai2cm/ace2s-shield-plus-paper](https://github.com/ai2cm/ace2s-shield-plus-paper)
("Disentangling the effects of sea surface temperature and CO2 in global
machine learned weather-climate emulators", arXiv 2606.07928) on the 4deg daily
FM / c96 training runs of this directory. Machinery lives in
`generate_paper_configs.py` (config generator, one "kind" per experiment type),
`submit_paper_jobs.py` (gantry submission), `run-ace-evaluator.sh`,
`run-ace-som-two-stage.sh`; dataset gaps in `MISSING_DATASETS.md`; vocabulary
in `CONTEXT.md` ("Slab-ocean experiments").

Resume with a fresh agent: read this file, `MISSING_DATASETS.md`, and the
docstrings of `generate_paper_configs.py` / `submit_paper_jobs.py`, then check the
status section below against `argo list` and the GCS/weka paths.

## Status (2026-09-16)

- Commits on `exp/alexeyfm`: `c9bda231b` (generator, configs, docs),
  `7a6cc9751` (submit + run scripts), `0ff67e8f5` + `bc8473bbc` (D1 processing
  config, prescribed-SST abrupt kind), `0746cf5b9` + `c5889ca08` (D2 processing
  config). Not pushed at time of writing.
- Argo (cluster `gke_vcm-ml_us-central1-c_ml-cluster-dev`, submit from this
  machine with `argo`, needs `brew install python-yq`):
  - `compute-fme-dataset-ensemble-xwpb9` — D1 abrupt-CO2 4deg (6-hourly +
    daily), 3 pods, running.
  - `compute-fme-dataset-ensemble-26c2p` — D2 spin-up 4deg (6-hourly + daily),
    4 pods, running.
- Spencer is regridding to 45x90 (Gaea/Snakemake, not argo): the 36-member
  abrupt-4xCO2 ensemble (D3) and 3xCO2 `ic_0003-0005`. Neither has landed in
  `gs://vcm-ml-raw-flexible-retention` yet.

## Experiment inventory

Legend: ✅ data on weka · ⏳ argo running, then weka copy + `available=True` ·
🚧 blocked on Spencer · ❌ not produced / not planned · 📝 kind not written yet.

### Slab-ocean (SST-interactive) — kinds exist

| Paper script | Our kind | Data (4deg daily) | Status |
|---|---|---|---|
| `run-ace-equilibrium-climate-inference.sh` (spin-up 2030 → 10 yr main, 4 climates x 5 ICs) | `eq` | SOM ensemble ✅ · D2 spin-up ⏳ | ⏳ `26c2p` |
| — single-stage variant (ours) | `eq-nospinup` | SOM ensemble ✅ | ✅ runnable |
| `run-ace-1000-year-equilibrium-climate-inference.sh` | `eq-1000yr` | SOM 1x member tiled ✅ | ✅ runnable |
| `run-ace-data-only-equilibrium-climate-evaluator.sh` | `data-only` | SOM ensemble ✅ (3x has `ic_0001-2` only) · 3x `ic_0003-5` 🚧 | ✅ 17 jobs; +3 🚧 |
| `run-ace-abrupt-4xCO2-evaluator.sh` | `abrupt-10yr-eval` (2x/3x/4x) | D1 ⏳ | ⏳ `xwpb9` |
| — prescribed-SST variant (ours) | `abrupt-10yr-eval-sst` | D1 ⏳ | ⏳ `xwpb9` |
| — free inference variant (ours) | `abrupt-10yr` | SOM 1x member ✅ | ✅ runnable |
| `run-ace-abrupt-4xCO2-data-only-evaluator.sh` | `abrupt-data-only` | D1 ⏳ | ⏳ `xwpb9` |
| `run-ace-abrupt-4xCO2-ensemble-evaluator.sh` (36 monthly ICs x 90 d) | `abrupt-ens` | SOM 1x member ✅ | ✅ runnable |
| `run-ace-abrupt-4xCO2-ensemble-data-only-evaluator.sh` | `abrupt-ens-data-only` | D3 🚧 | 🚧 |
| `run-seven-day-1xCO2-and-abrupt-4xCO2-inference-ensemble.sh` | `7day` | SOM 1x member ✅ | ✅ runnable |
| `run-ace-2pctCO2-inference.sh`, `run-ace-deterministic-2pctCO2-ensemble.sh`, `run-ace-2pctCO2-data-only-evaluator.sh` | 📝 `2pct*` | D2 ⏳ · D4 increasing-CO2 daily ❌ (6-hourly 4deg `2024-07-16-…` exists, old pipeline) | ❌ not planned |

Paper figures 8 and 10 (90-day abrupt-4xCO2 response) need `abrupt-ens`
(ACE lines and the SHiELD 1xCO2 dashed line, from the same job) and
`abrupt-ens-data-only` (SHiELD 4xCO2 line, D3).

### Prescribed-SST — TO ADD (decided 2026-09-16, not yet designed)

All 4deg daily inputs exist on GCS (`gs://vcm-ml-intermediate/`); weka
presence to verify per store. No kinds written yet; design questions below.

| Paper script | Proposed kind | Data (4deg daily, on GCS) | Notes |
|---|---|---|---|
| — (ours; control for `eq`) | 📝 `eq-eval-sst` | SOM ensemble ✅ | Evaluator on each climate's member with SST prescribed, no slab. Separates atmospheric error from slab-feedback error in `eq`. |
| `run-ace-split-amip-ensemble-inference.sh`, `run-ace-single-member-split-amip-inference-daily-PRATEsfc.sh` | 📝 `amip-split` | `2026-01-28-…-c96-4deg-daily-shield-amip-ensemble-dataset/ic_0002.zarr` (held out; training used `ic_0001`) | Paper: 3 chained stages (spin-up 1979, train/validate 1980-2011, test 2012-2020), 5 ICs. Overlaps existing eval suites (`long_43year_*`, single stage on `ic_0001`). Decide whether the split protocol adds value. |
| `run-ace-amip-constant-CO2-inference.sh` | 📝 `amip-constant-co2` | `2026-07-01-…-c96-4deg-daily-shield-amip-constant-co2-dataset/AMIP-constant-CO2.zarr` | Already covered by eval-suite entries `*_constant_co2`; likely skip. |
| `run-ace-split-amip-plus-4K-inference.sh` (+ daily PRATEsfc) | 📝 `amip-p4k` (and `amip-p2k`) | `2026-07-09-…-c96-4deg-daily-shield-amip-p4k-dataset/AMIP-p4K.zarr`, `…-p2k-…/AMIP-p2K.zarr` | Evaluator vs SHiELD's own +4K run. Differs from the SST sweep (`submit_sst_jobs.py`), which perturbs forcing SST by +4 K and has no SHiELD target. |
| `run-ace-amip-split-data-only-evaluator.sh`, `run-ace-amip-variant-data-only-evaluator.sh` | 📝 `amip-data-only` | AMIP `ic_0001/0002`, `AMIP-p4K`, `AMIP-constant-CO2` | Reference rows. |
| `run-ace-random-CO2-evaluator.sh` | 📝 `random-co2-eval` | `2026-06-08-…-ramped-climSST-random-CO2-…-4deg-daily/ramped-sst-{1x,2x,4x}CO2-random-perturbation-ic_0003.zarr` (held out; training used `.zarr` + `ic_0002`) | Paper: 7675 six-hourly steps → 1919 daily. Earlier note that `ic_0003` was missing at 4deg was wrong; it exists. |

Design questions to settle before writing these kinds:

1. Which of the above are worth running given the eval suites already cover
   AMIP `ic_0001` (varying and constant CO2) at 4deg? Leading candidates:
   `eq-eval-sst`, `amip-p4k`/`amip-p2k`, `random-co2-eval`, `amip-data-only`.
2. Inference (paper) vs evaluator mode for the AMIP runs. Evaluator gives
   metrics vs the SHiELD member for free; the paper used inference plus
   separate data-only evaluators.
3. Paper's three-stage AMIP protocol (spin-up / train-validate / test windows)
   vs one 1979-2020 run with the windows split in analysis.
4. Labels: `amip` for AMIP-family data, `ramped` for the random-CO2 data,
   `som` for `eq-eval-sst` (all three regimes' vocabularies include them).
5. Where the kinds live: extend `generate_paper_configs.py` (rename to a
   paper-wide generator) or a sibling `generate_prescribed_sst_configs.py`
   sharing the submit script.

## Datasets to produce

See `MISSING_DATASETS.md` for full detail.

| | Path (under `/climate-default/`) | Unblocks | Status |
|---|---|---|---|
| D1 | `2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-abrupt-co2-increase-fme-dataset/abrupt-{2x,3x,4x}CO2.zarr` | `abrupt-10yr-eval`, `abrupt-10yr-eval-sst`, `abrupt-data-only` | ⏳ argo `xwpb9` |
| D2 | `2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-ensemble-spin-up-fme-dataset/{climate}-spin-up-ic_000N.zarr` | `eq` | ⏳ argo `26c2p` |
| D3 | `TBD-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset/abrupt4xCO2-ic_00NN.zarr` | `abrupt-ens-data-only` | 🚧 Spencer regrid; processing config not written |
| 3x members | new-dated SOM ensemble store with 3xCO2 `ic_0003-0005` | 3 more `data-only` jobs | 🚧 Spencer regrid; config not written |
| D4 | daily increasing-CO2 (not planned) | `2pct*` | ❌ |

## Next steps

1. Push `exp/alexeyfm`.
2. When `xwpb9` / `26c2p` finish:
   ```bash
   python scripts/data_process/copy_zarrs_to_weka.py gs://vcm-ml-intermediate/2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-abrupt-co2-increase-fme-dataset
   python scripts/data_process/copy_zarrs_to_weka.py gs://vcm-ml-intermediate/2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-ensemble-spin-up-fme-dataset
   ```
   then set `available=True` on `MISSING_DATASETS["abrupt"]` / `["spin-up"]`
   in `generate_paper_configs.py`, commit, push.
3. Submit (from this directory, fme env; `--dry-run` first):
   ```bash
   python submit_paper_jobs.py --kind eq-nospinup abrupt-10yr abrupt-ens 7day --arm a1 a2 a3
   python submit_paper_jobs.py --kind data-only
   python submit_paper_jobs.py --kind abrupt-10yr-eval-sst abrupt-10yr-eval --climate 4xCO2 --arm a1 a2 a3   # after D1
   python submit_paper_jobs.py --kind abrupt-data-only --climate 4xCO2                                     # after D1
   python submit_paper_jobs.py --kind eq --arm a1 a2 a3 --skip-if-in-wandb                                 # after D2
   python submit_paper_jobs.py --kind eq-1000yr --arm a1 --arch nc-swin-v2                                 # long jobs
   ```
   Watch the first `eq-nospinup 1xCO2` run for SST drift: the slab is
   forward-Euler at a daily step, untested here (paper was 6-hourly).
4. When Spencer's 45x90 stores land: write the D3 processing config (clone
   `shield-som-abrupt4xCO2-ensemble-c96-1deg-8layer.yaml` to 4deg + daily
   `time_coarsen`) and the 3xCO2 `ic_0003-0005` processing (new-dated store;
   ask Spencer whether he runs the argo step), then extend
   `SOM_MEMBERS["3xCO2"]` and regenerate `data-only`.
5. Design and add the prescribed-SST kinds (questions above), via a grilling
   session.

## Decisions log

- Models: fm and c96 regimes (`submit_paper_jobs.REGIMES`); `--arm` narrows to
  norm-ablation cells. era5-regime cells are mechanically able to run every
  kind (all runs have SST in `in_names`/`out_names` and predict the surface
  fluxes) but are excluded by default as fully out-of-distribution; add
  `"era5"` to `REGIMES` to include them. Hand-written `nc-sfno-c96-v1/v2`
  never saw SOM data.
- Checkpoint: `best_inference_ckpt.tar` only.
- Slab `interpolate: false` (training default), not the paper's `true`.
- Paper CO2 constants for overrides (1x 0.00036343, 2x 0.00072686,
  3x 0.00109029, 4x 0.0014537); equilibrium runs read CO2 from the member.
- Members: 1x/2x/4x `ic_0005`, 3x `ic_0002` (paper). A-cells trained on
  `ic_0001` of 1x/2x/4x only; 3x is out-of-sample for them (not for
  `fm-random-v3`, `fm-0.x-v1`, which trained on 3x `ic_0001-2`).
- Staggered ICs are separate jobs, one day apart (paper: 6 h).
- D1 recomputed from raw with the current pipeline rather than coarsening the
  2024-08-14 six-hourly store (which lacks `total_frozen_precipitation_rate`
  and `PRMSL` and would fail the SOM `time_coarsen` name lists).
- Outputs: paper-exact (daily PRATEsfc zarr for 1x/3x equilibrium and
  data-only; monthly netCDF of 11 surface vars for abrupt; nothing for the
  ensembles), all to `/results`.
- Job volume is the main risk: `--kind` is required; all kinds x all runs is
  ~1750 jobs.
