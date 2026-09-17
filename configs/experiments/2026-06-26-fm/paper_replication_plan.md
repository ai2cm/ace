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

## Status (2026-09-16, end of day)

- 25 kinds written, all validated; every kind but the two on D3
  (`som-abrupt-4xCO2-ens-data-only`, `som-abrupt-4xCO2-ens-sst-eval`) is
  submittable. No experiment jobs submitted yet; wandb group
  `ace2-fm-paper-2026-06-26` is empty.
- Commits on `exp/alexeyfm` today: `593aa955e` (rename to paper-wide names),
  `792e0788c` (AMIP / ramped / eq prescribed-SST kinds), `62f3366e6` (D1/D2
  available), `c4159803c` (figure-8 ensemble kinds), then the kind-naming
  scheme with the ERA5 and fixed-SST abrupt kinds, plus docs. Pushed.
- Argo (`gke_vcm-ml_us-central1-c_ml-cluster-dev`): `xwpb9` (D1) and
  `26c2p` (D2) succeeded. Seven `gcs-to-weka-*` gantry copy jobs
  (`01M2P5FQ…` … `01M2P5GA…`, workspace `ai2/climate-titan`) finished with
  exit 0; D1 and D2 are on weka.
- Spencer's 45x90 regrids (Gaea/Snakemake): the 36-member abrupt-4xCO2
  ensemble (D3) and 3xCO2 `ic_0003-0005`. Nothing in
  `gs://vcm-ml-raw-flexible-retention` yet.

## Kind naming

```
{data}-{experiment}[-{co2}]-{shape}-{ocean}-{mode}
```

| Token | Values | Meaning |
|---|---|---|
| data | `som` `amip` `ramped` `era5` | forcing / reference store; also the config's label |
| experiment | `eq` `eq-nospinup` `abrupt` `control` `7day` `p4k` `p2k` `random-co2` | what is done to the forcing |
| co2 | `4xCO2` | present whenever CO2 is overwritten with a constant; abrupt kinds are 4x only, as in the paper |
| shape | `10yr` `1000yr` `ens` (36 monthly ICs × 90 d) `7day` (36 ICs × 7 d) | only where the paper has several shapes of one experiment |
| ocean | `slab` · `sst` · `sst-fixed` | see below |
| mode | `inference` (free, no target) · `eval` (evaluator vs reference) · `data-only` (reference vs itself; no ocean token) | |

Ocean, i.e. where SST comes from at inference (training always reads it from
the data; no training run ever ran a slab):

| | `slab` | `sst` | `sst-fixed` |
|---|---|---|---|
| SST at step 1 | from the store | from the store | from the control store |
| SST at steps 2…N | mixed-layer ocean integrating the model's own fluxes plus the SOM store's q-flux and depth (`stepper_override`, `interpolate: false`) | from the store, every step | from the control store, every step |
| CO2 | from the store | from the store | one constant (4x) from the first step |
| Reference | the store | the store | the control store |
| Measures | model error incl. its own surface response | atmospheric error given the store's SST | direct CO2 response with the surface held; no SHiELD counterpart |

`sst` is the training-time setup verbatim; `sst-fixed` is that setup with one
field overwritten; `slab` is the only mode that adds a mechanism the model
never saw. ERA5 has no slab fields, so ERA5 kinds are `sst` / `sst-fixed` only.

Job names are `{run}-{kind}[-{climate}][-ic{n}]` for per-run kinds and
`{kind}-{member}` for data-only kinds. Config files are
`run_configs/ace-paper-{kind}-config-4deg[-{parts}].yaml`.

## Experiment inventory

Legend: ✅ runnable (data on weka) · 🚧 blocked on Spencer's regrid ·
❌ not planned. "Paper" names the script in `ACE-experiments/inference`; "ours"
marks experiments the paper does not have.

### Slab-ocean (SHiELD-SOM)

| Paper script | Our kind | Data (4deg daily) | Window | Jobs / run | Label | Output | Status |
|---|---|---|---|---|---|---|---|
| `run-ace-equilibrium-climate-inference.sh` (spin-up 2030 → 10 yr main, 4 climates × 5 ICs) | `som-eq-10yr-slab-inference` | D2 spin-up member, then SOM paper member | 2030-01-01T06 + ic stagger, 365 steps; restart → 2031-01-01T06, 3652 steps | 20 (two-stage) | `som` | daily `PRATEsfc` zarr, 1x/3x | ✅ |
| — single-stage variant (ours) | `som-eq-nospinup-10yr-slab-inference` | SOM paper member | 2031-01-01T06 + ic stagger, `3652 - offset` steps | 20 | `som` | daily `PRATEsfc` zarr, 1x/3x | ✅ |
| `run-ace-1000-year-equilibrium-climate-inference.sh` | `som-eq-1000yr-slab-inference` | SOM 1x member tiled ×101, CO2 → climate | 2032-01-01T06, 365250 steps | 4 | `som` | none | ✅ (long) |
| `run-ace-data-only-equilibrium-climate-evaluator.sh` | `som-eq-10yr-data-only` | every SOM member vs itself (3x has `ic_0001-2` only) | 2031-01-01T06, 3652 steps | 17 total; +3 🚧 | `som` | daily `PRATEsfc` zarr, 1x/3x | ✅ |
| — free variant (ours) | `som-abrupt-4xCO2-10yr-slab-inference` | SOM 1x member, CO2 → 4x | 2031-01-01T06, 3652 steps | 1 | `som` | monthly netCDF | ✅ |
| `run-ace-abrupt-4xCO2-evaluator.sh` | `som-abrupt-4xCO2-10yr-slab-eval` | D1 `abrupt-4xCO2` | 2020-01-01T06, 3651 steps | 1 | `som` | monthly netCDF | ✅ |
| `run-ace-abrupt-4xCO2-data-only-evaluator.sh` | `som-abrupt-4xCO2-10yr-data-only` | D1 vs itself | same | 1 total | `som` | monthly netCDF | ✅ |
| `run-ace-abrupt-4xCO2-ensemble-evaluator.sh` | `som-abrupt-4xCO2-ens-slab-eval` | SOM 1x member, CO2 → 4x | 2031-01 … 2033-12 starts, 90 steps | 1 | `som` | none | ✅ |
| `run-ace-abrupt-4xCO2-ensemble-data-only-evaluator.sh` | `som-abrupt-4xCO2-ens-data-only` | D3 member vs itself | 89 steps from each member's start | 36 total (`--ens-member`) | `som` | none | 🚧 |
| `run-seven-day-1xCO2-and-abrupt-4xCO2-inference-ensemble.sh` | `som-control-7day-slab-inference`, `som-abrupt-4xCO2-7day-slab-inference` | SOM 1x member; CO2 as is / → 4x | 2031-01 … 2033-12 starts, 7 steps | 1 + 1 | `som` | none | ✅ |
| `run-ace-2pctCO2-*.sh` | — | D4 increasing-CO2 daily | — | — | — | — | ❌ |

D1 also holds `abrupt-2xCO2` and `abrupt-3xCO2`; no kind uses them (paper is
4x only).

### Prescribed-SST (SHiELD)

Aggregator `log_zonal_mean_images: false` (paper) and `forward_steps_in_memory:
40` for the long runs; the 36-IC ensemble kinds use the paper's ensemble
evaluator settings (default aggregator, `forward_steps_in_memory: 1`).

| Paper script | Our kind | Data (4deg daily) | Window | Jobs / run | Label | Output | Status |
|---|---|---|---|---|---|---|---|
| — control for the equilibrium runs (ours) | `som-eq-10yr-sst-eval` | SOM paper member per climate | 2031-01-01T06 + ic stagger, `3652 - offset` steps, 5 ICs | 20 | `som` | daily `PRATEsfc` zarr, 1x/3x | ✅ |
| — abrupt evaluator with SHiELD's SST instead of the slab (ours) | `som-abrupt-4xCO2-10yr-sst-eval` | D1 `abrupt-4xCO2`, its SST and CO2 | 2020-01-01T06, 3651 steps | 1 | `som` | monthly netCDF | ✅ |
| — CO2 step with SST held at 1x (ours) | `som-abrupt-4xCO2-10yr-sst-fixed-eval` | SOM 1x member SST, CO2 → 4x, vs 1x member | 2031-01-01T06, 3652 steps | 1 | `som` | monthly netCDF | ✅ |
| — ensemble with SST from SHiELD's 4xCO2 members (ours) | `som-abrupt-4xCO2-ens-sst-eval` | D3 member per job, SST/sea ice/CO2 from it | 89 steps from each member's start | 36 (`--ens-member`) | `som` | none | 🚧 |
| — ensemble CO2 step with SST held at 1x (ours) | `som-abrupt-4xCO2-ens-sst-fixed-eval` | SOM 1x member SST, CO2 → 4x, vs 1x member | 2031-01 … 2033-12 starts, 90 steps | 1 | `som` | none | ✅ |
| — control ensemble (ours) | `som-control-ens-sst-eval` | SOM 1x member as is | same | 1 | `som` | none | ✅ |
| `run-ace-split-amip-ensemble-inference.sh`, `…-single-member-split-amip-inference-daily-PRATEsfc.sh` | `amip-sst-eval` | AMIP `ic_0002` (held out) | 1979-01-01T06, 15689 steps (to 2021-12-15) | 1 | `amip` | daily `PRATEsfc` zarr | ✅ |
| `run-ace-split-amip-plus-4K-inference.sh` (+ daily PRATEsfc) | `amip-p4k-sst-eval`, `amip-p2k-sst-eval` | `AMIP-p4K.zarr`, `AMIP-p2K.zarr`, IC from own 1979 state | same | 1 + 1 | `amip` | daily `PRATEsfc` zarr | ✅ |
| `run-ace-amip-split-data-only-evaluator.sh`, `run-ace-amip-variant-data-only-evaluator.sh` | `amip-data-only` | `ic_0002`, `AMIP-p4K`, `AMIP-p2K` vs themselves | 1980-01-01T06, 15324 steps | 3 total | `amip` | daily `PRATEsfc` zarr | ✅ |
| `run-ace-random-CO2-evaluator.sh` | `ramped-random-co2-sst-eval` | ramped `ic_0003` (held out), 1x/2x/4x | 2019-10-01T06, 1918 steps | 3 | `ramped` | none | ✅ |
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
| `run-ace-abrupt-4xCO2-evaluator.sh` transferred | `era5-abrupt-4xCO2-10yr-sst-fixed-eval` | ERA5 1940–2025, observed SST, CO2 → 1.5945e-3 | 2015-01-01T00, 3652 steps (the eval suites' `10year` window, so the control run exists already) | 1 | `era5` | monthly netCDF | ✅ |
| `run-ace-abrupt-4xCO2-ensemble-evaluator.sh` transferred | `era5-abrupt-4xCO2-ens-sst-fixed-eval` | same | 36 ICs, 1st of each month 2015-01 … 2017-12, 90 steps | 1 | `era5` | none | ✅ |
| — control ensemble (ours; mirrors `som-control-ens-sst-eval`) | `era5-control-ens-sst-eval` | ERA5, CO2 as observed | same | 1 | `era5` | none | ✅ |

### Figure 8 (and 10) under each ocean

90-day global-mean response to abrupt 4xCO2, ensemble mean over 36 monthly
ICs, metrics from wandb `inference/mean/weighted_mean_{gen,target}/*` as in
the paper notebook `figures-08-10.ipynb`.

| Line | slab (paper) | SHiELD prescribed | ERA5 prescribed |
|---|---|---|---|
| ACE 4xCO2 | `som-abrupt-4xCO2-ens-slab-eval` | `som-abrupt-4xCO2-ens-sst-eval` (SHiELD's SST, D3) and `som-abrupt-4xCO2-ens-sst-fixed-eval` (1x SST) | `era5-abrupt-4xCO2-ens-sst-fixed-eval` |
| SHiELD / ERA5 4xCO2 | `som-abrupt-4xCO2-ens-data-only` (D3) | target of `som-abrupt-4xCO2-ens-sst-eval` | none exists |
| 1xCO2 dashed | target of the slab job | target of `som-control-ens-sst-eval` | target of `era5-control-ens-sst-eval` |

Ensemble means over the 36 per-member jobs are taken at analysis time; the
single-job kinds already log the ensemble mean.

## Datasets

See `MISSING_DATASETS.md` for full detail.

| | Path (under `/climate-default/`) | Unblocks | Status |
|---|---|---|---|
| D1 | `2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-abrupt-co2-increase-fme-dataset/abrupt-{2x,3x,4x}CO2.zarr` | `som-abrupt-4xCO2-10yr-{slab-eval,sst-eval,data-only}` | ✅ on weka |
| D2 | `2026-09-16-vertically-resolved-4deg-daily-c96-shield-som-ensemble-spin-up-fme-dataset/{climate}-spin-up-ic_000N.zarr` | `som-eq-10yr-slab-inference` | ✅ on weka |
| D3 | `TBD-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-fme-dataset/abrupt4xCO2-ic_00NN.zarr` | `som-abrupt-4xCO2-ens-data-only`, `som-abrupt-4xCO2-ens-sst-eval` | 🚧 Spencer regrid; processing config not written |
| 3x members | new-dated SOM ensemble store with 3xCO2 `ic_0003-0005` | 3 more `som-eq-10yr-data-only` jobs | 🚧 Spencer regrid; config not written |
| D4 | daily increasing-CO2 | `2pct*` | ❌ not planned |
| AMIP `ic_0002`, `AMIP-p4K`, `AMIP-p2K`, ramped `ic_0003`, ERA5 | see `generate_paper_configs.AMIP_VARIANTS` / `RAMPED_DATASET` / `ERA5_DATASET` | prescribed-SST kinds | ✅ (p2k/p4k copied in July via `amip_p2k_p4k_transfer.yaml`, not re-verified) |

## How to run

From this directory in the `fme` env, configs committed and pushed (gantry
clones HEAD; `--dry-run` skips that check and the `validate_config` pass).
`--kind` is required. Each kind runs on the regimes that trained on its data
(SHiELD-data kinds: fm + c96; ERA5 kinds: fm + era5). `--arm a1 a2 a3`
restricts to the norm-ablation cells; `--arch`, `--regime`, `--run` narrow
further; `--climate` applies to `som-eq-*` and `ramped-random-co2-sst-eval`,
`--ic` to the staggered-IC kinds, `--ens-member` to the per-member D3 kinds;
`--skip-if-in-wandb` fills in only what has no finished wandb run.

```bash
# Regenerate after editing the generator (all kinds, or --kind ...)
python generate_paper_configs.py

# Preview job expansion without submitting
python submit_paper_jobs.py --kind som-eq-10yr-sst-eval amip-sst-eval --arm a1 --dry-run

# Prescribed-SST first (see decisions log): SHiELD ...
python submit_paper_jobs.py --kind som-abrupt-4xCO2-10yr-sst-eval som-abrupt-4xCO2-10yr-sst-fixed-eval --arm a1 a2 a3
python submit_paper_jobs.py --kind som-control-ens-sst-eval som-abrupt-4xCO2-ens-sst-fixed-eval --arm a1 a2 a3
python submit_paper_jobs.py --kind som-eq-10yr-sst-eval ramped-random-co2-sst-eval --arm a1 a2 a3
python submit_paper_jobs.py --kind amip-sst-eval amip-p4k-sst-eval amip-p2k-sst-eval --arm a1 a2 a3
# ... and ERA5
python submit_paper_jobs.py --kind era5-control-ens-sst-eval era5-abrupt-4xCO2-ens-sst-fixed-eval era5-abrupt-4xCO2-10yr-sst-fixed-eval --arm a1 a2 a3
# Reference rows (one checkpoint, --data-only-run; default ace2-fm-nc-swin-v2-fm-a1)
python submit_paper_jobs.py --kind som-eq-10yr-data-only som-abrupt-4xCO2-10yr-data-only amip-data-only
# Slab-ocean: cheap first, watch the 1xCO2 SST drift before the long ones
python submit_paper_jobs.py --kind som-eq-nospinup-10yr-slab-inference --arm a1 --climate 1xCO2 --ic 1
python submit_paper_jobs.py --kind som-abrupt-4xCO2-10yr-slab-inference som-abrupt-4xCO2-10yr-slab-eval som-abrupt-4xCO2-ens-slab-eval --arm a1 a2 a3
python submit_paper_jobs.py --kind som-control-7day-slab-inference som-abrupt-4xCO2-7day-slab-inference --arm a1 a2 a3
python submit_paper_jobs.py --kind som-eq-nospinup-10yr-slab-inference som-eq-10yr-slab-inference --arm a1 a2 a3 --skip-if-in-wandb
python submit_paper_jobs.py --kind som-eq-1000yr-slab-inference --arm a1 --arch nc-swin-v2
# After D3
python submit_paper_jobs.py --kind som-abrupt-4xCO2-ens-data-only
python submit_paper_jobs.py --kind som-abrupt-4xCO2-ens-sst-eval --arm a1 --ens-member 1 2 3   # 36/run
```

Per-run job counts on SHiELD data (53 fm/c96 runs, 41 with `--arm`, after
the nc-swin-v2.1 cells landed): slab-ocean 49 (`eq` 20, `eq-nospinup` 20,
`1000yr` 4, abrupt 10yr 2, ens 1, 7day 2), prescribed 30 (`eq` sst 20, abrupt
10yr 2, ens 2, amip 3, ramped 3) plus 36 `som-abrupt-4xCO2-ens-sst-eval` after
D3. On ERA5 (39 fm/era5 runs, 29 with `--arm`): 3. Data-only 21 now, +36 with
D3. Everything ≈ 4300 jobs now, ≈ 6300 with D3; the arms alone
≈ 3300. Submit by kind and arm; run the per-member D3 kind on a subset
(`--run`, `--ens-member`).

The slab is forward-Euler at a daily step, untested here (paper was 6-hourly):
run one `som-eq-nospinup-10yr-slab-inference` 1xCO2 job and check SST drift
before submitting the rest of the slab kinds.

## Next steps

1. Submit the prescribed-SST kinds (SHiELD, then ERA5), then the data-only
   rows. Slab kinds after the drift check.
2. When Spencer's 45x90 stores land: write the D3 processing config (clone
   `shield-som-abrupt4xCO2-ensemble-c96-1deg-8layer.yaml` to 4deg + daily
   `time_coarsen`, as `shield-som-abrupt-co2-increase-c96-4deg-8layer.yaml`
   did for D1) and the 3xCO2 `ic_0003-0005` processing (new-dated store; ask
   Spencer whether he runs the argo step). Then name the store in
   `MISSING_DATASETS["abrupt-ensemble"]`, flip `available`, extend
   `SOM_MEMBERS["3xCO2"]`, regenerate, submit the two D3 kinds and the three
   new `som-eq-10yr-data-only` jobs.
3. Analysis: the AMIP windows (discard 1979; 1980-2011 train/validate;
   2012-2020 test) are cut from the single `amip-sst-eval` runs at analysis
   time; figure-8 ensemble means over the per-member D3 jobs likewise.
4. Slab ocean on ERA5: needs mixed-layer depth and q-flux fields for ERA5,
   which do not exist. Spencer: "eventually".

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
- Naming scheme (see "Kind naming"), adopted 2026-09-16 before any job was
  submitted: kind names spell out data, experiment, CO2, shape, ocean and
  mode, so nothing is implied. Abrupt kinds are 4xCO2 only, as in the paper;
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
  `som-eq-10yr-sst-eval` follows the `eq` IC protocol so it is a like-for-like
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
- Figure 8 under prescribed SST: `som-abrupt-4xCO2-ens-sst-eval` needs
  SHiELD's own 4xCO2 SST, hence one job per D3 member (36/run, 89 steps, CO2
  from the member) rather than the slab version's one 36-IC job. The
  `sst-fixed` ensembles isolate the direct CO2 response and have no SHiELD
  counterpart. No aggregator override for the ensemble kinds, as the paper's
  ensemble evaluator config.
- `amip-constant-co2` skipped: eval suites already run AMIP `ic_0001` with
  constant CO2. `amip-sst-eval` on `ic_0002` is kept because the suites never
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
