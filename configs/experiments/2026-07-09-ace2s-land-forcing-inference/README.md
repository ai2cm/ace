# ACE2S land-forcing short-lead evaluation

Short (`n_forward_steps: 40`, 6-hourly ≈ 10 days) `fme.ace.evaluator` runs that measure whether
giving ACE2S **persistent land-surface state as forcing** (snow cover, soil moisture) improves
**near-surface skill** in the first few days after initialization, versus the deployed **control**
ACE2S checkpoints. Because true land state is fed in as forcing, it leaks future information — so
this is a **short-lead** question only; long rollouts would be meteorologically confounded. The
treatment checkpoints come from `exp/ace2s-land-forcing-training`
(`configs/experiments/2026-07-06-ace2s-land-forcing-training/`).

## Design

Each job rolls the model 40 steps from 100 ICs (3 interleaved chunks per checkpoint = 300 ICs
total), computes skill vs the target (RMSE/bias vs lead
time, logged to W&B), and writes paired **prediction + target** fields to one GCS zarr per run.
Both prediction and target flow through the same derived-variable computation, so
`surface_evaporative_fraction` and `implied_surface_albedo` are skill-scored on both sides.

Treatments are the **1-step pretrain** checkpoints, **not** the multi-step finetuned ones:
multi-step finetuning trains over long rollouts, which for a land-forcing-fed model means fitting
against leaked future forcing — invalid here. Within the pretrain job we use **`best_ckpt.tar`**
(lowest *validation* loss — the short-horizon criterion, and exactly what finetuning would have
warm-started from), **not** `best_inference_ckpt.tar` (selected on a 7300-step rollout, i.e. the
confounded long-rollout regime). With `validate_using_ema: true` both files store EMA-applied
weights, so this choice only changes the selection criterion, not raw-vs-EMA. Controls use their
as-deployed `best_inference_ckpt.tar`.

One evaluator config per dataset (`era5.yaml`, `cm4.yaml`) is reused across checkpoints — the
forcing variables live in each checkpoint's `in_names`, so the loader auto-supplies them and the
same config works for control and treatment. `run-inference.sh` mounts each checkpoint at
`/ckpt.tar` and sets the per-run `experiment_dir` via `--override`.

## Variables written (each as `_prediction` and `_target`)
`surface_temperature`, `air_temperature_7`, `TMP2m`, `PRATEsfc`,
`total_frozen_precipitation_rate`, `surface_evaporative_fraction`, `implied_surface_albedo`, plus
the derived vars' raw flux components (`LHTFLsfc`, `SHTFLsfc`, `USWRFsfc`, `DSWRFsfc`) for
independent reconstruction / QC.

## Runs

| config | checkpoint | checkpoint file | job name |
|---|---|---|---|
| `era5.yaml` | ERA5 control (`01KSVC6YS7C18SGYV4VPZYZ232`) | `best_inference_ckpt.tar` | `lf-eval-era5-control` |
| `era5.yaml` | era5-snow (1-step pretrain, **fill in**) | `best_ckpt.tar` | `lf-eval-era5-snow` |
| `era5.yaml` | era5-soil (1-step pretrain, **fill in**) | `best_ckpt.tar` | `lf-eval-era5-soil` |
| `cm4.yaml`  | CM4 control rs0 (`01KTYXNSJX90Y5E2CQ6SV8K37D`) | `best_inference_ckpt.tar` | `lf-eval-cm4-control-rs0` |
| `cm4.yaml`  | cm4-snow (1-step pretrain, **fill in**) | `best_ckpt.tar` | `lf-eval-cm4-snow` |
| `cm4.yaml`  | cm4-soil (1-step pretrain, **fill in**) | `best_ckpt.tar` | `lf-eval-cm4-soil` |

Each row expands to **3 interleaved chunk-jobs** `<job>-c{0,1,2}` (100 ICs each; **18 jobs total**);
concatenate the 3 zarrs per checkpoint on the sample axis downstream. Treatment checkpoint IDs are
set in `run-inference.sh` (see Inputs).

```bash
./run-inference.sh              # submit all 18 (6 checkpoints x 3 chunks)
./run-inference.sh era5         # filter by substring (dataset/checkpoint)
./run-inference.sh -c0          # only chunk 0 across all checkpoints
```

## Inputs

| | value |
|---|---|
| ERA5 data (IC + forcing + target) | `/climate-default/2026-03-19-era5-1deg-8layer-1940-2025.zarr` (Weka) |
| CM4 data (IC + forcing + target) | `/climate-default/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr` (Weka) |
| Output prefix (GCS) | `gs://vcm-ml-intermediate/2026-07-09-ace2s-land-forcing-inference/<job>` |
| Control ERA5 | beaker `01KSVC6YS7C18SGYV4VPZYZ232` |
| Control CM4-piControl rs0 / rs1 | beaker `01KTYXNSJX90Y5E2CQ6SV8K37D` / `01KTWGH2VEZ4DNXXF1H5FTJK1S` |
| Treatment era5-snow (1-step pretrain `best_ckpt.tar`) | beaker `01KX47AYFXZR5GBP5236ZQ2H8G` |
| Treatment era5-soil | beaker `01KWZDSCZ69NPBR39H1JS946RF` |
| Treatment cm4-snow | beaker `01KX017K0Z48ZW91769NF7MQXH` |
| Treatment cm4-soil | beaker `01KX1RVPA4YQYZK9NK76ZEPWMJ` |

The treatment IDs are the most-recent *committed* result dataset of each 1-step-pretrain
experiment (`ace2s-lf-<model>-pretrain`); result-dataset IDs are ULIDs, so they sort in retry
order.

## Train/val/test split (out-of-sample rationale)

ICs are drawn from each dataset's **holdout** period, and the split is **identical for the
treatment and control checkpoints**, so the IC window is out-of-sample for every model compared.

| dataset | train | val | **holdout (ICs)** |
|---|---|---|---|
| ERA5 | ≤1995, 2011–2019, ≥2021 | 1996–1997 | **1998–2010 (+2020)** |
| CM4  | ≤0306 | 0306–0311 | **0311–0351** |

## Notes / things to confirm

- **IC selection**: integer `start_indices` (`n_initial_conditions`/`first`/`interval`) into each
  dataset's time axis (indices verified against the GCS mirrors: ERA5 record
  `1940-01-01T12:00`→`2025-12-31T18:00`, 125646 steps; CM4 noleap `0151-01-01T06:00`→`0351-01-01T00:00`,
  292000 steps). Each config holds **chunk 0** (100 ICs, `first`=holdout start, per-chunk
  `interval`); `run-inference.sh` runs 3 chunks by shifting `first` one dense stride each, so the
  chunks **interleave** to tile the holdout at the dense stride. ERA5: dense stride 15 days,
  `first` 84738/84798/84858 (=1998-01-01/-16/-31), per-chunk `interval` 180 steps (45 days), chunk-2
  last IC `2010-04-13` (before the 2011 train block). CM4: dense stride 48 days, `first`
  233600/233792/233984 (=0311-01-01/-02-18/-04-07), per-chunk `interval` 576 steps (144 days;
  neither divides 365, so no seasonal aliasing), chunk-2 last IC `0350-04-28`. Per-chunk interval
  > the 40-step window (no overlap) and non-commensurate with the year, so ICs precess through the
  seasons.
- **GPU memory lever**: peak GPU memory scales with the IC batch `n_initial_conditions`
  (× `forward_steps_in_memory`, held at 1). Runs on titan B200s (~178 GiB usable). Measured from
  the evaluator runs themselves (W&B "GPU Memory Allocated"): at **150 ICs**, CM4 peaked at
  **99.4%** (barely survived) and ERA5 **OOMed** (ERA5 is slightly heavier per IC — a real, if
  small, dataset difference). Memory is **flat over the rollout** (no growth with lead time), so it
  is set by the batch, not `n_forward_steps` — confirming no aggregator accumulation. Both datasets
  therefore use **`n_initial_conditions: 100` per job** (≈66% CM4 / ≈68% ERA5 of the ~150-IC
  ceiling), and the 300-IC target is recovered by running 3 such chunks per checkpoint (see IC
  selection). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (launcher) reduces fragmentation.
  Earlier single-job 300-IC and 150-IC(ERA5) batches OOMed.
- **Shared memory (`/dev/shm`)**: the IC load batches all ICs at once (~5–6 GiB at 100) and the
  DataLoader workers double-buffer it, so the launcher requests `--shared-memory 400GiB` (the
  land-feedback recipe's 50 GiB is too small at 300 ICs and caused worker bus errors). Kept to
  `num_data_workers: 4` to bound the concurrent worker footprint, which also eases co-location when
  several single-GPU jobs share a node. If shm errors persist, lower `num_data_workers` further.
- **Recipe caveat**: treatments are 1-step pretrain models while the controls are the deployed
  multi-step-finetuned checkpoints, so a skill delta blends the land-forcing effect with the
  training-recipe difference. A 1-step control would isolate land forcing if one is available.
- **Checkpoint-selection asymmetry (accepted)**: treatments use `best_ckpt.tar` (best validation)
  while controls use `best_inference_ckpt.tar` (best long-rollout inference). Strictly matching
  would require the controls' own 1-step-pretrain `best_ckpt.tar`; that's deliberately not pursued
  here as the effect on short-lead skill is expected to be minor.
- **ERA5 holdout is small** (1998–2010 + 2020, ~14 yr); 300 ICs (3×100) across 1998–2010 are dense.
- **Code dependency**: the `surface_evaporative_fraction` / `implied_surface_albedo` derived
  variables come from commit `c97b4bcc6` (branch `feature/surface-ef-isa-derived-variables`),
  cherry-picked onto this branch. Drop the cherry-pick once it lands on main.
