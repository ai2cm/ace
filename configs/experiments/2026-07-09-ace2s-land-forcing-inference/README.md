# ACE2S land-forcing short-lead evaluation

Short (`n_forward_steps: 40`, 6-hourly ≈ 10 days) `fme.ace.evaluator` runs that measure whether
giving ACE2S **persistent land-surface state as forcing** (snow cover, soil moisture) improves
**near-surface skill** in the first few days after initialization, versus the deployed **control**
ACE2S checkpoints. Because true land state is fed in as forcing, it leaks future information — so
this is a **short-lead** question only; long rollouts would be meteorologically confounded. The
treatment checkpoints come from `exp/ace2s-land-forcing-training`
(`configs/experiments/2026-07-06-ace2s-land-forcing-training/`).

## Design

Each run rolls the model 40 steps from ~300 ICs, computes skill vs the target (RMSE/bias vs lead
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

```bash
./run-inference.sh              # submit every run with a set checkpoint ID
./run-inference.sh era5         # or filter by substring
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

- **IC selection**: explicit datestrings (`start_indices.times`), generated from the datasets'
  verified time axes (read from the GCS mirrors: ERA5 record `1940-01-01T12:00`→`2025-12-31T18:00`,
  125646 steps; CM4 noleap `0151-01-01T06:00`→`0351-01-01T00:00`, 292000 steps). 300 ICs each:
  ERA5 every 15 days from `1998-01-01T00:00` (last `2010-04-13`, before the 2011 train block); CM4
  every 48 days from `0311-01-01T06:00` (last `0350-04-28`). Spacing > the 40-step window (no
  overlap) and non-commensurate with the annual cycle, so ICs precess through the seasons
  (≈24–27 ICs/month). Regenerate via the `xr.open_zarr(...).time` + fixed-stride recipe if the
  count/window changes.
- **Memory lever**: peak ≈ `n_initial_conditions × forward_steps_in_memory` (300 × 2 = 600
  in-flight IC-steps). If a job OOMs, drop `forward_steps_in_memory` to 1 or reduce ICs.
- **Recipe caveat**: treatments are 1-step pretrain models while the controls are the deployed
  multi-step-finetuned checkpoints, so a skill delta blends the land-forcing effect with the
  training-recipe difference. A 1-step control would isolate land forcing if one is available.
- **Checkpoint-selection asymmetry (accepted)**: treatments use `best_ckpt.tar` (best validation)
  while controls use `best_inference_ckpt.tar` (best long-rollout inference). Strictly matching
  would require the controls' own 1-step-pretrain `best_ckpt.tar`; that's deliberately not pursued
  here as the effect on short-lead skill is expected to be minor.
- **ERA5 holdout is small** (1998–2010 + 2020, ~14 yr); 300 ICs across 1998–2010 are fairly dense.
- **Code dependency**: the `surface_evaporative_fraction` / `implied_surface_albedo` derived
  variables come from commit `c97b4bcc6` (branch `feature/surface-ef-isa-derived-variables`),
  cherry-picked onto this branch. Drop the cherry-pick once it lands on main.
