# ACE2S land-feedback inference (experiments E1–E7)

Inference runs that generate ACE2S output for testing the **land–atmosphere feedback null
hypothesis** — that current ACE2S already reproduces the documented land feedbacks without new
land-state inputs. The hypothesis and the E1–E7 experiment definitions live in the land-atm
coupling analysis project (synthesis report); this directory holds only the **run recipes**. The
analysis/ingestion side (reading these zarrs, computing E1–E7 diagnostics) stays in that project.

## Design in one paragraph
Every run writes one **6-hourly, global** zarr to GCS with the same 10-variable set, so the output
is a drop-in third data `source` (alongside ERA5 and CM4 reanalysis/model data) for the analysis
pipeline. Two complementary frameworks:
- **Framework A** — long free-running rollout, 1 IC × 5 stochastic members. Climate statistics
  (E1 timescales, E2 hot-day composites, E3, E4 diurnal) + within-checkpoint stochastic noise floor.
- **Framework B** — initialized, 48 Jan-1 ICs × 2 stochastic members (96 samples, single job),
  1-year rollouts. Seasonal/initialized skill (E5 cold-season short-lead, E6 spring→summer).

Three checkpoints: 1× ACE2S-ERA5, 2× ACE2S-CM4-piControl (RS0/RS1, **seed-only difference** → an
empirical model noise floor). Combined with the inference-stochastic member spread, the analysis
pass-bands get two empirical model-noise sources.

## Variables written (E1–E7 union)
`surface_temperature`, `TMP2m`, `Q2m`, `air_temperature_7`, `specific_total_water_7`, `LHTFLsfc`,
`SHTFLsfc`, `USWRFsfc`, `DSWRFsfc`, `PRATEsfc` — 6-hourly native (daily mean/max/min derived
downstream).

## Inputs
| | value |
|---|---|
| Checkpoint ACE2S-ERA5 | beaker `01KSVC6YS7C18SGYV4VPZYZ232` |
| Checkpoint ACE2S-CM4-piControl-RS0 | beaker `01KTYXNSJX90Y5E2CQ6SV8K37D` |
| Checkpoint ACE2S-CM4-piControl-RS1 | beaker `01KTWGH2VEZ4DNXXF1H5FTJK1S` |
| ERA5 IC + forcing | `/climate-default/2026-03-19-era5-1deg-8layer-1940-2025.zarr` (Weka) |
| CM4 IC + forcing | `/climate-default/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr` (Weka) |
| Output prefix (GCS) | `gs://vcm-ml-intermediate/2026-06-16-ace2s-land-feedback-inference/<run>` |

All checkpoints use `training_checkpoints/best_inference_ckpt.tar`. CM4 checkpoints are the
**uncoupled atmosphere-only** flavor (`fme.ace.inference`), trained on the single atmosphere-land
zarr above (not the merged coupled-SST dataset).

## Runs
All runs are launched from one script, `run-inference.sh`, which holds the common gantry settings
(clusters, image, budget, secrets) in a single `submit()` function and calls it once per run. Each
call validates its config then submits a single-GPU `fme.ace.inference` job.

| config | checkpoint | job name | experiments |
|---|---|---|---|
| `pilot-era5.yaml`         | ACE2S-ERA5 | `ace2s-lf-pilot-era5`  | pipeline validation |
| `frameworkA-era5.yaml`    | ACE2S-ERA5 | `ace2s-lf-fwA-era5`    | E1–E4, A-parts of E5/E6 |
| `frameworkA-cm4-rs0.yaml` | CM4-RS0    | `ace2s-lf-fwA-cm4-rs0` | E1–E4 (CM4), noise floor |
| `frameworkA-cm4-rs1.yaml` | CM4-RS1    | `ace2s-lf-fwA-cm4-rs1` | E1–E4 (CM4), noise floor |
| `frameworkB-era5.yaml`    | ACE2S-ERA5 | `ace2s-lf-fwB-era5`    | E5, E6 |
| `frameworkB-cm4-rs0.yaml` | CM4-RS0    | `ace2s-lf-fwB-cm4-rs0` | E5, E6 (CM4), noise floor |
| `frameworkB-cm4-rs1.yaml` | CM4-RS1    | `ace2s-lf-fwB-cm4-rs1` | E5, E6 (CM4), noise floor |

```bash
./run-inference.sh pilot-era5   # pilot first — confirm the GCS zarr ingests downstream...
./run-inference.sh              # ...then submit every run
# the optional argument is a substring filter, e.g. `./run-inference.sh frameworkB-cm4`
```

## Notes / things to confirm before production
- **Memory lever**: per-window cost ∝ `n_ics × n_ensemble_per_ic × forward_steps_in_memory`; all
  runs sit at ≤96 IC-steps/window. If a job OOMs, drop `forward_steps_in_memory` (A) or trim ICs (B).
- **CM4 record**: dataset spans `0151-01-01T06:00:00` → `0351-01-01T00:00:00` (200 yr, noleap;
  train ≤`0306`, val `0306`–`0311`, holdout/test `0311`–`0351`). **Framework A** free-runs the
  *whole* record (`n_forward_steps=291900` ≈ 199.9 yr from the first IC) — in-sample is fine for a
  long free-running climate characterization. **Framework B** uses the **holdout** (Jan-1 of
  `0311`–`0349`, 39 ICs; `0350` is dropped because a full 1460-step year from it runs 1 step past the
  data end) — cleaner out-of-sample for initialized seasonal skill.
- **ERA5 record**: 1940–2025; Framework A runs 80 yr from 1940. Note ERA5's holdout (1998–2010 +
  2020 ≈ 14 yr) is too small for robust seasonal skill, so `frameworkB-era5` uses 48 Jan-1 ICs across
  1977–2024 (accepts in-sample) — unlike CM4 B, which has a large holdout. Flag if you'd rather
  restrict ERA5 B to the 14 holdout years.
- **Long A runs**: add `--segments N` to the `fme.ace.inference` call (chained restarts) if you want
  checkpointing across the multi-hour CM4 rollout.
- The zarr `time` dim is **lead time**; `init_time` / `valid_time(sample, time)` are coordinates —
  downstream ingestion must build the calendar axis from `valid_time` per sample.
