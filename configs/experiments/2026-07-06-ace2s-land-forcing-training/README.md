# ACE2S land-forcing training (land–feedback follow-up)

Training runs that test whether giving ACE **persistent land-surface state** as *forcing*
input improves its near-surface temperature/moisture predictions. The land–atmosphere
coupling evaluations (explore2 `brianh/2026-06-09-land-atm-coupling/reports/ace-eval`, which
consumed the `2026-06-16-ace2s-land-feedback-inference` runs) suggested ACE's lack of
snow-covered-area and soil-moisture information limits its near-surface responses in some
regions. Here we make those fields available as forcing variables (prescribed inputs, not
predicted) and retrain, isolating each variable in a separate model.

## Design

- **Control**: the existing deployed ACE2S checkpoints (no extra land forcing). We do **not**
  retrain a control — the dynamics-affecting code changes since those checkpoints were trained
  are all default-off, so they remain a valid baseline (see the plan notes / commit history).
- **Treatments** (one added forcing variable each, to isolate effects), one seed each:
  | model | dataset | added forcing variable(s) |
  |---|---|---|
  | `era5-snow` | ERA5  | `surface_snow_area_fraction` |
  | `era5-soil` | ERA5  | `soil_moisture_0..3` |
  | `cm4-snow`  | CM4   | `surface_snow_area_fraction` |
  | `cm4-soil`  | CM4   | `total_moisture_content_of_soil_layer_0..3` |
- **Two-stage recipe** per model, matching the controls: 1-step pretrain → multi-step
  finetune (warm-started from the pretrain checkpoint). Adding an input channel changes the
  encoder input dims, so each model trains **from scratch** (no warm-start from the control).
- Snow-covered area (`surface_snow_area_fraction`) is identically named/defined in both
  datasets → the cleanest, cross-dataset-comparable experiment. Soil moisture is **not**
  cross-dataset consistent (ERA5 = volumetric fraction on ERA5 layers; CM4 = mass content on
  coarsened CM4 layers), so the soil result is per-dataset, not apples-to-apples across the two.

## How the forcing variable is added

A variable is *forcing* iff it is in `stepper.step.config.in_names` but not `out_names`
(the loader auto-requires it from the dataset; it is fed as input each step and excluded from
prediction/loss). Each pretrain config differs from its control recipe only by the added
name(s) in `in_names` — plus, for CM4 soil, an `input_masking` block (below).

**Spatial-NaN handling (resolved by inspecting the zarrs — see "verified dataset facts").**
Only **CM4 soil moisture** needs masking; everything else is defined everywhere:

- **CM4 soil** (`total_moisture_content_of_soil_layer_0..3`) is NaN over non-soil cells (~69%).
  The CM4 dataset ships time-invariant per-variable masks
  (`mask_total_moisture_content_of_soil_layer_N`, 1 over valid soil / 0 elsewhere), which the
  loader turns into a `SpatialMaskProvider` (`fme/core/dataset/xarray.py`). `cm4-soil` therefore
  sets `stepper.input_masking: {mask_value: 0, fill_value: 0.0}` — the blessed static-mask
  method (same as ocean models). The soil NaN region matches `mask==0` exactly, so this fills
  precisely the NaN cells; only variables with a matching `mask_*` are touched (atmospheric
  inputs have none, and the dataset has no generic `mask_2d`/`mask_N`, so there is no `_N`
  level-pattern collision).
- **ERA5 snow & soil, and CM4 snow** have **no NaN** (verified), so those three models need no
  masking and no `fill_nans_on_normalize` — they differ from the control only by `in_names`.

The **finetune** configs are variable-agnostic: they load `in_names`, `input_masking`, etc. from
the mounted pretrain checkpoint, so one finetune config per dataset serves both its snow and
soil variants.

## Verified dataset facts (inspected 2026-07-06)

| dataset | `mask_*` present? | `surface_snow_area_fraction` | soil moisture |
|---|---|---|---|
| ERA5 | none | 0–1, no NaN | `soil_moisture_0..3`: no NaN |
| CM4  | yes, incl. `mask_total_moisture_content_of_soil_layer_0..3` | **0–100 (percent)**, no NaN | `total_moisture_content_of_soil_layer_0..3`: ~69% NaN, NaN region == `mask==0` |

Note the snow units differ across datasets (CM4 percent vs ERA5 fraction); per-dataset
normalization absorbs the scale, but the *snow* effect is not directly comparable across
datasets in raw units. `column_soil_moisture` (CM4) has no NaN if a single-field alternative is
preferred later.

## Base recipes (ported)

- ERA5 pretrain/finetune ← `feature/add-ACE2S-ERA5-baseline:configs/baselines/era5/`
  (`ace-train-config-1-step-pretrain.yaml`, `ace-train-config-multi-step-finetuning.yaml`).
  Ported to modern schema (`validation_loader:` → `validation.loader:`).
- CM4 pretrain/finetune ← `exp/ace2s-cm4-piControl-train:configs/experiments/2026-06-05-ace2s-cm4-picontrol/`.

## Inputs

| | value |
|---|---|
| ERA5 train/forcing data | `/climate-default/2026-03-19-era5-1deg-8layer-1940-2025.zarr` (Weka) |
| CM4 train/forcing data | `/climate-default/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr` (Weka) |
| ERA5 stats (network norm) | beaker `andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019` → `/statsdata` |
| CM4 stats (network norm) | beaker `jamesd/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-stats` → `/statsdata` |
| Control checkpoints | ERA5 `01KSVC6YS7C18SGYV4VPZYZ232`; CM4 `01KTYXNSJX90Y5E2CQ6SV8K37D` (rs0), `01KTWGH2VEZ4DNXXF1H5FTJK1S` (rs1) |

Stats datasets are mounted via each config's `# arg:` header; the launcher extracts them.

## Prerequisites to verify BEFORE launching

1. **Normalization stats contain the new variables — VERIFIED (2026-07-06).** The network
   normalizer reads `centering.nc` (mean) and `scaling-full-field.nc` (std) from the mounted
   stats dataset (the `# arg:` header). Both files were confirmed to include the added variables:
   - ERA5 stats `andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019`: `surface_snow_area_fraction`
     and `soil_moisture_0..3` present.
   - CM4 stats `jamesd/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-stats`:
     `surface_snow_area_fraction` and `total_moisture_content_of_soil_layer_0..3` present.
2. **Confirm early-epoch loss is finite (non-NaN)** on a short run before scaling up (spot-checks
   the CM4-soil masking end-to-end).

The dataset masking/NaN facts and the stats presence check are both resolved; the loss-finiteness
check is a routine launch-time sanity check.

## Launch order

```bash
./run-ace-train.sh            # submit the 4 pretrain jobs (start with a snow model)
# when the pretrain jobs finish, paste their beaker dataset IDs into run-ace-train.sh
# (PRETRAIN_DATASETS_*), uncomment the finetune block, then:
./run-ace-train.sh <name>     # e.g. cm4-snow-finetune
```

Each job runs `torchrun --nproc_per_node 4 -m fme.ace.train <config>` on `ai2/titan`,
logging to W&B (project `ace`, group `ace2s-land-forcing`).

## Evaluation

After a model finetunes, run inference (reuse the `2026-06-16-ace2s-land-feedback-inference`
recipe, adding the same forcing variable to that config's stepper/forcing) and compare
near-surface T/Q skill against the corresponding control checkpoint.
