# 05 — Teacher registry + durable submission path (replace `run.sh`)

## Goal

Submitting a distillation run should not require editing a dated shell
script.  Teacher-specific facts live in one reviewable place; submission is
a thin, stable wrapper.

## Current state (verified)

`configs/experiments/2026-05-18-distillation-with-val/run.sh`:

- CLI: `./run.sh <dmd2|fdistill|scm> [--suffix <variant>] [--moe-teacher]`.
- Per-teacher facts are shell branches:
  - default: Beaker dataset `01KNM6H3JB1ZNS76HX17AAZRF7:checkpoints` →
    `/checkpoints/best_histogram_tail.ckpt`, val zarr
    `/climate-default/2026-04-29-distillation-teacher-val-dataset/conus_val_2023.zarr`,
    15 teacher steps.
  - `--moe-teacher`: dataset `01KTCHVDHY0SATWH9E0AW2PDS6` →
    `/checkpoints/bundled_moe_multivariate.ckpt`, val zarr
    `/climate-default/2026-06-09-distillation-teacher-moe-multivar-val-dataset/conus_multivar_val_2023.zarr`,
    18 steps, env flags `ACE_C_OUT=5 ACE_NOISE_DIST=loguniform
    ACE_SIGMA_MIN=0.005 ACE_SIGMA_MAX=200.0` (these flags disappear after
    spec 02).
- Infra: gantry on `ai2/titan`, 4 GPUs, image from
  `latest_distillation_image.txt` (repo root), weka mount
  `climate-default:/climate-default`, secrets `wandb-api-key-ai2cm-sa` and
  `google-credentials`, budget `ai2/atec-climate`,
  `pip install --no-deps .` at job start, `FASTGEN_OUTPUT_ROOT=/results`.
- Data YAMLs live next to it: `data-config.yaml` (train coarse, 2014–2022,
  lat −66..70) and `val-data-config.yaml`.

## Design

1. **Teacher registry** — once spec 03 lands, a "teacher" is fully described
   by a YAML fragment: checkpoint path + Beaker dataset ID + val zarr (+
   optional `teacher_num_steps` override).  Express each known teacher as a
   committed config fragment under
   `configs/distillation/teachers/{conus-single,conus-moe-multivar}.yaml`.
   The sigma/noise/channel parameters are **not** in the registry — they are
   derived from the checkpoint (spec 02).  Keep the table in
   `ARCHITECTURE.md` as documentation only.
2. **Run configs** — full `DistillationConfig` YAMLs under
   `configs/distillation/`, composing a teacher fragment with a method +
   data section.  Composition mechanism: simplest is full explicit YAMLs per
   (method × teacher) since there are ~8; do NOT build a templating system.
3. **Submission script** — one `submit.py` (or keep a much smaller
   `run.sh`) under `configs/distillation/` taking
   `--config <yaml> [--name-suffix s] [--dry-run]`, which:
   - reads the Beaker dataset ID + val mounts from the run config (add a
     `beaker:` section to the YAML schema — datasets to mount, cluster,
     gpus, priority — or keep infra in the script; decide with reviewer,
     but datasets must live with the teacher fragment since they are
     teacher-coupled),
   - derives `JOB_NAME`/wandb name the way `run.sh` does today
     (`ace-downscaling-distillation-<method>-...`),
   - shells out to `gantry run ... -- torchrun --nproc-per-node $NGPU -m
     fme.downscaling.distillation.train <config.yaml>`.
4. Delete `configs/experiments/2026-05-18-distillation-with-val/` after
   migrating its data YAMLs into the new run configs (keep git history; the
   directory is referenced only by its own README lines — verify with
   `grep -rn "2026-05-18-distillation" --include="*.md" --include="*.sh" .`).

## Constraints / gotchas

- `latest_distillation_image.txt` is the image-pinning mechanism shared
  with other workflows — keep reading it; do not invent a second pin.
- Spike-era job names are load-bearing for wandb continuity
  (`multivariate-downscaling` project, group conventions) — keep name
  format compatible or note the cut in the PR description.
- `--system-python --install "pip install --no-deps ."` exists because the
  image already holds all deps; preserve unless spec 01 changes the image
  layering (coordinate: after spec 01 the image has fastgen via the extra,
  and `--no-deps` still applies).
- gantry quoting: env flags must be separate `--env` args (the current
  script uses a bash array `TEACHER_ENV_FLAGS=(...)`; a Python submitter
  sidesteps this class of bug).

## Acceptance criteria

- `python configs/distillation/submit.py --config
  configs/distillation/fdistill-moe.yaml --dry-run` prints the full gantry
  command; without `--dry-run` it submits.
- Both teachers and at least fdistill + scm have committed run configs.
- No teacher-specific shell branching anywhere; adding a teacher = adding
  one YAML fragment + one run config.
- Old `run.sh` removed; README points at the new flow.
