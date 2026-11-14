# 2026-10-19 FM experiments

Scripts resolve paths from their own location, so they can be run from anywhere.

## Directories

| Path | Contents |
| --- | --- |
| `base_configs/` | Hand-edited training parents, one per architecture x data arm. |
| `run_configs/` | Generated training configs, one per run (`create_fm_v2_jobs.py`). |
| `inference_configs/` | Generated inference / eval-suite configs (`generate_inference_configs.py`). |
| `inference_sources/` | Inputs read by `generate_inference_configs.py`. |
| `jeremy_configs/` | Reference training configs. |
| `records/` | Script-written state: `wandb_to_beaker_map.json`, `anneal_soups.json`. |
| `docs/PRs.md` | PRs merged into this branch. |
| `scripts/` | All scripts, listed below. |

## Scripts

Training:

- `create_fm_v2_jobs.py`, `submit_fm_v2_jobs.py`: v2 grid (wandb group ace2-fm-2026-11-09): sfno-v2, swin-v2.2-v2 x fm-era5, fm-balanced, fm-e3sm-balanced x 3 seeds.
- `create_fm_v2_followup_jobs.py`, `submit_fm_v2_followup_jobs.py`: 2026-11-11 follow-ups of the v2 maskoff runs, constant LR + 1-sqrt cooldown: `lrsweep` (swin fm-balanced s0, LR 1e-4/3e-4/5e-4, 30 epochs), `ext80` (sfno-v2 and swin-v2.2-v2 forked from each parent's ema_ckpt_0040, 80 more epochs), `e120 --lr X` (swin, 120 epochs from scratch at the chosen LR).
- `create_fm_1deg_jobs.py`, `submit_fm_1deg_jobs.py`: 1deg 6-hourly runs (wandb group ace2-fm-1deg-2026-11-13, 8 GPUs, 400GiB shared memory): sfno-v2 x fm-e3sm-balanced, fm-era5 x 3 seeds, 120 epochs. `--smoke` submits a 2-epoch smoke test per arm (group ace2-fm-1deg-2026-11-13-smoke).
- `create_fm_jobs.py`, `submit_fm_jobs.py`: v1 seed x masking grid.
- `create_anneal_jobs.py`, `submit_anneal_jobs.py`: anneal fine-tuning sweep.
- `fill_e3sm_group_stats.py`: fills the E3SM normalization group (`groups/e3sm-filled`).
- `run-ace-train.sh`, `run-ace-anneal.sh`: gantry wrappers called by the submitters.

Inference and evaluation:

- `generate_inference_configs.py`, `submit_inference_jobs.py`, `run-ace-inference-job.sh`, `run_eval_suite.py`.
- `eval_table.py`: status table of the inference jobs of one `--arch`, a row per family and checkpoint (`--family`, `--ckpt` to narrow).
- ft sweep evals (`--arch sfno-v2-ft`, `swin-v2.2-v2-ft`: the 96 runs of `run_configs/ft/`, 48 each). Run per arch, as often as wanted, while the runs finish: the `eval` family at bestinf, lastema (`ema_ckpt_0008`) and the per-epoch `ema1`..`ema7` (`ema_ckpt_0001`..`0007`) of every run whose training succeeded, skipping jobs already in Beaker (9 jobs per run, 432 per arch):
  `python scripts/submit_inference_jobs.py --arch sfno-v2-ft --family eval --ckpt all --skip-if-in-beaker`
  Later, the other families at bestinf and lastema (28 jobs per run, 1344 per arch):
  `python scripts/submit_inference_jobs.py --arch sfno-v2-ft --family cm4xshield cm4-1pct sst paper-som paper-era5 --ckpt bestinf lastema --skip-if-in-beaker`
  Status: `python scripts/eval_table.py --arch sfno-v2-ft [--family eval]`.
- `run_anneal_postprocess.sh`: runs `update_beaker_map.py`, `make_soup_checkpoints.py`, `submit_anneal_eval_jobs.py`.
- `anneal_eval_status.py`: anneal eval progress table.

Status and diagnostics:

- `fm_table.py`: per-architecture progress table (`/fm-job-watch`).
- `diagnose_dataset_properties.py`: diagnoses `DatasetProperties.update` warnings.

Shared helpers: `_beaker_listing.py`, `_submit_common.py`, `_inference_common.py`, `_version_select.py`.
