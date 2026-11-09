# 2026-10-19 FM experiments

Layout:

| Path | Contents |
| --- | --- |
| `scripts/` | Every generator, submitter, status and helper script (`.py`, `.sh`). Run them from anywhere: they resolve paths from their own location. |
| `base_configs/` | Hand-edited training base configs, one per architecture x data arm. |
| `run_configs/` | Generated training configs, one per submitted run (`create_fm*_jobs.py`); `anneal/`, `v2-smoke/`, `v3-smoke/` subsets. |
| `variants/` | One-off training configs outside the generated grid. |
| `inference_configs/` | Generated run-agnostic inference / eval-suite configs (`generate_inference_configs.py`). |
| `inference_sources/` | Inputs `generate_inference_configs.py` reads. |
| `records/` | Committed state written by scripts: `wandb_to_beaker_map.json` (`update_beaker_map.py`), `anneal_soups.json` (`make_soup_checkpoints.py`). |
| `docs/` | `PRs.md` (PRs merged into this branch). |
| `e3sm/` | E3SM dataset preparation scripts. |

## Scripts

Training:

- `create_fm_jobs.py` / `submit_fm_jobs.py`: v1 seed x masking grid from `base_configs/`.
- `create_fm_v2_jobs.py` / `submit_fm_v2_jobs.py`: v2 runs (wandb group ace2-fm-2026-11-05).
- `create_fm_v3_jobs.py` / `submit_fm_v3_jobs.py`: sfno-v3 runs, warm-started from v2 parents.
- `create_fm_v3_swin_jobs.py` / `submit_fm_v3_swin_jobs.py`: swin-v2.2-v3 runs.
- `create_anneal_jobs.py` / `submit_anneal_jobs.py`: anneal fine-tuning sweep.
- `run-ace-train.sh`, `run-ace-anneal.sh`: gantry wrappers the submitters call.

Inference and evaluation:

- `generate_inference_configs.py`, `submit_inference_jobs.py`, `run-ace-inference-job.sh`, `run_eval_suite.py`.
- Anneal post-processing: `run_anneal_postprocess.sh` (runs `update_beaker_map.py`, `make_soup_checkpoints.py`, `submit_anneal_eval_jobs.py`), `anneal_eval_status.py`.

Status and diagnostics:

- `fm_table.py`: per-architecture progress table (used by `/fm-job-watch`).
- `diagnose_dataset_properties.py`: `DatasetProperties.update` warning diagnosis.

Shared helpers: `_beaker_listing.py`, `_submit_common.py`, `_inference_common.py`, `_version_select.py`.

The gantry wrappers take config paths relative to the experiment root; the submit scripts run them with the experiment root as cwd.
