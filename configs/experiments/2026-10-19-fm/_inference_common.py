"""Shared constants and run discovery for the post-training inference jobs.

The inference families (generate_inference_configs.py, submit_inference_jobs.py)
evaluate the `best_inference_ckpt.tar` of each finished 2026-10-19 training
run. A run is one config in run_configs/; its job name (and wandb name) comes
from submit_fm_jobs.config_to_job_name. Which runs have finished, and which
Beaker result dataset holds their checkpoints, is read from Beaker
(_beaker_listing), not from wandb.
"""

import pathlib

from _beaker_listing import OK, NamedExperiment, fetch_experiments_by_name
from submit_fm_jobs import CONFIG_PREFIX, RUN_CONFIGS_DIR, config_to_job_name

HERE = pathlib.Path(__file__).parent
INFERENCE_CONFIGS_DIR = HERE / "inference_configs"
BEAKER_MAP_PATH = HERE / "wandb_to_beaker_map.json"

WANDB_ENTITY = "ai2cm"
WANDB_PROJECT = "FM"

# Mounted at /ckpt.tar by the run scripts.
CHECKPOINT_PATH = "training_checkpoints/best_inference_ckpt.tar"
MOUNTED_CHECKPOINT_PATH = "/ckpt.tar"

# The training runs these inference families cover: the nc-sfno cells of the
# four-source mix, balanced and unbalanced. The e3sm cells, the swin cells and
# one-off diagnostics are left out.
INFERENCE_ARCH = "nc-sfno"
EXCLUDED_TOKENS = ("-e3sm", "-diag", "-smoke")


def training_config_names() -> list[str]:
    """Every training config filename in run_configs/ (all architectures)."""
    return sorted(
        path.name
        for path in RUN_CONFIGS_DIR.glob(f"{CONFIG_PREFIX}*.yaml")
        if not path.name.endswith(("-finetune.yaml", "-cooldown.yaml"))
    )


def inference_run_names() -> list[str]:
    """Job names of the training runs that get inference jobs."""
    prefix = f"{CONFIG_PREFIX}{INFERENCE_ARCH}-"
    return [
        config_to_job_name(name)
        for name in training_config_names()
        if name.startswith(prefix)
        and not any(token in name for token in EXCLUDED_TOKENS)
    ]


def all_training_run_names() -> list[str]:
    return [config_to_job_name(name) for name in training_config_names()]


def finished_runs(
    run_names: list[str],
    experiments: dict[str, NamedExperiment] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Split `run_names` by Beaker state.

    Returns (run -> result dataset id) for runs whose experiment succeeded and
    has a result dataset, and (run -> reason) for the rest.
    """
    if experiments is None:
        experiments = fetch_experiments_by_name()
    ready: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for name in run_names:
        named = experiments.get(name)
        if named is None:
            skipped[name] = "no Beaker experiment"
        elif named.status != OK:
            skipped[name] = f"Beaker status {named.status} ({named.id})"
        elif named.result_dataset is None:
            skipped[name] = f"succeeded but no result dataset ({named.id})"
        else:
            ready[name] = named.result_dataset
    return ready, skipped
