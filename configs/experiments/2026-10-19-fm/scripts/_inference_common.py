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
EXP_DIR = HERE.parent  # experiment root (configs, records); scripts live in scripts/
INFERENCE_CONFIGS_DIR = EXP_DIR / "inference_configs"
BEAKER_MAP_PATH = EXP_DIR / "records" / "wandb_to_beaker_map.json"

WANDB_ENTITY = "ai2cm"
WANDB_PROJECT = "FM"

# Mounted at /ckpt.tar by the run scripts.
CHECKPOINT_PATH = "training_checkpoints/best_inference_ckpt.tar"
MOUNTED_CHECKPOINT_PATH = "/ckpt.tar"

# Checkpoint choices of submit_inference_jobs.py --ckpt: (file in the training
# result dataset, job-name suffix). Both hold EMA weights as their stepper
# weights: with validate_using_ema the trainer saves best_inference_ckpt.tar and
# every ema_ckpt_XXXX.tar inside its EMA context (ckpt.tar instead holds the raw
# weights, with the EMA weights in a side entry). ema_ckpt_0150.tar is the EMA
# checkpoint of the last epoch (max_epochs 150, ema_checkpoint_save_epochs
# start 5 step 5).
CHECKPOINTS = {
    "bestinf": (CHECKPOINT_PATH, "-bestinf"),
    "lastema": ("training_checkpoints/ema_ckpt_0150.tar", "-lastema"),
}

# The architectures of the inference families (submit_inference_jobs.py
# --arch): arch -> run-config token. The runs of an arch are its four-source
# training cells, balanced and unbalanced; the e3sm cells, the res variants
# (variants/, not globbed) and one-off diagnostics are left out.
ARCHS = {
    "sfno": "nc-sfno",
    "swin-v2.2": "nc-swin-v2.2",
}
DEFAULT_ARCH = "sfno"
EXCLUDED_TOKENS = ("-e3sm", "-diag", "-smoke")

# The 2026-06-26 single/two-dataset baselines (wandb group
# ace2-fm-norm-ablation-2026-06-26) the notebooks compare against, per arch:
# run name -> (training result dataset, eval-suite config in
# inference_configs/). The configs are the suites of their old -bestinf jobs
# (fme eac5362, wandb group ace2-fm-eval-2026-09-28), copied unchanged. The
# swin-v2.2 notebook's baselines are the swin-v2.1 runs.
OLD_RUNS = {
    "sfno": {
        "ace2-fm-nc-sfno-era5-a1": (
            "01M0ZHY0XK03PKWZ5PVVRAGP7W",
            "ace-eval-suite-config-4deg-2026-06-26-old-nc-sfno-era5-a1.yaml",
        ),
        "ace2-fm-nc-sfno-fm-a2-cond": (
            "01M13B36FHZZJQX0A1HW5YJVYC",
            "ace-eval-suite-config-4deg-2026-06-26-old-nc-sfno-fm-a2-cond.yaml",
        ),
        "ace2-fm-nc-sfno-c96-a1": (
            "01M11FKTJ6H2HJ61S3HAWXKQ25",
            "ace-eval-suite-config-4deg-2026-06-26-old-nc-sfno-c96-a1.yaml",
        ),
    },
    "swin-v2.2": {
        "ace2-fm-nc-swin-v2.1-era5-a1": (
            "01M37Q293G4CEMYMP58RWJJN77",
            "ace-eval-suite-config-4deg-2026-06-26-old-nc-swin-v2.1-era5-a1.yaml",
        ),
        "ace2-fm-nc-swin-v2.1-fm-a2-cond": (
            "01M3B08B44195SG3A5RRXDKEF3",
            "ace-eval-suite-config-4deg-2026-06-26-old-nc-swin-v2.1-fm-a2-cond.yaml",
        ),
        "ace2-fm-nc-swin-v2.1-c96-a1": (
            "01M399A3NWG2V9SMA7D5SYHMYR",
            "ace-eval-suite-config-4deg-2026-06-26-old-nc-swin-v2.1-c96-a1.yaml",
        ),
    },
}


def training_config_names() -> list[str]:
    """Every training config filename in run_configs/ (all architectures)."""
    return sorted(
        path.name
        for path in RUN_CONFIGS_DIR.glob(f"{CONFIG_PREFIX}*.yaml")
        if not path.name.endswith(("-finetune.yaml", "-cooldown.yaml"))
    )


def inference_run_names(arch: str = DEFAULT_ARCH) -> list[str]:
    """Job names of the training runs of `arch` (ARCHS) that get inference
    jobs.
    """
    prefix = f"{CONFIG_PREFIX}{ARCHS[arch]}-"
    return [
        config_to_job_name(name)
        for name in training_config_names()
        if name.startswith(prefix)
        and not any(token in name for token in EXCLUDED_TOKENS)
    ]


def anneal_run_names() -> list[str]:
    """Job names of the anneal runs in run_configs/anneal/ (smoke tests left
    out), from the JOB_NAME line create_anneal_jobs.py writes in each header.
    """
    from create_anneal_jobs import ANNEAL_DIR, read_header

    return sorted(
        read_header(path)["JOB_NAME"]
        for path in ANNEAL_DIR.glob(f"{CONFIG_PREFIX}*.yaml")
        if not path.stem.endswith("-smoke")
    )


def all_training_run_names() -> list[str]:
    """Every training run of update_beaker_map.py: run_configs/ and the
    anneal runs. Inference families use inference_run_names, which has no
    anneal runs.
    """
    return [
        config_to_job_name(name) for name in training_config_names()
    ] + anneal_run_names()


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
