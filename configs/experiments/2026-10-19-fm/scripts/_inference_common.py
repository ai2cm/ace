"""Shared constants and run discovery for the post-training inference jobs.

The inference families (generate_inference_configs.py, submit_inference_jobs.py)
evaluate a checkpoint of each finished 2026-10-19 training run, and of each
finished 2026-11-09 v2 run (--arch sfno-v2, swin-v2.2-v2). A run is one config in
run_configs/ (run_configs/ft/ for the ft archs, RUN_CONFIG_SUBDIRS); its job
name (and wandb name) comes from submit_fm_jobs.config_to_job_name. Which runs
have finished, and which Beaker result dataset holds their checkpoints, is
read from Beaker (_beaker_listing), not from wandb.
"""

import pathlib

import yaml
from _beaker_listing import OK, NamedExperiment, fetch_experiments_by_name
from submit_fm_jobs import (
    CONFIG_PREFIX,
    RUN_CONFIGS_DIR,
    WANDB_PREFIX,
    config_to_job_name,
)

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
# start 5 step 5); checkpoint_path swaps in the arch's last epoch
# (LAST_EMA_EPOCHS) for archs with a different schedule.
CHECKPOINTS = {
    "bestinf": (CHECKPOINT_PATH, "-bestinf"),
    "lastema": ("training_checkpoints/ema_ckpt_0150.tar", "-lastema"),
}
# Last EMA checkpoint epoch of archs whose runs do not train for 150 epochs:
# the v2 runs (sfno-v2, swin-v2.2-v2; create_fm_v2_jobs.py) train 50 epochs
# with an EMA checkpoint every epoch, so their last one is ema_ckpt_0050.tar.
LAST_EMA_EPOCHS = {"sfno-v2": 50, "swin-v2.2-v2": 50}
# Per-epoch EMA checkpoints, further --ckpt choices of the archs in
# EPOCH_CHECKPOINT_ARCHS (the ft archs, _add_ft_archs) for the families in
# EPOCH_CHECKPOINT_FAMILIES only: name -> (file, job-name suffix), e.g.
# "ema3": ("training_checkpoints/ema_ckpt_0003.tar", "-ema3"). The arch's last
# epoch is not among them: it is lastema.
EPOCH_CHECKPOINTS: dict[str, tuple[str, str]] = {}
EPOCH_CHECKPOINT_ARCHS: dict[str, tuple[str, ...]] = {}
EPOCH_CHECKPOINT_FAMILIES = ("eval",)

# The architectures of the inference families (submit_inference_jobs.py
# --arch): arch -> run-config token. The token must be followed by "-fm-" in
# the config name, so nc-sfno never selects an nc-sfno-v2 config. The runs of
# sfno and swin-v2.2 are their four-source training cells, balanced and
# unbalanced; the e3sm cells, the res variants (variants/, not globbed) and
# one-off diagnostics are left out. The runs of sfno-v2 and swin-v2.2-v2
# (wandb group ace2-fm-2026-11-09, create_fm_v2_jobs.py) are all four of their
# data arms (fm-era5, fm-balanced, fm-e3sm-balanced, fm-spencer-balanced) x 3
# seeds; "-fm-" also keeps nc-swin-v2.2 from selecting nc-swin-v2.2-v2.
ARCHS = {
    "sfno": "nc-sfno",
    "swin-v2.2": "nc-swin-v2.2",
    "sfno-v2": "nc-sfno-v2",
    "swin-v2.2-v2": "nc-swin-v2.2-v2",
}
DEFAULT_ARCH = "sfno"
# Subdirectory of run_configs/ holding the run configs of an arch whose
# configs are not in run_configs/ itself (the ft archs, _add_ft_archs).
RUN_CONFIG_SUBDIRS: dict[str, str] = {}
EXCLUDED_TOKENS = ("-e3sm", "-diag", "-smoke")
# Archs whose excluded tokens differ from EXCLUDED_TOKENS.
EXCLUDED_TOKENS_BY_ARCH = {
    "sfno-v2": ("-diag", "-smoke"),
    "swin-v2.2-v2": ("-diag", "-smoke"),
}
# Date of the training wandb group of each arch whose runs are not the
# 2026-10-19 ones; it replaces "2026-10-19" in the inference wandb group names
# (submit_inference_jobs.wandb_groups). For the follow-up archs below it also
# carries the phase tag, e.g. ace2-fm-eval-ext80-2026-11-11.
GROUP_DATES = {"sfno-v2": "2026-11-09", "swin-v2.2-v2": "2026-11-09"}
DEFAULT_GROUP_DATE = "2026-10-19"
# Config variant of the CM4/X-SHiELD suites (cm4xshield, cm4-1pct) per arch
# (generate_inference_configs.CM4_XSHIELD_VARIANTS). The v2 runs train with
# HGTsfc spherical-harmonic roundtripped on every cm4/xshield dataset, so
# their suites read those datasets the same way. The swin-v2.2-v2 runs share
# the sfno-v2 runs' data config (check_shared_sources), so they share the suite.
CM4_XSHIELD_VARIANT = {"sfno-v2": "v2", "swin-v2.2-v2": "v2"}


def _add_followup_archs() -> None:
    """The 2026-11-11 follow-ups of the v2 runs (create_fm_v2_followup_jobs.py)
    as archs of their own, with the v2 archs' 174 jobs per checkpoint:
    sfno-v2-ext80 and swin-v2.2-v2-ext80 (phase B, forks trained 80 more
    epochs, last EMA checkpoint ema_ckpt_0080.tar, groups tagged
    ext80-2026-11-11) and, for each chosen X in E120_RUNS,
    swin-v2.2-v2-lr{X}-e120 (phase C, 120 epochs, ema_ckpt_0120.tar, groups
    tagged lr{X}-e120-<date>). Their tokens are followed by "-fm-" too, so
    nc-swin-v2.2-v2-fm- never selects them and theirs never select a v2 run.
    Phase A (the LR sweep) gets no inference jobs.
    """
    from create_fm_v2_followup_jobs import E120_RUNS, e120_runs, ext80_runs

    runs = ext80_runs() + [run for lr in E120_RUNS for run in e120_runs(lr)]
    for run in runs:
        base = "sfno-v2" if run.arch == "nc-sfno-v2" else "swin-v2.2-v2"
        arch = f"{base}-{run.variant}"
        if arch in ARCHS:
            continue
        ARCHS[arch] = run.tag
        EXCLUDED_TOKENS_BY_ARCH[arch] = EXCLUDED_TOKENS_BY_ARCH[base]
        LAST_EMA_EPOCHS[arch] = run.max_epochs
        # The group's tag and date: ace2-fm-swin-ext80-2026-11-11 ->
        # ext80-2026-11-11.
        GROUP_DATES[arch] = run.group.split("-", 3)[3]
        CM4_XSHIELD_VARIANT[arch] = CM4_XSHIELD_VARIANT[base]


def _add_ft_archs() -> None:
    """The 2026-11-12 fine-tuning sweep of the ext80 fm-spencer-balanced runs
    (create_fm_v2_finetune_jobs.py) as archs of its own: sfno-v2-ft and
    swin-v2.2-v2-ft, 48 runs each (3 parents x 4 ERA5 mixes x 4 LRs). Their
    configs are in run_configs/ft/ and share the ext80 archs' token
    (nc-sfno-v2-ext80, nc-swin-v2.2-v2-ext80), so the two are told apart by
    directory (RUN_CONFIG_SUBDIRS): the ext80 archs read run_configs/ only.
    8 epochs with an EMA checkpoint every epoch: lastema is ema_ckpt_0008.tar,
    and the eval family alone also runs on ema_ckpt_0001.tar ..
    ema_ckpt_0007.tar (--ckpt ema1 .. ema7, EPOCH_CHECKPOINTS). Groups are
    tagged ft-2026-11-12. The models are their ext80 parents' (same data
    config, labels and normalization), so they use the v2 CM4/X-SHiELD suites.
    """
    from create_fm_v2_finetune_jobs import FT_DIR, ft_runs

    for run in ft_runs():
        base = "sfno-v2" if run.arch == "nc-sfno-v2" else "swin-v2.2-v2"
        arch = f"{base}-ft"
        if arch in ARCHS:
            continue
        ARCHS[arch] = run.parent.tag
        RUN_CONFIG_SUBDIRS[arch] = FT_DIR.name
        EXCLUDED_TOKENS_BY_ARCH[arch] = EXCLUDED_TOKENS_BY_ARCH[base]
        LAST_EMA_EPOCHS[arch] = run.max_epochs
        epochs = range(1, run.max_epochs)
        for epoch in epochs:
            EPOCH_CHECKPOINTS[f"ema{epoch}"] = (
                f"training_checkpoints/ema_ckpt_{epoch:04d}.tar",
                f"-ema{epoch}",
            )
        EPOCH_CHECKPOINT_ARCHS[arch] = tuple(f"ema{epoch}" for epoch in epochs)
        # ace2-fm-sfno-ft-2026-11-12 -> ft-2026-11-12.
        GROUP_DATES[arch] = run.group.split("-", 3)[3]
        CM4_XSHIELD_VARIANT[arch] = CM4_XSHIELD_VARIANT[base]


_add_followup_archs()
_add_ft_archs()
# Training labels a run needs for the CM4/X-SHiELD suites to be in-sample; runs
# without them (the v2 fm-era5 arms) get no cm4xshield / cm4-1pct jobs.
CM4_XSHIELD_LABELS = ("cm4", "xshield")

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


def training_config_names(subdir: str = "") -> list[str]:
    """Every training config filename in run_configs/ (all architectures), or
    in its subdirectory `subdir`.
    """
    return sorted(
        path.name
        for path in (RUN_CONFIGS_DIR / subdir).glob(f"{CONFIG_PREFIX}*.yaml")
        if not path.name.endswith(("-finetune.yaml", "-cooldown.yaml"))
    )


def inference_run_names(arch: str = DEFAULT_ARCH) -> list[str]:
    """Job names of the training runs of `arch` (ARCHS) that get inference
    jobs.
    """
    prefix = f"{CONFIG_PREFIX}{ARCHS[arch]}-fm-"
    excluded = EXCLUDED_TOKENS_BY_ARCH.get(arch, EXCLUDED_TOKENS)
    return [
        config_to_job_name(name)
        for name in training_config_names(RUN_CONFIG_SUBDIRS.get(arch, ""))
        if name.startswith(prefix) and not any(token in name for token in excluded)
    ]


def run_config_path(run: str) -> pathlib.Path:
    """Training config of the run named `run`: in run_configs/ or, if not
    there, in one of its RUN_CONFIG_SUBDIRS.
    """
    name = f"{CONFIG_PREFIX}{run.removeprefix(WANDB_PREFIX)}.yaml"
    for subdir in sorted(set(RUN_CONFIG_SUBDIRS.values())):
        if (RUN_CONFIGS_DIR / subdir / name).exists():
            return RUN_CONFIGS_DIR / subdir / name
    return RUN_CONFIGS_DIR / name


def run_train_labels(run: str) -> set[str]:
    """Labels of every train_loader dataset of `run`'s training config."""
    cfg = yaml.safe_load(run_config_path(run).read_text())
    return {
        label
        for dataset in cfg["train_loader"]["dataset"]["concat"]
        for label in dataset.get("labels", [])
    }


def checkpoint_path(ckpt: str, arch: str = DEFAULT_ARCH) -> str:
    """File in a training result dataset that --ckpt `ckpt` mounts for `arch`
    (CHECKPOINTS, with the last EMA epoch from LAST_EMA_EPOCHS, or
    EPOCH_CHECKPOINTS).
    """
    if ckpt in EPOCH_CHECKPOINTS:
        return EPOCH_CHECKPOINTS[ckpt][0]
    path = CHECKPOINTS[ckpt][0]
    if ckpt == "lastema" and arch in LAST_EMA_EPOCHS:
        path = f"training_checkpoints/ema_ckpt_{LAST_EMA_EPOCHS[arch]:04d}.tar"
    return path


def checkpoint_suffix(ckpt: str) -> str:
    """Job-name suffix of --ckpt `ckpt` (CHECKPOINTS or EPOCH_CHECKPOINTS)."""
    return (CHECKPOINTS | EPOCH_CHECKPOINTS)[ckpt][1]


def arch_checkpoints(arch: str, family: str | None = None) -> list[str]:
    """The --ckpt choices `arch` has jobs at, in `family` (default: in any
    family): CHECKPOINTS, then the arch's EPOCH_CHECKPOINT_ARCHS for the
    EPOCH_CHECKPOINT_FAMILIES.
    """
    ckpts = list(CHECKPOINTS)
    if family is None or family in EPOCH_CHECKPOINT_FAMILIES:
        ckpts += EPOCH_CHECKPOINT_ARCHS.get(arch, ())
    return ckpts


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
