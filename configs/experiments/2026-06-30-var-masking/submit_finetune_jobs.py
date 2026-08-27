"""Submit a gantry training job for each multi-step fine-tuning config.

Each fine-tuning config in run_configs/ (from generate_finetune_configs.py) is
submitted via run-ace-train.sh, which validates it and calls gantry. All
fine-tunes are v5 (1-degree) and take the 400GiB footprint; GPU count is
per-cluster (titan/B200 = 4, jupiter/H100 = 8).

generate_finetune_configs.py writes one config per (cell, FT_VARIANTS entry),
so ``--variant`` selects which set to submit. It defaults to ``aimip`` rather
than ``all`` because the ``-mstepft`` (best_ckpt) runs were submitted first and
are long-lived: a resubmit writes a new /results and restarts fine-tuning at
epoch 0, discarding their progress. Pass ``--variant all`` deliberately.

Naming configs positionally submits exactly those and ignores ``--variant`` --
use it to resubmit a single cell (e.g. one that died) without touching the rest
of its variant set.

Usage:
    python submit_finetune_jobs.py [CONFIG ...]
                                   [--dry-run]
                                   [--variant {aimip,best,all}]
                                   [--beaker-workspace WORKSPACE]
                                   [--beaker-cluster CLUSTER [CLUSTER ...]]
                                   [--beaker-priority PRIORITY]
                                   [--cm-priority PRIORITY]
"""

import argparse
import pathlib
import subprocess

import beaker_submit
from generate_masking_configs import CONFIG_PREFIX, RUN_CONFIGS_DIR, WANDB_PREFIX

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-train.sh"
WANDB_PROJECT = "VarMasking8"
WANDB_GROUP = "ace2-var-masking-mstepft-2026-06-30"

# Config-stem suffix per --variant choice, matching FT_VARIANTS in
# generate_finetune_configs.py. "all" is spelled as a tuple of both so the glob
# below stays a simple suffix match.
VARIANT_SUFFIXES = {
    "aimip": ("-mstepftaimip",),
    "best": ("-mstepft",),
    "all": ("-mstepft", "-mstepftaimip"),
}

# All fine-tunes are v5 (1-degree); the run-ace-train.sh defaults (N_GPUS=2 /
# 100GiB) are for the 4-degree runs, so we override the footprint here.
#
# GPU count is per-cluster to avoid wasting the more powerful accelerators:
# jupiter (H100) uses the standard 8, but titan (B200) does the same work with 4.
# batch_size (8) is the *global* batch (local = batch_size // world_size), so 4
# vs 8 GPUs gives identical training and 8 stays divisible by both.
GPUS_PER_CLUSTER = {
    "ai2/titan": "4",  # B200
    "ai2/jupiter": "8",  # H100
}
DEFAULT_N_GPUS = "8"
V5_SHARED_MEMORY = "400GiB"


def n_gpus_for_clusters(clusters: list[str]) -> str:
    """GPU count for the requested clusters.

    A single beaker job requests a fixed GPU count and can then land on any of
    its allowed clusters, so mixing clusters whose standard counts differ (e.g.
    titan=4 and jupiter=8) is ambiguous -- reject it and ask for one at a time.
    """
    counts = {
        cluster: GPUS_PER_CLUSTER.get(cluster, DEFAULT_N_GPUS) for cluster in clusters
    }
    distinct = set(counts.values())
    if len(distinct) > 1:
        raise ValueError(
            "Requested clusters have different standard GPU counts "
            f"({counts}); a job requests a fixed GPU count and could land on "
            "either, so submit one cluster at a time "
            "(e.g. --beaker-cluster ai2/titan, then --beaker-cluster ai2/jupiter)."
        )
    return distinct.pop()


def config_to_job_name(config_filename: str) -> str:
    # ace-train-config-4deg-nc-sfno-era5-gmroff-mask0-seed1-v5-mstepft.yaml
    # -> ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5-mstepft
    suffix = pathlib.Path(config_filename).stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}"


def available_configs() -> list[str]:
    """Every fine-tune config in run_configs/, whatever the variant."""
    return sorted(
        path.name
        for path in RUN_CONFIGS_DIR.glob("*-mstepft*.yaml")
        if path.name.startswith(CONFIG_PREFIX)
    )


def select_configs(variant: str, named: list[str]) -> list[str]:
    """Config filenames to submit, either named explicitly or by variant.

    Explicitly named configs bypass the variant filter -- naming a config is
    already a deliberate choice, so it should not also have to match --variant.
    """
    present = available_configs()
    if named:
        wanted = [pathlib.Path(name).name for name in named]
        missing = [name for name in wanted if name not in present]
        if missing:
            raise FileNotFoundError(
                f"not in {RUN_CONFIGS_DIR}: {', '.join(missing)}\n"
                f"available:\n" + "\n".join(f"  {name}" for name in present)
            )
        return sorted(set(wanted))

    suffixes = VARIANT_SUFFIXES[variant]
    configs = [name for name in present if pathlib.Path(name).stem.endswith(suffixes)]
    if not configs:
        if present:
            raise FileNotFoundError(
                f"no {variant} fine-tune configs in {RUN_CONFIGS_DIR}, but "
                f"{len(present)} config(s) of another variant are present:\n"
                + "\n".join(f"  {name}" for name in present)
                + "\nPass --variant for one of those, or name a config "
                "positionally to submit it regardless of variant."
            )
        raise FileNotFoundError(
            f"no fine-tune configs at all in {RUN_CONFIGS_DIR} — run "
            "generate_finetune_configs.py first"
        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="*",
        metavar="CONFIG",
        help=(
            "Specific config filename(s) in run_configs/ to submit. Bypasses"
            " --variant. Default: every config of the selected variant."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_SUFFIXES),
        default="aimip",
        help=(
            "Which fine-tune variant to submit: aimip (best_inference_ckpt),"
            " best (best_ckpt), or all. Default: aimip -- submitting 'best'"
            " restarts those runs from epoch 0."
        ),
    )
    beaker_submit.add_arguments(parser)
    args = parser.parse_args()

    configs = select_configs(args.variant, args.configs)
    described = "named" if args.configs else args.variant
    print(f"Submitting {len(configs)} {described} fine-tune config(s).")

    n_gpus = n_gpus_for_clusters(args.beaker_cluster)
    base_env = beaker_submit.env(
        args,
        WANDB_PROJECT=WANDB_PROJECT,
        N_GPUS=n_gpus,
        BEAKER_SHARED_MEMORY=V5_SHARED_MEMORY,
    )
    print(f"Using N_GPUS={n_gpus} for cluster(s): {' '.join(args.beaker_cluster)}")
    for config_filename in configs:
        config_text = (RUN_CONFIGS_DIR / config_filename).read_text()
        if "REPLACE_WITH_BEAKER_DATASET_ID" in config_text:
            raise ValueError(
                f"{config_filename} still contains a placeholder dataset ID — "
                "run generate_finetune_configs.py with a real source map first."
            )
        job_name = config_to_job_name(config_filename)
        cmd = [str(RUN_SCRIPT), config_filename, job_name, WANDB_GROUP]
        print(f"Submitting ({WANDB_PROJECT}):", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=HERE, env=base_env)


if __name__ == "__main__":
    main()
