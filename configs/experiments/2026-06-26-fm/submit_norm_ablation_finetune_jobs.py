"""Submit a gantry training job for each ERA5 fine-tuning config.

Each config produced by generate_norm_ablation_finetune_configs.py is submitted
via run-ace-train.sh, which validates the config and calls gantry.

Configs are checked to be committed at HEAD before anything is submitted:
gantry clones the repo from GitHub rather than uploading the working tree, so
an uncommitted config never reaches the job as written. The same applies to the
`fme/` change these configs depend on
(`parameter_init.override_labels_from_weights`) -- if it is not pushed, the
conditional cells fail at startup on a label-shape mismatch.

Usage:
    python submit_norm_ablation_finetune_jobs.py [--arm ARM]
                                        [--conditional | --no-conditional]
                                        [--dry-run]
                                        [--beaker-workspace WORKSPACE]
                                        [--beaker-cluster CLUSTER [CLUSTER ...]]
                                        [--beaker-priority PRIORITY]
"""

import argparse
import pathlib

from _submit_common import add_beaker_args, check_configs_at_head, submit_job
from generate_norm_ablation_configs import ARMS, CONFIG_PREFIX
from generate_norm_ablation_finetune_configs import (
    FINETUNE_SUFFIX,
    source_cells,
    source_config_name,
)

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_PROJECT = "FM"
#: A group of its own, so the ablation notebooks' existing group query keeps
#: returning only the base training runs.
WANDB_GROUP = "ace2-fm-norm-ablation-finetune-2026-06-26"


def config_to_job_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    return f"ace2-fm-{stem.removeprefix(CONFIG_PREFIX)}"


def finetune_config_name(arm: str, conditional: bool) -> str:
    stem = pathlib.Path(source_config_name(arm, conditional)).stem
    return f"{stem}{FINETUNE_SUFFIX}.yaml"


def selected_configs(args: argparse.Namespace) -> list[str]:
    names = []
    for arm, conditional in source_cells():
        if args.arm and arm != args.arm:
            continue
        if args.conditional is not None and conditional != args.conditional:
            continue
        names.append(finetune_config_name(arm, conditional))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), help="Only this grouping arm.")
    conditioning = parser.add_mutually_exclusive_group()
    conditioning.add_argument(
        "--conditional",
        dest="conditional",
        action="store_true",
        default=None,
        help="Only the module-conditioning cells.",
    )
    conditioning.add_argument(
        "--no-conditional",
        dest="conditional",
        action="store_false",
        help="Only the cells without module conditioning.",
    )
    add_beaker_args(
        parser,
        default_workspace="ai2/ace",
        default_cluster=["ai2/titan", "ai2/jupiter", "ai2/ceres"],
        default_priority="high",
    )
    args = parser.parse_args()

    config_filenames = selected_configs(args)
    config_paths = []
    for config_filename in config_filenames:
        config_path = RUN_CONFIGS_DIR / config_filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_filename} not found — run "
                "generate_norm_ablation_finetune_configs.py first"
            )
        config_paths.append(config_path)
    check_configs_at_head(config_paths)

    for config_filename in config_filenames:
        submit_job(
            RUN_SCRIPT,
            [
                f"{RUN_CONFIGS_DIRNAME}/{config_filename}",
                config_to_job_name(config_filename),
                WANDB_GROUP,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=HERE,
        )


if __name__ == "__main__":
    main()
