"""Submit a gantry training job for each per-dataset normalization ablation config.

Each config produced by generate_norm_ablation_configs.py is submitted via
run-ace-train.sh, which validates the config and calls gantry.

Unlike submit_fm_jobs.py, this does not filter by architecture (the ablation
spans both nc-sfno and nc-swin-v2) or by version tag (the ablation configs are
named by regime and arm, not by version).

Usage:
    python submit_norm_ablation_jobs.py [--arch ARCH] [--regime REGIME]
                                        [--arm ARM]
                                        [--conditional | --no-conditional]
                                        [--include-degenerate]
                                        [--dry-run]
                                        [--beaker-workspace WORKSPACE]
                                        [--beaker-cluster CLUSTER [CLUSTER ...]]
                                        [--beaker-priority PRIORITY]
                                        [--cm-priority PRIORITY]
"""

import argparse
import pathlib

from _submit_common import add_beaker_args, submit_job
from generate_norm_ablation_configs import (
    ARCH_SOURCES,
    ARMS,
    CONFIG_PREFIX,
    REGIME_SOURCES,
    all_cells,
    config_name,
    degenerate_reason,
)

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-train.sh"

WANDB_PROJECT = "FM"
WANDB_GROUP = "ace2-fm-norm-ablation-2026-06-26"


def config_to_job_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    return f"ace2-fm-{stem.removeprefix(CONFIG_PREFIX)}"


def selected_configs(args: argparse.Namespace) -> list[str]:
    """Every non-degenerate cell matching the filters, in a stable order."""
    names = []
    for arch, regime, arm, conditional in all_cells():
        if args.arch and arch != args.arch:
            continue
        if args.regime and regime != args.regime:
            continue
        if args.arm and arm != args.arm:
            continue
        if args.conditional is not None and conditional != args.conditional:
            continue
        if (
            not args.include_degenerate
            and degenerate_reason(regime, arm, conditional) is not None
        ):
            continue
        names.append(config_name(arch, regime, arm, conditional))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=sorted(ARCH_SOURCES), help="Only this arch.")
    parser.add_argument(
        "--regime", choices=sorted(REGIME_SOURCES), help="Only this data regime."
    )
    parser.add_argument("--arm", choices=sorted(ARMS), help="Only this grouping arm.")
    parser.add_argument(
        "--include-degenerate",
        action="store_true",
        help=(
            "Also submit the cells which reduce to their regime's A1 control. "
            "Requires generate_norm_ablation_configs.py --include-degenerate, "
            "and is only meaningful once their seeds have been changed."
        ),
    )
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

    for config_filename in selected_configs(args):
        config_path = RUN_CONFIGS_DIR / config_filename
        if not config_path.exists():
            generate = "generate_norm_ablation_configs.py"
            if args.include_degenerate:
                generate += " --include-degenerate"
            raise FileNotFoundError(
                f"{config_filename} not found — run {generate} first"
            )
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
