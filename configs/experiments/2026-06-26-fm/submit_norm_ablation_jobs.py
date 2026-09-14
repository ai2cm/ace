"""Submit a gantry training job for each per-dataset normalization ablation config.

Each config produced by generate_norm_ablation_configs.py is submitted via
run-ace-train.sh, which validates the config and calls gantry.

Unlike submit_fm_jobs.py, this does not filter by version tag: the ablation
configs are named by regime and arm, not by version.

Unfiltered, the generator writes every cell of the arch x regime x arm x
conditioning x masking cross, which is more than is ever queued at once, so the
filters below are the normal way to use this script rather than an occasional
convenience. They are the generator's own filters (select_cells), with one
different default: --masking here takes the unmasked cells, matching the runs
the ablation started with. Because the generator may itself have been run with
filters, and then wrote only the cells it was asked for, the default
all-architectures selection here can name cells that were never written; pass
--arch (and --arm and the rest as needed) to match what was generated.

Usage:
    python submit_norm_ablation_jobs.py [--arch ARCH] [--regime REGIME]
                                        [--arm ARM] [--masking MASKING]
                                        [--conditional | --no-conditional]
                                        [--include-degenerate]
                                        [--dry-run]
                                        [--beaker-workspace WORKSPACE]
                                        [--beaker-cluster CLUSTER [CLUSTER ...]]
                                        [--beaker-priority PRIORITY]
"""

import argparse
import pathlib

from _submit_common import add_beaker_args, submit_job
from generate_norm_ablation_configs import (
    CONFIG_PREFIX,
    UNMASKED,
    add_cell_filter_args,
    config_name,
    select_cells,
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
    """Filenames of the cells matching the filters, in a stable order."""
    return [
        config_name(*cell)
        for cell in select_cells(
            arch=args.arch,
            regime=args.regime,
            arm=args.arm,
            conditional=args.conditional,
            masking=args.masking,
            include_degenerate=args.include_degenerate,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cell_filter_args(parser, masking_default=UNMASKED)
    parser.add_argument(
        "--include-degenerate",
        action="store_true",
        help=(
            "Also submit the cells which reduce to their regime's A1 control. "
            "Requires generate_norm_ablation_configs.py --include-degenerate, "
            "and is only meaningful once their seeds have been changed."
        ),
    )
    add_beaker_args(
        parser,
        default_workspace="ai2/ace",
        default_cluster=["ai2/titan", "ai2/jupiter", "ai2/ceres"],
        default_priority="high",
    )
    args = parser.parse_args()

    config_filenames = selected_configs(args)
    # Every config is checked before the first submission: a missing one
    # partway down the list would otherwise leave the earlier cells submitted
    # and the rest not, which is not a state worth being in.
    missing = [
        name for name in config_filenames if not (RUN_CONFIGS_DIR / name).exists()
    ]
    if missing:
        generate = "generate_norm_ablation_configs.py"
        if args.include_degenerate:
            generate += " --include-degenerate"
        raise SystemExit(
            "\n".join(
                [f"{len(missing)} selected configs were never generated:"]
                + [f"  {name}" for name in missing]
                + [f"Run {generate} first, or narrow the selection with --arch."]
            )
        )

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
