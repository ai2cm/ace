"""Submit SST-perturbation inference jobs for the FM checkpoints.

For each primary training run with a result dataset in
wandb_to_beaker_map.json, submits one free-running inference job per
(forcing grid, SST perturbation level) pair the run applies to: C96-trained
runs (nc-sfno-c96) only run on C96, ERA5-trained runs (nc-sfno-vN) only on
ERA5, and FM runs (nc-sfno-fm) on both (see generate_sst_configs.run_grids).
Each job mounts that run's best_inference_ckpt.tar at /ckpt.tar and runs the
matching run-agnostic config from run_configs/ (produced by
generate_sst_configs.py) via run-ace-inference.sh.

Usage:
    python submit_sst_jobs.py [--dry-run] [--run RUN [RUN ...]]
                              [--perturbation {p2k,p4k} ...]
                              [--forcing-grid {era5,c96} ...]
                              [--version {v1,v2}]
                              [--beaker-workspace WORKSPACE]
                              [--beaker-cluster CLUSTER [CLUSTER ...]]
                              [--beaker-priority PRIORITY]
"""

import argparse
import pathlib
import subprocess
import sys

from _submit_common import add_beaker_args, submit_job
from _version_select import add_version_arg
from generate_eval_configs import TRAINING_RESULT_DATASETS, WANDB_PROJECT
from generate_sst_configs import (
    DATASETS,
    RUN_CONFIGS_DIR,
    SST_PERTURBATIONS,
    sst_config_filename,
    sst_job_name,
    sst_runs,
)

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
RUN_SCRIPT = HERE / "run-ace-inference.sh"
WANDB_GROUP = "ace2-fm-sst-perts-2026-06-26"
# best_inference_ckpt.tar is always written by training; mounted at /ckpt.tar.
CHECKPOINT_PATH = "training_checkpoints/best_inference_ckpt.tar"


def validate_configs(config_filenames: list[str]) -> None:
    for config_filename in config_filenames:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fme.ace.validate_config",
                "--config_type",
                "inference",
                str(RUN_CONFIGS_DIR / config_filename),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        metavar="RUN",
        help="Restrict to these training run names (default: all primary runs).",
    )
    parser.add_argument(
        "--perturbation",
        nargs="+",
        default=None,
        choices=list(SST_PERTURBATIONS),
        help="Restrict to these SST perturbation levels (default: all).",
    )
    parser.add_argument(
        "--forcing-grid",
        nargs="+",
        default=None,
        choices=list(DATASETS),
        help=(
            "Restrict to these forcing grids (default: each run's native " "grid(s))."
        ),
    )
    add_beaker_args(
        parser,
        default_workspace="ai2/climate-titan",
        default_cluster=["ai2/titan"],
        default_priority="urgent",
    )
    args = parser.parse_args()

    levels = args.perturbation or list(SST_PERTURBATIONS)
    runs = sst_runs(args.version)
    if args.run is not None:
        unknown_runs = sorted(set(args.run) - set(runs))
        if unknown_runs:
            raise KeyError(
                f"unknown training run(s) {unknown_runs} — "
                f"available: {sorted(runs)}"
            )
        runs = {name: runs[name] for name in args.run}

    jobs: list[tuple[str, str, str, str]] = []
    for run_name, grids in sorted(runs.items()):
        if args.forcing_grid is not None:
            grids = tuple(grid for grid in grids if grid in args.forcing_grid)
        for grid in grids:
            for level in levels:
                jobs.append((run_name, grid, level, sst_config_filename(grid, level)))

    needed_configs = sorted({config_filename for *_, config_filename in jobs})
    for config_filename in needed_configs:
        if not (RUN_CONFIGS_DIR / config_filename).exists():
            raise FileNotFoundError(
                f"{config_filename} not found — run generate_sst_configs.py first"
            )

    if not args.dry_run:
        validate_configs(needed_configs)

    for run_name, grid, level, config_filename in jobs:
        submit_job(
            RUN_SCRIPT,
            [
                f"{RUN_CONFIGS_DIRNAME}/{config_filename}",
                sst_job_name(run_name, grid, level),
                WANDB_GROUP,
                TRAINING_RESULT_DATASETS[run_name],
                CHECKPOINT_PATH,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=HERE,
            extra_env={"SKIP_VALIDATE": "1"},
        )


if __name__ == "__main__":
    main()
