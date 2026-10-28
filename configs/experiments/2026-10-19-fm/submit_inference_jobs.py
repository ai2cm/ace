"""Submit inference jobs on the best_inference_ckpt.tar of the 2026-10-19 runs.

Runs: the nc-sfno four-source training runs (_inference_common.
inference_run_names; no e3sm, no swin) whose Beaker experiment succeeded.
Runs still training are listed as skipped; re-run with --skip-if-in-beaker
once they finish to submit only what is missing. Each job mounts the run's
training_checkpoints/best_inference_ckpt.tar at /ckpt.tar and runs a
run-agnostic config from inference_configs/ (generate_inference_configs.py)
through run-ace-inference-job.sh.

Families (--family, default all):

  eval        {run}-bestinf                     evaluator suite, 2026-06-26 windows
  cm4xshield  {run}-cm4xshield-bestinf          evaluator suite, CM4/AM4/X-SHiELD
  cm4-1pct    {run}-cm4-1pct-bestinf            evaluator suite, CM4 1pctCO2 140 yr
  sst         {run}-sst-{grid}-{level}          inference, 2 grids x 3 levels
  paper-som   {run}-{kind}                      evaluator, 2 SOM slab kinds
  paper-era5  {run}-{kind}                      evaluator, 4 ERA5 kinds

Gantry clones the repository at HEAD, so the configs must be committed and
pushed before a real submission; this is checked unless --dry-run is given.
Every config a submission uses is validated against this checkout's fme
first (--dry-run validates too, so a dry run proves the configs load).

Usage:
    python submit_inference_jobs.py [--family FAMILY ...] [--run RUN ...]
                                    [--kind KIND ...] [--sst-grid GRID ...]
                                    [--sst-level LEVEL ...] [--job-name NAME ...]
                                    [--use-map] [--dry-run]
                                    [--skip-if-in-beaker] [--exclude-job NAME ...]
                                    [--beaker-workspace WORKSPACE]
                                    [--beaker-cluster CLUSTER [CLUSTER ...]]
                                    [--beaker-priority PRIORITY]
"""

import argparse
import json
import pathlib
import subprocess
import sys
from typing import NamedTuple

from _beaker_listing import fetch_experiments_by_name
from _inference_common import (
    BEAKER_MAP_PATH,
    CHECKPOINT_PATH,
    INFERENCE_CONFIGS_DIR,
    WANDB_PROJECT,
    finished_runs,
    inference_run_names,
)
from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from generate_inference_configs import (
    EVAL_SUITE_CONFIGS,
    FAMILIES,
    PAPER_KINDS,
    SST_GRIDS,
    SST_PERTURBATIONS,
    paper_config_filename,
    sst_config_filename,
)
from submit_fm_jobs import add_default_beaker_args

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-inference-job.sh"
CONFIGS_DIRNAME = INFERENCE_CONFIGS_DIR.name

WANDB_GROUPS = {
    "eval": "ace2-fm-eval-2026-10-19",
    "cm4xshield": "ace2-fm-cm4xshield-eval-2026-10-19",
    "cm4-1pct": "ace2-fm-cm4xshield-eval-2026-10-19",
    "sst": "ace2-fm-sst-perts-2026-10-19",
    "paper-som": "ace2-fm-paper-2026-10-19",
    "paper-era5": "ace2-fm-paper-2026-10-19",
}
# run-ace-inference-job.sh MODE per family.
MODES = {
    "eval": "suite",
    "cm4xshield": "suite",
    "cm4-1pct": "suite",
    "sst": "inference",
    "paper-som": "evaluator",
    "paper-era5": "evaluator",
}
EVAL_SUITE_NAME_SUFFIXES = {
    "eval": "-bestinf",
    "cm4xshield": "-cm4xshield-bestinf",
    "cm4-1pct": "-cm4-1pct-bestinf",
}


class Job(NamedTuple):
    name: str
    family: str
    config: str  # filename in inference_configs/
    dataset_id: str


def family_jobs(
    family: str,
    run: str,
    kinds: list[str] | None = None,
    sst_grids: list[str] | None = None,
    sst_levels: list[str] | None = None,
) -> list[tuple[str, str]]:
    """(job name, config filename) of one run's jobs in one family, narrowed
    by `kinds` (paper families) and `sst_grids`/`sst_levels` (sst).
    """
    if family in EVAL_SUITE_CONFIGS:
        return [
            (f"{run}{EVAL_SUITE_NAME_SUFFIXES[family]}", EVAL_SUITE_CONFIGS[family])
        ]
    if family == "sst":
        return [
            (f"{run}-sst-{grid}-{level}", sst_config_filename(grid, level))
            for grid in SST_GRIDS
            if sst_grids is None or grid in sst_grids
            for level in SST_PERTURBATIONS
            if sst_levels is None or level in sst_levels
        ]
    if family in ("paper-som", "paper-era5"):
        return [
            (f"{run}-{kind}", paper_config_filename(kind))
            for kind, (kind_family, _) in PAPER_KINDS.items()
            if kind_family == family and (kinds is None or kind in kinds)
        ]
    raise ValueError(f"unknown family {family!r}")


def resolve_runs(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, str]]:
    runs = inference_run_names()
    if args.run is not None:
        unknown = sorted(set(args.run) - set(runs))
        if unknown:
            raise SystemExit(f"unknown run(s) {unknown}; available: {runs}")
        runs = [r for r in runs if r in args.run]
    if args.use_map:
        mapping = json.loads(BEAKER_MAP_PATH.read_text())
        ready = {r: mapping[r] for r in runs if r in mapping}
        skipped = {
            r: f"not in {BEAKER_MAP_PATH.name}" for r in runs if r not in mapping
        }
        return ready, skipped
    print(f"Listing experiments in {args.beaker_workspace}...")
    experiments = fetch_experiments_by_name(args.beaker_workspace)
    return finished_runs(runs, experiments)


def validate_configs(families_configs: list[tuple[str, str]]) -> None:
    """Validate each (family, config filename) against this checkout's fme."""
    for family, config in families_configs:
        path = str(INFERENCE_CONFIGS_DIR / config)
        mode = MODES[family]
        if mode == "suite":
            cmd = [
                sys.executable,
                str(HERE / "run_eval_suite.py"),
                "--validate-only",
                path,
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "fme.ace.validate_config",
                "--config_type",
                mode,
                path,
            ]
        subprocess.run(cmd, check=True)
        print(f"Validated ({mode}): {config}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--family",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
        help="Job families to submit (default: all).",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        metavar="RUN",
        help="Restrict to these training run names (default: all finished).",
    )
    parser.add_argument(
        "--use-map",
        action="store_true",
        help=(
            f"Take result datasets from {BEAKER_MAP_PATH.name} "
            "(update_beaker_map.py) instead of a live Beaker listing."
        ),
    )
    parser.add_argument(
        "--kind",
        nargs="+",
        choices=list(PAPER_KINDS),
        default=None,
        help="paper-som/paper-era5: only these kinds (default: all of the family).",
    )
    parser.add_argument(
        "--sst-grid",
        nargs="+",
        choices=list(SST_GRIDS),
        default=None,
        help="sst: only these forcing grids (default: all).",
    )
    parser.add_argument(
        "--sst-level",
        nargs="+",
        choices=list(SST_PERTURBATIONS),
        default=None,
        help="sst: only these perturbation levels (default: all).",
    )
    parser.add_argument(
        "--job-name",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only these exact job names (applied after the other filters).",
    )
    # Same Beaker defaults as the training submitter (submit_fm_jobs.py).
    add_default_beaker_args(parser)
    args = parser.parse_args()

    ready, skipped = resolve_runs(args)
    print(f"\n{len(ready)} run(s) with a succeeded training job:")
    for run, dataset in sorted(ready.items()):
        print(f"  {run}  {dataset}")
    if skipped:
        print(f"{len(skipped)} run(s) skipped:")
        for run, reason in sorted(skipped.items()):
            print(f"  {run}  ({reason})")
    print()

    jobs = [
        Job(name, family, config, ready[run])
        for family in args.family
        for run in sorted(ready)
        for name, config in family_jobs(
            family, run, args.kind, args.sst_grid, args.sst_level
        )
    ]
    if args.job_name is not None:
        unknown = sorted(set(args.job_name) - {job.name for job in jobs})
        if unknown:
            raise SystemExit(f"--job-name not among the selected jobs: {unknown}")
        jobs = [job for job in jobs if job.name in args.job_name]
    jobs = drop_jobs_in_beaker(jobs, lambda job: job.name, args)
    if not jobs:
        print("Nothing to submit.")
        return

    needed = sorted({(job.family, job.config) for job in jobs})
    for _, config in needed:
        if not (INFERENCE_CONFIGS_DIR / config).exists():
            raise FileNotFoundError(
                f"{config} not found - run generate_inference_configs.py first"
            )
    if not args.dry_run:
        check_configs_at_head(
            [INFERENCE_CONFIGS_DIR / config for config in {c for _, c in needed}]
        )
    validate_configs(needed)

    for job in jobs:
        submit_job(
            RUN_SCRIPT,
            [
                MODES[job.family],
                f"{CONFIGS_DIRNAME}/{job.config}",
                job.name,
                WANDB_GROUPS[job.family],
                job.dataset_id,
                CHECKPOINT_PATH,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=HERE,
            extra_env={"SKIP_VALIDATE": "1"},
        )

    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.family] = counts.get(job.family, 0) + 1
    verb = "would be submitted" if args.dry_run else "submitted"
    summary = ", ".join(f"{family}: {n}" for family, n in counts.items())
    print(f"\n{len(jobs)} job(s) {verb} ({summary})")
    longest = max(jobs, key=lambda job: len(job.name))
    print(f"Longest job name ({len(longest.name)} chars): {longest.name}")


if __name__ == "__main__":
    main()
