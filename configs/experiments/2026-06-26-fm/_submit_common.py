"""Shared CLI/submission plumbing for the submit_*_jobs.py scripts.

Each submit script discovers its own configs and builds its own gantry-job
argument list, but they share the same `--dry-run`/`--beaker-*` flags and the
same "print the command, build the env, subprocess.run it" submission step.
"""

import argparse
import os
import pathlib
import subprocess

#: Accepted values of the CM_PRIORITY label read by
#: scripts/beaker_balancer/balance.py. `immediate` is deliberately absent: Beaker
#: requires a human-supplied reason for it, so it is not something the balancer
#: may hand out.
CM_PRIORITIES = ("low", "normal", "high", "urgent")


def add_beaker_args(
    parser: argparse.ArgumentParser,
    *,
    default_workspace: str,
    default_cluster: list[str],
    default_priority: str,
) -> None:
    """Register --dry-run and the --beaker-* flags shared by all submit scripts."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--beaker-workspace",
        default=default_workspace,
        help=f"Beaker workspace to submit jobs to (default: {default_workspace}).",
    )
    parser.add_argument(
        "--beaker-cluster",
        nargs="+",
        default=default_cluster,
        metavar="CLUSTER",
        help=(
            "Beaker cluster(s) to target (default: " f"{' '.join(default_cluster)})."
        ),
    )
    parser.add_argument(
        "--beaker-priority",
        default=default_priority,
        help=f"Beaker job priority (default: {default_priority}).",
    )
    parser.add_argument(
        "--cm-priority",
        choices=CM_PRIORITIES,
        default=None,
        help=(
            "Opt the job in to scripts/beaker_balancer/balance.py at this rank. "
            "The balancer only modifies jobs which set it, and never raises the "
            "priority of a job that has already started, so submit at the "
            "--beaker-priority you want and let the balancer take it away "
            "(default: unset, leaving the job unmanaged)."
        ),
    )


def submit_job(
    run_script: pathlib.Path,
    job_args: list[str],
    *,
    wandb_project: str,
    args: argparse.Namespace,
    cwd: pathlib.Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Print and (unless --dry-run) submit a gantry job via `run_script`.

    `job_args` are the positional arguments passed to `run_script` (config
    path, job name, job group, and any script-specific trailing args).
    """
    cmd = [str(run_script), *job_args]
    print("Submitting:", " ".join(cmd))
    if args.dry_run:
        return
    env = {
        **os.environ,
        "WANDB_PROJECT": wandb_project,
        "BEAKER_WORKSPACE": args.beaker_workspace,
        "BEAKER_CLUSTER": " ".join(args.beaker_cluster),
        "BEAKER_PRIORITY": args.beaker_priority,
        # Empty rather than absent, so an exported CM_PRIORITY cannot opt a job
        # in behind the flag's back.
        "CM_PRIORITY": args.cm_priority or "",
        **(extra_env or {}),
    }
    subprocess.run(cmd, check=True, cwd=cwd, env=env)
