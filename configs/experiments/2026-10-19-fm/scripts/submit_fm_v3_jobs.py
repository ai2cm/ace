"""Submit the 2026-11-18 FM v3 per-grid-cell normalization runs
(create_fm_v3_jobs.py configs) through the agent VM's gantry shim.

The shim (/usr/local/bin/gantry) sends each request to the Beaker proxy, which
runs the pushed commit of BRANCH (never the working tree), in workspace ai2/ace
and budget ai2/atec-climate. Each job runs, from the repo root,

    torchrun --nproc_per_node 4 -m fme.ace.train <run config>

as run-ace-train.sh does (4 GPUs, image latest_deps_only_image.txt, system
python with `pip install --no-deps .`, shared memory 200GiB, min runtime 4h),
with the wandb env of run-ace-train.sh (project FM, run name = job name, group
ace2-fm-sfno-v3-2026-11-18, or its -smoke group) and the wandb key from the
allowlisted secret. `--weka climate-default:/climate-default` makes the shim
mount every /climate-default/<dir> named in the config (each config's
WEKA_MOUNT header names the ERA5 store, which the datasets name only as
data_path + file_pattern). Clusters: CLUSTERS (the v2 clusters without
saturn). Priority: high for seed 0 (and the smoke test), normal for s1/s2.

The proxy names the Beaker experiment agent-alexeyy-<job name>-<suffix>. Each
submission is recorded in records/fm_v3_jobs.json (job name -> experiment
name, Beaker id, url, commit, priority, submission time), which fm_table.py
reads; a job already recorded is skipped unless --resubmit.

Usage:
    python submit_fm_v3_jobs.py [--dry-run] [--only NAME ...] [--smoke]
                                [--resubmit]

--only takes job names (ace2-fm-nc-sfno-v3-...) or their unique substrings
(e.g. c1-cellloss-s0). --smoke submits only the smoke-test config. --dry-run
prints each gantry command and the request the shim would send (gantry run
--dry-run), and submits nothing.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _submit_common import check_configs_at_head  # noqa: E402
from create_fm_v3_jobs import (  # noqa: E402
    N_GPUS,
    SEEDS,
    Run,
    describe,
    runs,
    smoke_run,
)

HERE = pathlib.Path(__file__).parent
EXP_DIR = HERE.parent
REPO_ROOT = EXP_DIR.parents[2]
RECORD_PATH = EXP_DIR / "records" / "fm_v3_jobs.json"

GANTRY = "/usr/local/bin/gantry"
BRANCH = "agent/alexeyy/exp/alexeyfm2"
EXPERIMENT_PREFIX = "agent-alexeyy-"
CLUSTERS = ("ai2/jupiter", "ai2/titan")
HIGH_PRIORITY = "high"
DEFAULT_PRIORITY = "normal"
SHARED_MEMORY = "200GiB"
MIN_RUNTIME = "4h"
WANDB_PROJECT = "FM"
# The wandb user the v2 runs are attributed to (run-ace-train.sh derives it
# from the beaker account, which the VM does not have).
WANDB_USERNAME = "yyexela"
WANDB_SECRET = "WANDB_API_KEY=wandb-api-key-ai2cm-sa"
WEKA = "climate-default:/climate-default"

SUBMITTED_RE = re.compile(
    r"^Submitted (?P<name>\S+)\s+id:\s+(?P<id>\S+)\s+url:\s+(?P<url>\S+)", re.M
)


def priority(run: Run) -> str:
    return HIGH_PRIORITY if run.smoke or run.seed == SEEDS[0] else DEFAULT_PRIORITY


def image() -> str:
    return (REPO_ROOT / "latest_deps_only_image.txt").read_text().strip()


def gantry_command(run: Run, dry_run: bool) -> list[str]:
    config = run.path.relative_to(REPO_ROOT)
    cmd = [GANTRY, "run", "--branch", BRANCH]
    cmd += ["--name", run.job_name, "--task-name", run.job_name]
    cmd += ["--description", describe(run)]
    cmd += ["--beaker-image", image(), "--priority", priority(run)]
    cmd += ["--min-runtime", MIN_RUNTIME]
    for cluster in CLUSTERS:
        cmd += ["--cluster", cluster]
    env = {
        "WANDB_USERNAME": WANDB_USERNAME,
        "WANDB_NAME": run.job_name,
        "WANDB_JOB_TYPE": "training",
        "WANDB_RUN_GROUP": run.group,
        "WANDB_PROJECT": WANDB_PROJECT,
    }
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    cmd += ["--env-secret", WANDB_SECRET]
    cmd += ["--gpus", str(N_GPUS), "--shared-memory", SHARED_MEMORY]
    cmd += ["--weka", WEKA]
    cmd += ["--system-python", "--install", "pip install --no-deps ."]
    # Jobs run the pushed commit; other agents' uncommitted edits in this
    # worktree do not reach them (check_configs_at_head covers the configs).
    cmd += ["--allow-dirty"]
    if dry_run:
        cmd += ["--dry-run"]
    cmd += ["--", "torchrun", "--nproc_per_node", str(N_GPUS)]
    cmd += ["-m", "fme.ace.train", str(config)]
    return cmd


def read_record() -> dict:
    if RECORD_PATH.exists():
        return json.loads(RECORD_PATH.read_text())
    return {}


def write_record(record: dict) -> None:
    RECORD_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def select(all_runs: list[Run], only: list[str] | None) -> list[Run]:
    if not only:
        return all_runs
    selected = []
    for token in only:
        matches = [r for r in all_runs if r.job_name == token]
        matches = matches or [r for r in all_runs if token in r.job_name]
        if len(matches) != 1:
            names = [r.job_name for r in matches]
            raise SystemExit(f"--only {token!r} matches {len(matches)} jobs: {names}")
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def head_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    parser.add_argument(
        "--smoke", action="store_true", help="Submit the smoke-test config only."
    )
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Submit even jobs already recorded in records/fm_v3_jobs.json.",
    )
    args = parser.parse_args()

    selected = select([smoke_run()] if args.smoke else runs(), args.only)
    record = read_record()
    pending = []
    for run in selected:
        if run.job_name in record and not args.resubmit:
            entry = record[run.job_name]
            print(f"Skipping (recorded as {entry['experiment']}): {run.job_name}")
        else:
            pending.append(run)
    if not args.dry_run:
        check_configs_at_head([run.path for run in pending])
    for run in pending:
        cmd = gantry_command(run, args.dry_run)
        print("Submitting:", subprocess.list2cmdline(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise SystemExit(f"gantry exited {proc.returncode} for {run.job_name}")
        if args.dry_run:
            continue
        match = SUBMITTED_RE.search(proc.stdout)
        if match is None:
            raise SystemExit(f"could not parse the gantry output for {run.job_name}")
        assert match["name"].startswith(EXPERIMENT_PREFIX + run.job_name), match
        record = read_record()
        record.setdefault(run.job_name, {})
        previous = record[run.job_name].get("experiment")
        record[run.job_name] = {
            "experiment": match["name"],
            "id": match["id"],
            "url": match["url"],
            "commit": head_commit(),
            "priority": priority(run),
            "submitted": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": str(run.path.relative_to(EXP_DIR)),
            **({"replaces": previous} if previous else {}),
        }
        write_record(record)
    if pending and not args.dry_run:
        print(f"Recorded in {RECORD_PATH.relative_to(REPO_ROOT)}; commit it.")


if __name__ == "__main__":
    main()
