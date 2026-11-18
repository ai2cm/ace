"""Submit the 2026-11-18 FM C96 AMIP-only runs and their ERA5-only
era5-1979 baseline (create_fm_c96amip_jobs.py configs) through the agent VM's
gantry shim.

The same mechanics as submit_fm_v3_jobs.py: the shim (/usr/local/bin/gantry)
sends each request to the Beaker proxy, which runs the pushed commit of BRANCH
(never the working tree), in workspace ai2/ace. Each job runs, from the repo
root,

    bash -c '<LAUNCH_SCRIPT>' fme-train <run config>

i.e. `torchrun --nproc_per_node 4 -m fme.ace.train <run config> --override
experiment_dir=$AGENT_OUTPUT_DIR` (4 GPUs, image latest_deps_only_image.txt,
system python with `pip install --no-deps .`, shared memory 200GiB, min
runtime 4h), with the wandb env of run-ace-train.sh (project FM, run name =
job name, group ace2-fm-c96amip-2026-11-18, or its -smoke group) and the wandb
key from the allowlisted secret. `--weka climate-default:/climate-default`
makes the shim mount every /climate-default/<dir> named in the config (each
config's WEKA_MOUNT header names the ERA5 store, which the datasets name only
as data_path + file_pattern; the stats live under the always-mounted
/climate-default/agents/alexeyy). Clusters: CLUSTERS (jupiter, titan and
saturn; --cluster overrides them). Priority: high for seed 0 (and the smoke
tests), normal for s1/s2.

Checkpoints and logs go to $AGENT_OUTPUT_DIR
(/climate-default/agents/alexeyy/<experiment name>), which survives Beaker's
automatic resume of a preempted job; the experiment name is the same for every
retry, so a retry finds ckpt.tar there and resumes (with its wandb run id).

Each submission is recorded in records/fm_c96amip_jobs.json (job name ->
experiment name, Beaker id, url, commit, priority, clusters, submission time,
and the experiment_dir / checkpoint paths under $AGENT_OUTPUT_DIR), which
fm_table.py reads; a job already recorded is skipped unless --resubmit.

Usage:
    python submit_fm_c96amip_jobs.py --arm ARM [ARM ...] [--dry-run]
                                     [--only NAME ...] [--smoke] [--resubmit]
                                     [--cluster CLUSTER ...]

--arm (required unless --smoke) picks the arms, fm-c96amip and/or
fm-era5-1979 (6 runs each). --only takes job names
(ace2-fm-nc-sfno-v2-fm-c96amip-s0-maskoff, ...) or their unique substrings
(e.g. sfno-v2-fm-c96amip-s1) within them. --smoke selects from the two (C96
AMIP) smoke tests instead. --dry-run prints each gantry command and
the request the shim would send (gantry run --dry-run), and submits nothing.
"""

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _submit_common import check_configs_at_head  # noqa: E402
from create_fm_c96amip_jobs import (  # noqa: E402
    ARMS,
    N_GPUS,
    SEEDS,
    Run,
    describe,
    runs,
    smoke_runs,
)
from submit_fm_v3_jobs import (  # noqa: E402
    BRANCH,
    DEFAULT_PRIORITY,
    EXPERIMENT_PREFIX,
    GANTRY,
    HIGH_PRIORITY,
    MIN_RUNTIME,
    SHARED_MEMORY,
    SUBMITTED_RE,
    WANDB_PROJECT,
    WANDB_SECRET,
    WANDB_USERNAME,
    WEKA,
    head_commit,
    image,
    output_paths,
)

HERE = pathlib.Path(__file__).parent
EXP_DIR = HERE.parent
REPO_ROOT = EXP_DIR.parents[2]
RECORD_PATH = EXP_DIR / "records" / "fm_c96amip_jobs.json"
CLUSTERS = ("ai2/jupiter", "ai2/titan", "ai2/saturn")
# $1 is the run config. Fails fast if the proxy did not set AGENT_OUTPUT_DIR.
LAUNCH_SCRIPT = (
    'set -eu; : "${AGENT_OUTPUT_DIR:?AGENT_OUTPUT_DIR is not set}"; '
    'mkdir -p "$AGENT_OUTPUT_DIR"; '
    f"exec torchrun --nproc_per_node {N_GPUS} -m fme.ace.train "
    '"$1" --override "experiment_dir=$AGENT_OUTPUT_DIR"'
)


def priority(run: Run) -> str:
    return HIGH_PRIORITY if run.smoke or run.seed == SEEDS[0] else DEFAULT_PRIORITY


def gantry_command(run: Run, clusters: tuple[str, ...], dry_run: bool) -> list[str]:
    config = run.path.relative_to(REPO_ROOT)
    cmd = [GANTRY, "run", "--branch", BRANCH]
    cmd += ["--name", run.job_name, "--task-name", run.job_name]
    cmd += ["--description", describe(run)]
    cmd += ["--beaker-image", image(), "--priority", priority(run)]
    cmd += ["--min-runtime", MIN_RUNTIME]
    for cluster in clusters:
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
    cmd += ["--", "bash", "-c", LAUNCH_SCRIPT, "fme-train", str(config)]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    parser.add_argument("--arm", nargs="+", choices=ARMS)
    parser.add_argument(
        "--smoke", action="store_true", help="Submit the smoke tests instead."
    )
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Submit even jobs already recorded in records/fm_c96amip_jobs.json.",
    )
    parser.add_argument(
        "--cluster",
        nargs="+",
        default=list(CLUSTERS),
        help=f"Beaker clusters (default: {' '.join(CLUSTERS)}).",
    )
    args = parser.parse_args()
    clusters = tuple(args.cluster)
    if not args.smoke and not args.arm:
        parser.error("--arm is required (unless --smoke)")

    candidates = smoke_runs() if args.smoke else runs(tuple(args.arm))
    selected = select(candidates, args.only)
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
        cmd = gantry_command(run, clusters, args.dry_run)
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
        previous = record.get(run.job_name, {}).get("experiment")
        record[run.job_name] = {
            "experiment": match["name"],
            "id": match["id"],
            "url": match["url"],
            "commit": head_commit(),
            "priority": priority(run),
            "clusters": list(clusters),
            "submitted": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": str(run.path.relative_to(EXP_DIR)),
            **output_paths(match["name"]),
            **({"replaces": previous} if previous else {}),
        }
        write_record(record)
    if pending and not args.dry_run:
        print(f"Recorded in {RECORD_PATH.relative_to(REPO_ROOT)}; commit it.")


if __name__ == "__main__":
    main()
