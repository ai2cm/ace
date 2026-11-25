"""Submit the 2026-11-18 FM v3 per-grid-cell normalization runs
(create_fm_v3_jobs.py configs) through the agent VM's gantry shim.

The shim (/usr/local/bin/gantry) sends each request to the Beaker proxy, which
runs the pushed commit of BRANCH (never the working tree), in workspace ai2/ace
and budget ai2/atec-climate. Each job runs, from the repo root,

    bash -c '<LAUNCH_SCRIPT>' fme-train <run config>

i.e. `torchrun --nproc_per_node 4 -m fme.ace.train <run config> --override
experiment_dir=$AGENT_OUTPUT_DIR`, as run-ace-train.sh launches training
(4 GPUs, image latest_deps_only_image.txt, system
python with `pip install --no-deps .`, shared memory 200GiB, min runtime 4h),
with the wandb env of run-ace-train.sh (project FM, run name = job name, group
ace2-fm-sfno-v3-2026-11-18, or its -smoke group) and the wandb key from the
allowlisted secret. `--weka climate-default:/climate-default` makes the shim
mount every /climate-default/<dir> named in the config (each config's
WEKA_MOUNT header names the ERA5 store, which the datasets name only as
data_path + file_pattern). Clusters: CLUSTERS, the v2 clusters. saturn
(A100 80GB) was added back on 2026-11-18 for jobs still queued: the compiled
1deg model does not fit an A100's shared memory (see submit_fm_1deg_jobs.py),
but the 4deg SFNO-v2 ran on saturn (wandb hao37vth,
...-sfno-v2-fm-spencer-balanced-s2-maskoff, 50 epochs on 4x A100), and v3
differs from v2 only in normalization. Priority: high for seed 0 (and the
smoke test), normal for s1/s2.

Checkpoints and logs go to $AGENT_OUTPUT_DIR
(/climate-default/agents/alexeyy/<experiment name>), not the configs'
experiment_dir /results. The v2 runs (real gantry) resumed after preemption
from /results/training_checkpoints/ckpt.tar (their wandb output.log shows
"Resuming training from /results/training_checkpoints/ckpt.tar"), but under
the agent proxy only $AGENT_OUTPUT_DIR is guaranteed to survive Beaker's
automatic resume of a preempted job. The experiment name, and so
$AGENT_OUTPUT_DIR, is the same for every retry, so a retry finds ckpt.tar
there and resumes (fme resumes whenever experiment_dir holds one, and its
wandb run id too); the command is safe to restart. The config path stays a
separate argument so the shim still reads the config for its weka mounts.

The proxy names the Beaker experiment agent-alexeyy-<job name>-<suffix>. Each
submission is recorded in records/fm_v3_jobs.json (job name -> experiment
name, Beaker id, url, commit, priority, submission time, and the
experiment_dir / checkpoint paths under $AGENT_OUTPUT_DIR that downstream
evaluations need), which fm_table.py reads; a job already recorded is skipped
unless --resubmit.

Usage:
    python submit_fm_v3_jobs.py [--dry-run] [--only NAME ...] [--smoke]
                                [--resubmit] [--resume-from EXPERIMENT]
                                [--grid {v3,v3-grid-5src,v3-grid-single}]
                                [--stats-jobs] [--record PATH]
                                [--priority {low,normal,high}]

--grid picks the run list: `v3` (default) the 18 create_fm_v3_jobs.py runs;
`v3-grid-5src` the 21 5-source and `v3-grid-single` the 12 single-source
runs of create_fm_v3_grid_jobs.py (wandb group ace2-fm-v3-grid-2026-11-25;
same launch, GPUs, shared memory, clusters and priority policy: s0 high,
s1/s2 normal). The grids are recorded in records/fm_v3_grid_jobs.json, `v3`
in records/fm_v3_jobs.json, unless --record. --priority overrides the
priority of every selected job.

--stats-jobs submits the two per-grid-cell stats jobs the single-source grid
needs instead (create_fm_v3_grid_jobs.STATS_JOBS: get_pooled_stats.py with
scripts/data_process/configs/fm-v3-{c96amip,era5-1979}-percell-stats.yaml,
then verify_percell_stats.py, on one GPU node like the fm-c96amip scalar
stats jobs), writing to $AGENT_OUTPUT_DIR/<output name>. Each is recorded
(job name fm-v3-<source>-percell-stats) with its `stats_dir`, and the
`--single-source-stats` argument of create_fm_v3_grid_jobs.py is printed.

--only takes job names (ace2-fm-nc-sfno-v3-...) or their unique substrings
(e.g. c1-cellloss-s0). --smoke selects from the smoke tests (c2-cellloss,
the main one, and c3-cellloss) instead of the 18 runs. --dry-run
prints each gantry command and the request the shim would send (gantry run
--dry-run), and submits nothing.

--resume-from EXPERIMENT (an experiment name or its directory under
AGENT_OUTPUT_ROOT; one run only; implies --resubmit) continues a run that a
previous job of a different experiment name trained: a new experiment name
means a new, empty $AGENT_OUTPUT_DIR, where fme would start again from epoch
0 under a new wandb run. Before training, the job's wrapper copies the
source directory (training_checkpoints with ckpt.tar, best and EMA
checkpoints, wandb_run_id, metrics, logs) into its own $AGENT_OUTPUT_DIR,
reading the source only, and marks the copy complete with
$AGENT_OUTPUT_DIR/resume_copy_complete. fme then resumes from
$AGENT_OUTPUT_DIR/training_checkpoints/ckpt.tar and resumes the wandb run in
wandb_run_id, as after a preemption. A retry of the job finds the marker
and does not copy again (the copy finishes before training starts, so an
interrupted copy is redone in full). This is what fme's resume_results
does, but done in bash with a completion marker, since fme skips its copy
whenever training_checkpoints exists, even half-copied.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import create_fm_v3_grid_jobs as grid  # noqa: E402
from _submit_common import check_configs_at_head  # noqa: E402
from create_fm_v3_jobs import (  # noqa: E402
    N_GPUS,
    SEEDS,
    Run,
    describe,
    runs,
    smoke_runs,
)

HERE = pathlib.Path(__file__).parent
EXP_DIR = HERE.parent
REPO_ROOT = EXP_DIR.parents[2]
RECORD_PATH = EXP_DIR / "records" / "fm_v3_jobs.json"
AnyRun = Run | grid.Run
GRID_RECORD_PATH = EXP_DIR / "records" / "fm_v3_grid_jobs.json"
# --grid: (all runs, description of a run, default record). The describe
# functions take their own grid's runs, hence Any.
GRIDS: dict[
    str, tuple[Callable[[], Sequence[AnyRun]], Callable[[Any], str], pathlib.Path]
] = {
    "v3": (runs, describe, RECORD_PATH),
    "v3-grid-5src": (grid.multi_source_runs, grid.describe, GRID_RECORD_PATH),
    "v3-grid-single": (grid.single_source_runs, grid.describe, GRID_RECORD_PATH),
}
PRIORITIES = ("low", "normal", "high")

GANTRY = "/usr/local/bin/gantry"
BRANCH = "agent/alexeyy/exp/alexeyfm2"
EXPERIMENT_PREFIX = "agent-alexeyy-"
CLUSTERS = ("ai2/jupiter", "ai2/titan", "ai2/saturn")
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
AGENT_OUTPUT_ROOT = "/climate-default/agents/alexeyy"
RESUME_MARKER = "resume_copy_complete"
# $1 is the run config, $2 (optional) the experiment directory to resume from
# (see --resume-from). Fails fast if the proxy did not set AGENT_OUTPUT_DIR.
# The copy only reads $2 and writes only inside $AGENT_OUTPUT_DIR.
LAUNCH_SCRIPT = (
    'set -eu; : "${AGENT_OUTPUT_DIR:?AGENT_OUTPUT_DIR is not set}"; '
    'mkdir -p "$AGENT_OUTPUT_DIR"; '
    'src="${2:-}"; '
    f'marker="$AGENT_OUTPUT_DIR/{RESUME_MARKER}"; '
    'if [ -n "$src" ] && [ -e "$marker" ]; then '
    'echo "Resume copy from $(cat "$marker") already complete"; '
    'elif [ -n "$src" ]; then '
    'case "$src" in "$AGENT_OUTPUT_DIR"|"$AGENT_OUTPUT_DIR"/*) '
    'echo "resume source $src is inside AGENT_OUTPUT_DIR" >&2; exit 1;; esac; '
    "for f in training_checkpoints/ckpt.tar wandb_run_id; do "
    '[ -f "$src/$f" ] || { echo "missing $src/$f" >&2; exit 1; }; done; '
    'echo "Copying $src to $AGENT_OUTPUT_DIR to resume from it"; '
    # Default modes, not the source's: the source may be read-only.
    'cp -R --preserve=timestamps --no-preserve=mode "$src/." "$AGENT_OUTPUT_DIR/"; '
    'rm -f "$AGENT_OUTPUT_DIR"/training_checkpoints/.*.tmp; '
    'echo "$src" > "$marker"; '
    'echo "Copied; wandb_run_id $(cat "$AGENT_OUTPUT_DIR/wandb_run_id")"; '
    "fi; "
    f"exec torchrun --nproc_per_node {N_GPUS} -m fme.ace.train "
    '"$1" --override "experiment_dir=$AGENT_OUTPUT_DIR"'
)

# The per-cell stats jobs (--stats-jobs): $1 the output name under
# $AGENT_OUTPUT_DIR, $2 the stats config, $3 the training config the check
# compares names and grids against.
STATS_GPUS = 1
STATS_SHARED_MEMORY = "64GiB"
STATS_WORKERS = 32
STATS_LAUNCH_SCRIPT = (
    'set -euo pipefail; : "${AGENT_OUTPUT_DIR:?AGENT_OUTPUT_DIR is not set}"; '
    'OUT="$AGENT_OUTPUT_DIR/$1"; echo "OUTPUT_DIR=$OUT"; '
    'python scripts/data_process/get_pooled_stats.py compute "$2" "$OUT" '
    f"--workers {STATS_WORKERS}; "
    "python configs/experiments/2026-10-19-fm/scripts/verify_percell_stats.py "
    '"$OUT" "$3" "$2"'
)

SUBMITTED_RE = re.compile(
    r"^Submitted (?P<name>\S+)\s+id:\s+(?P<id>\S+)\s+url:\s+(?P<url>\S+)", re.M
)


def priority(run: AnyRun) -> str:
    return HIGH_PRIORITY if run.smoke or run.seed == SEEDS[0] else DEFAULT_PRIORITY


def image() -> str:
    return (REPO_ROOT / "latest_deps_only_image.txt").read_text().strip()


def gantry_command(
    run: AnyRun,
    dry_run: bool,
    resume_from: str | None = None,
    *,
    description: str,
    job_priority: str,
) -> list[str]:
    config = run.path.relative_to(REPO_ROOT)
    cmd = [GANTRY, "run", "--branch", BRANCH]
    cmd += ["--name", run.job_name, "--task-name", run.job_name]
    cmd += ["--description", description]
    cmd += ["--beaker-image", image(), "--priority", job_priority]
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
    cmd += ["--", "bash", "-c", LAUNCH_SCRIPT, "fme-train", str(config)]
    if resume_from is not None:
        cmd += [resume_from]
    return cmd


def stats_gantry_command(job: grid.StatsJob, dry_run: bool) -> list[str]:
    cmd = [GANTRY, "run", "--branch", BRANCH]
    cmd += ["--name", job.job_name, "--task-name", job.job_name]
    cmd += ["--description", job.description]
    cmd += ["--beaker-image", image(), "--priority", HIGH_PRIORITY]
    for cluster in CLUSTERS:
        cmd += ["--cluster", cluster]
    cmd += ["--gpus", str(STATS_GPUS), "--shared-memory", STATS_SHARED_MEMORY]
    cmd += ["--weka", WEKA]
    cmd += ["--system-python", "--install", "pip install --no-deps ."]
    cmd += ["--allow-dirty"]
    if dry_run:
        cmd += ["--dry-run"]
    cmd += ["--", "bash", "-c", STATS_LAUNCH_SCRIPT, "fm-v3-stats"]
    cmd += [
        job.output_name,
        str(job.stats_config.relative_to(REPO_ROOT)),
        str(job.train_config.relative_to(REPO_ROOT)),
    ]
    return cmd


def submit_stats_jobs(
    jobs: list[grid.StatsJob], dry_run: bool, resubmit: bool, record_path: pathlib.Path
) -> None:
    record = read_record(record_path)
    pending = []
    for job in jobs:
        if job.job_name in record and not resubmit:
            entry = record[job.job_name]
            print(f"Skipping (recorded as {entry['experiment']}): {job.job_name}")
        else:
            pending.append(job)
    if not dry_run:
        check_configs_at_head(
            [path for job in pending for path in (job.stats_config, job.train_config)]
        )
    for job in pending:
        cmd = stats_gantry_command(job, dry_run)
        print("Submitting:", subprocess.list2cmdline(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise SystemExit(f"gantry exited {proc.returncode} for {job.job_name}")
        if dry_run:
            continue
        match = SUBMITTED_RE.search(proc.stdout)
        if match is None:
            raise SystemExit(f"could not parse the gantry output for {job.job_name}")
        assert match["name"].startswith(EXPERIMENT_PREFIX + job.job_name), match
        record = read_record(record_path)
        record[job.job_name] = {
            "experiment": match["name"],
            "id": match["id"],
            "url": match["url"],
            "commit": head_commit(),
            "submitted": datetime.now(UTC).isoformat(timespec="seconds"),
            "stats_config": str(job.stats_config.relative_to(REPO_ROOT)),
            "stats_dir": f"{AGENT_OUTPUT_ROOT}/{match['name']}/{job.output_name}",
        }
        write_record(record, record_path)
    record = read_record(record_path)
    recorded = [
        f"{job.source}={record[job.job_name]['stats_dir']}"
        for job in jobs
        if job.job_name in record
    ]
    if recorded:
        print(
            "Once these jobs succeed: python create_fm_v3_grid_jobs.py "
            f"--single-source-stats {' '.join(recorded)}"
        )
    if pending and not dry_run:
        print(f"Recorded in {record_path.relative_to(REPO_ROOT)}; commit it.")


def resume_dir(experiment_or_dir: str) -> str:
    """The experiment directory to resume from, given its experiment name or
    its directory under AGENT_OUTPUT_ROOT.
    """
    name = experiment_or_dir.rstrip("/").removeprefix(AGENT_OUTPUT_ROOT + "/")
    if not name.startswith(EXPERIMENT_PREFIX) or "/" in name:
        raise SystemExit(
            f"--resume-from {experiment_or_dir!r}: expected an experiment name "
            f"{EXPERIMENT_PREFIX}... or {AGENT_OUTPUT_ROOT}/<experiment name>"
        )
    return f"{AGENT_OUTPUT_ROOT}/{name}"


def output_paths(experiment: str) -> dict[str, str]:
    """Where the job writes, from its experiment name (AGENT_OUTPUT_DIR)."""
    experiment_dir = f"{AGENT_OUTPUT_ROOT}/{experiment}"
    checkpoints = f"{experiment_dir}/training_checkpoints"
    return {
        "experiment_dir": experiment_dir,
        "latest_ckpt": f"{checkpoints}/ckpt.tar",
        "best_ckpt": f"{checkpoints}/best_ckpt.tar",
        "best_inference_ckpt": f"{checkpoints}/best_inference_ckpt.tar",
        "ema_ckpt_pattern": f"{checkpoints}/ema_ckpt_{{epoch:04d}}.tar",
    }


def read_record(path: pathlib.Path = RECORD_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def write_record(record: dict, path: pathlib.Path = RECORD_PATH) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


class _Named(Protocol):
    @property
    def job_name(self) -> str: ...


NamedT = TypeVar("NamedT", bound=_Named)


def select(all_runs: Sequence[NamedT], only: list[str] | None) -> list[NamedT]:
    if not only:
        return list(all_runs)
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
        "--smoke",
        action="store_true",
        help="Submit the smoke tests (c2-cellloss, c3-cellloss) instead.",
    )
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Submit even jobs already recorded in the record (see --record).",
    )
    parser.add_argument(
        "--grid",
        choices=list(GRIDS),
        default="v3",
        help="Run list: the v3 runs (default) or a create_fm_v3_grid_jobs.py grid.",
    )
    parser.add_argument(
        "--stats-jobs",
        action="store_true",
        help="Submit the single-source grid's per-cell stats jobs instead.",
    )
    parser.add_argument(
        "--record",
        type=pathlib.Path,
        help=(
            "Record file (default: records/fm_v3_jobs.json for --grid v3, "
            "records/fm_v3_grid_jobs.json for the grids and --stats-jobs)."
        ),
    )
    parser.add_argument(
        "--priority",
        choices=PRIORITIES,
        help="Beaker priority of every selected job (default: s0 high, else normal).",
    )
    parser.add_argument(
        "--resume-from",
        metavar="EXPERIMENT",
        help=(
            "Copy this experiment's directory into the new job's "
            "$AGENT_OUTPUT_DIR before training, to resume its checkpoint and "
            "wandb run. One run only; implies --resubmit."
        ),
    )
    args = parser.parse_args()
    all_runs, describe_run, default_record = GRIDS[args.grid]
    if args.stats_jobs:
        default_record = GRID_RECORD_PATH
    record_path = (args.record or default_record).resolve()
    if args.smoke and args.grid != "v3":
        parser.error("--smoke is only for --grid v3")
    if args.stats_jobs:
        if args.smoke or args.resume_from or args.grid != "v3" or args.priority:
            parser.error(
                "--stats-jobs takes only --dry-run, --only, --resubmit, --record"
            )
        submit_stats_jobs(
            select(grid.stats_jobs(), args.only),
            args.dry_run,
            args.resubmit,
            record_path,
        )
        return

    selected = select(smoke_runs() if args.smoke else all_runs(), args.only)
    missing = [run.path.name for run in selected if not run.path.exists()]
    if missing:
        raise SystemExit(
            f"Configs not generated (run create_fm_v3_grid_jobs.py): {missing}"
        )
    resume_from = None
    if args.resume_from is not None:
        if len(selected) != 1:
            raise SystemExit("--resume-from needs exactly one run (use --only)")
        resume_from = resume_dir(args.resume_from)
        args.resubmit = True
    record = read_record(record_path)
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
        job_priority = args.priority or priority(run)
        cmd = gantry_command(
            run,
            args.dry_run,
            resume_from,
            description=describe_run(run),
            job_priority=job_priority,
        )
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
        record = read_record(record_path)
        previous_entry = record.get(run.job_name, {})
        previous = previous_entry.get("experiment")
        # Every earlier job of this run, oldest first.
        superseded = list(previous_entry.get("superseded", []))
        if not superseded and "replaces" in previous_entry:
            superseded.append(
                {
                    "experiment": previous_entry["replaces"],
                    "id": previous_entry.get("replaces_id"),
                }
            )
        if previous:
            superseded.append({"experiment": previous, "id": previous_entry["id"]})
        record[run.job_name] = {
            "experiment": match["name"],
            "id": match["id"],
            "url": match["url"],
            "commit": head_commit(),
            "priority": job_priority,
            "submitted": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": str(run.path.relative_to(EXP_DIR)),
            **output_paths(match["name"]),
            **({"replaces": previous} if previous else {}),
            **({"replaces_id": previous_entry["id"]} if previous else {}),
            **({"superseded": superseded} if superseded else {}),
            **({"resumed_from": resume_from} if resume_from else {}),
        }
        write_record(record, record_path)
    if pending and not args.dry_run:
        print(f"Recorded in {record_path.relative_to(REPO_ROOT)}; commit it.")


if __name__ == "__main__":
    main()
