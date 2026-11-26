"""Status tables for the post-training inference jobs of the six C96 AMIP-only
runs, separately of the six ERA5-1979 runs, and of the 2026-11-25 v3 grid
runs (its 24 5-source runs, then its 12 single-source runs).

The expected jobs are those submit_inference_jobs.py submits for every
inference run of the architecture (_inference_common.inference_run_names),
finished or not, with every supported family (--family; select_jobs;
Beaker names from beaker_job_name), at each checkpoint the architecture has
for the family (--ckpt, default all of them:
_inference_common.arch_checkpoints, i.e. bestinf and lastema and, for the eval
family of the ft archs, their per-epoch ones: ema1 .. ema7 for sfno-v2-ft and
swin-v2.2-v2-ft, ema2, ema4, ema8, ema16 for their -e32 long fine-tunes and
ema8, ema16, ema32, ema64 for the -e128 ones), one row per family and
checkpoint. By default, one table per cohort (COHORTS): the C96 AMIP-only
architectures, then the ERA5-1979 ones, then the v3 grid's 5-source and
single-source ones, at bestinf and the epoch-120 lastema checkpoint. Use
--arch for a single architecture. Beaker state comes from a listing per architecture;
no logs are read. As in fm_table.py, the newest experiment of each job name
counts, and preemption retries resolve to that experiment's latest status
(_beaker_listing.experiment_status).

"Queued" counts running experiments whose unfinished job has not started;
"Not submitted" counts job names with no Beaker experiment; "Failed / Other"
counts experiments that exist and failed, were canceled, or have no job.
Each of those is listed below the table with its experiment id.

Usage:
    python eval_table.py [--arch ARCH]
                         [--family FAMILY ...]
                         [--ckpt {bestinf,lastema,emaN} ...]
"""

import argparse
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import NamedTuple

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _agent_vm import (  # noqa: E402
    GANTRY_RUNNING,
    LOOKUP_THREADS,
    find_experiments,
    gantry_started,
    gantry_status,
    newest,
    on_agent_vm,
)
from _beaker_listing import OK, RUNNING  # noqa: E402
from _inference_common import (  # noqa: E402
    ARCHS,
    C96AMIP_ARCHS,
    CHECKPOINTS,
    EPOCH_CHECKPOINTS,
    ERA5_1979_ARCHS,
    V3_GRID_ARCHS,
    V3_SINGLE_SOURCE_ARCHS,
    arch_checkpoints,
    inference_run_names,
)
from fm_table import _print_table, _queued, newest_by_name  # noqa: E402
from generate_inference_configs import FAMILIES  # noqa: E402
from submit_fm_v3_jobs import read_record  # noqa: E402
from submit_inference_jobs import (  # noqa: E402
    EVAL_RECORD_PATH,
    agent_listing_groups,
    job_name,
    job_name_aliases,
    listing_text,
    select_jobs,
)

MISSING = "missing"
# Title -> archs of each separately tracked cohort (the default tables).
COHORTS = {
    "C96 AMIP-only SFNO v2 and Swin v2.2 v2": list(C96AMIP_ARCHS),
    "ERA5-1979 SFNO v2 and Swin v2.2 v2": list(ERA5_1979_ARCHS),
    "v3 grid 5-source SFNO v3 and Swin v2.2 v3": list(V3_GRID_ARCHS),
    "v3 grid single-source (C96 AMIP-only, ERA5-1979) SFNO v3 and Swin v2.2 v3": (
        list(V3_SINGLE_SOURCE_ARCHS)
    ),
}
COLUMNS = (
    "Family",
    "Checkpoint",
    "Finished",
    "Running",
    "Queued",
    "Failed / Other",
    "Not submitted",
    "Total expected",
)


class ExpectedJob(NamedTuple):
    family: str
    ckpt: str
    name: str  # Beaker job name on this machine (submit_inference_jobs.job_name)
    run: str
    wandb_name: str


class JobState(NamedTuple):
    status: str
    queued: bool
    id: str


def expected_jobs(
    arch: str, ckpts: list[str], families: list[str] | None = None
) -> list[ExpectedJob]:
    """Every job submit_inference_jobs.py submits for `arch` in `families`
    (default all) at each of `ckpts`, over all of its inference runs; a family
    has jobs only at its own checkpoints (arch_checkpoints).
    """
    runs = inference_run_names(arch)
    selected = FAMILIES if families is None else families
    return [
        ExpectedJob(family, ckpt, job_name(name), run, name)
        for ckpt in ckpts
        for family, run, name, _ in select_jobs(
            [f for f in selected if ckpt in arch_checkpoints(arch, f)],
            runs,
            ckpt,
            arch,
        )
    ]


def _counts(statuses: list[tuple[str, bool]]) -> tuple[int, ...]:
    """(finished, running, queued, failed / other, not submitted, total) of
    (status, queued) pairs.
    """
    return (
        sum(s == OK for s, _ in statuses),
        sum(s == RUNNING and not q for s, q in statuses),
        sum(s == RUNNING and q for s, q in statuses),
        sum(s not in (OK, RUNNING, MISSING) for s, _ in statuses),
        sum(s == MISSING for s, _ in statuses),
        len(statuses),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arch",
        choices=list(ARCHS),
        help="Architecture to count (default: one table per cohort, COHORTS).",
    )
    parser.add_argument(
        "--family",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
        help="Job families to count (default: all).",
    )
    parser.add_argument(
        "--ckpt",
        nargs="+",
        choices=[*CHECKPOINTS, *EPOCH_CHECKPOINTS],
        default=None,
        help="Checkpoints to count (default: all the architecture has).",
    )
    args = parser.parse_args()
    if args.arch is None:
        tables = {f"Inference jobs for {title}": a for title, a in COHORTS.items()}
    else:
        tables = {f"Inference jobs of {args.arch} ({ARCHS[args.arch]})": [args.arch]}
    for i, (title, arches) in enumerate(tables.items()):
        if i:
            print()
        _report(title, arches, args.family, args.ckpt)


def _report(
    title: str, arches: list[str], families: list[str], ckpt_args: list[str] | None
) -> None:
    """Print the status table of `arches`' jobs and list the failed ones."""
    ckpts = ckpt_args or list(
        dict.fromkeys(ckpt for arch in arches for ckpt in arch_checkpoints(arch))
    )
    by_arch = {arch: expected_jobs(arch, ckpts, families) for arch in arches}
    jobs = [job for arch_jobs in by_arch.values() for job in arch_jobs]
    states = _agent_states(jobs) if on_agent_vm() else _beaker_states(by_arch)
    status = {
        job.name: (states[job.name][:2] if job.name in states else (MISSING, False))
        for job in jobs
    }
    rows = []
    for ckpt in ckpts:
        for family in families:
            cell = [
                status[j.name] for j in jobs if j.family == family and j.ckpt == ckpt
            ]
            if cell:
                rows.append((family, ckpt, *_counts(cell)))
    rows.append(("Total", "", *_counts([status[j.name] for j in jobs])))

    print(title)
    print(f"As of {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")
    print()
    _print_table(COLUMNS, rows)
    failed = [
        (job.name, states[job.name].status, states[job.name].id)
        for job in jobs
        if status[job.name][0] not in (OK, RUNNING, MISSING)
    ]
    print()
    print(f"Failed ({len(failed)}):")
    for name, state, experiment_id in failed:
        print(f"  {name}  {experiment_id}  ({state})")


def _beaker_states(by_arch: dict[str, list[ExpectedJob]]) -> dict[str, JobState]:
    """Job name -> state of the newest experiment of each job of `by_arch`
    (arch -> its jobs), from one real-beaker listing per arch
    (fm_table.newest_by_name).
    """
    states = {}
    for arch, jobs in by_arch.items():
        names = {job.name for job in jobs}
        for name, named in newest_by_name(listing_text(arch), names).items():
            experiment_id = named.experiment.get("id", "")
            states[name] = JobState(named.status, _queued(named), experiment_id)
    return states


def _agent_states(jobs: list[ExpectedJob]) -> dict[str, JobState]:
    """Job name -> state of the newest experiment of each of `jobs`, under any
    of its job_name_aliases, seen from an agent VM (_agent_vm.find_experiments:
    gantry list, EVAL_RECORD_PATH, one beaker shim listing per run). A job the
    shims call running counts as queued until `gantry status` gives it a start
    time (a job gantry does not know stays running).
    """
    aliases = {job.name: job_name_aliases(job.wandb_name) for job in jobs}
    groups = agent_listing_groups((job.run, job.wandb_name) for job in jobs)
    recorded = {n: e["id"] for n, e in read_record(EVAL_RECORD_PATH).items()}
    found = find_experiments(groups, recorded)
    latest = {
        name: job
        for name, names in aliases.items()
        if (job := newest(j for alias in names for j in found.get(alias, ())))
        is not None
    }

    def started(job) -> bool:
        try:
            return gantry_started(job)
        except RuntimeError:
            return True

    running = [job for job in latest.values() if job.status in GANTRY_RUNNING]
    with ThreadPoolExecutor(LOOKUP_THREADS) as ex:
        not_started = {
            job.id for job, ok in zip(running, ex.map(started, running)) if not ok
        }
    states = {}
    for name, job in latest.items():
        status, queued = gantry_status(job.status)
        states[name] = JobState(status, queued or job.id in not_started, job.id)
    return states


if __name__ == "__main__":
    main()
