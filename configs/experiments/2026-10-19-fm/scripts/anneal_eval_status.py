"""Status of the anneal sweep's C/B eval jobs (J1, J3, J4) in Beaker.

One row per anneal run, one `ok/run/queue/fail` cell per checkpoint and
family, read from a single `beaker workspace experiments` listing (the same
canonical-name matching as the submit scripts; unshortened legacy names
count for their shortened twins). Lists failed and canceled jobs at the end.

    python anneal_eval_status.py [--failed-ids]
"""

import argparse

from _beaker_listing import (
    CANCELED,
    FAILED,
    NO_JOB,
    OK,
    RUNNING,
    NamedExperiment,
    fetch_experiments_by_name,
)
from _inference_common import anneal_run_names
from submit_anneal_eval_jobs import (
    J1_SUFFIX,
    J1_SUITE,
    eval_job_base,
    family_jobs,
    legacy_job_name,
)

FAMILIES = ("J1", "J3", "J4")
CKPTS = ("C", "B")
STATES = ("ok", "run", "queue", "fail", "miss")


def state(named: NamedExperiment | None) -> str:
    if named is None or named.status == NO_JOB:
        return "miss"
    if named.status == OK:
        return "ok"
    if named.status == RUNNING:
        jobs = named.experiment.get("jobs") or []
        started = any("started" in (j.get("status") or {}) for j in jobs[-1:])
        return "run" if started else "queue"
    if named.status in (FAILED, CANCELED):
        return "fail"
    return "miss"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="ai2/ace")
    parser.add_argument(
        "--failed-ids", action="store_true", help="Print experiment ids of failures."
    )
    args = parser.parse_args()
    experiments = fetch_experiments_by_name(args.workspace)
    runs = sorted(anneal_run_names(), key=lambda r: (r.split("-anneal-")[-1], r))
    totals = {(c, f): dict.fromkeys(STATES, 0) for c in CKPTS for f in FAMILIES}
    failed: list[tuple[str, str, str]] = []
    header = f"{'run':<22}" + "".join(
        f" {c+'-'+f:<14}" for c in CKPTS for f in FAMILIES
    )
    print(header)
    for run in runs:
        base = eval_job_base(run)
        cells = []
        for ckpt in CKPTS:
            for family in FAMILIES:
                counts = dict.fromkeys(STATES, 0)
                for name, _ in family_jobs(
                    f"{base}-ckpt{ckpt}", family, J1_SUITE, J1_SUFFIX
                ):
                    named = experiments.get(name)
                    legacy_name = legacy_job_name(name)
                    legacy = experiments.get(legacy_name) if legacy_name else None
                    if legacy is not None and (named is None or state(legacy) == "ok"):
                        named = legacy
                    s = state(named)
                    counts[s] += 1
                    totals[(ckpt, family)][s] += 1
                    if s == "fail" and named is not None:
                        failed.append((name, named.status, named.id))
                cells.append(
                    "/".join(str(counts[k]) for k in STATES[:4])
                    + (f"+{counts['miss']}m" if counts["miss"] else "")
                )
        short = base[len("ace2-fm-nc-sfno-anneal-") :].replace("-anneal-", "-")
        print(f"{short:<22}" + "".join(f" {c:<14}" for c in cells))
    print("\ncells: ok/running/queued/failed (+Nm = not in Beaker)")
    print("\ntotals:")
    grand = dict.fromkeys(STATES, 0)
    for (ckpt, family), counts in totals.items():
        for k in STATES:
            grand[k] += counts[k]
        print(f"  {ckpt} {family}: " + " ".join(f"{k}={counts[k]}" for k in STATES))
    print("  all : " + " ".join(f"{k}={grand[k]}" for k in STATES))
    if failed:
        print(f"\n{len(failed)} failed/canceled:")
        for name, status, eid in failed:
            print(f"  {status:<8} {eid}  {name}")


if __name__ == "__main__":
    main()
