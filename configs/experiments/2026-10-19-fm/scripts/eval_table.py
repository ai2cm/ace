"""Status table of the post-training inference jobs of one architecture.

The expected jobs are those submit_inference_jobs.py submits for every
inference run of the architecture (_inference_common.inference_run_names),
finished or not, with every family (--family, default all; select_jobs;
Beaker names from beaker_job_name), at each checkpoint the architecture has
for the family (--ckpt, default all of them:
_inference_common.arch_checkpoints, i.e. bestinf and lastema and, for the eval
family of the ft archs, their per-epoch ones: ema1 .. ema7 for sfno-v2-ft and
swin-v2.2-v2-ft, ema2, ema4, ema8, ema16 for their -e32 long fine-tunes and
ema8, ema16, ema32, ema64 for the -e128 ones), one row per family and
checkpoint. Their Beaker
state comes from one workspace listing (submit_inference_jobs.listing_text);
no logs are read. As in fm_table.py, the newest experiment of each job name
counts, and preemption retries resolve to that experiment's latest status
(_beaker_listing.experiment_status).

"Queued" counts running experiments whose unfinished job has not started;
"Not submitted" counts job names with no Beaker experiment; "Failed / Other"
counts experiments that exist and failed, were canceled, or have no job.
Each of those is listed below the table with its experiment id.

Usage:
    python eval_table.py [--arch {sfno,swin-v2.2,sfno-v2,swin-v2.2-v2,
                                  sfno-v2-ext80,swin-v2.2-v2-ext80,
                                  sfno-v2-ft,swin-v2.2-v2-ft,
                                  sfno-v2-ft-e32,sfno-v2-ft-e128,
                                  swin-v2.2-v2-ft-e32,swin-v2.2-v2-ft-e128}]
                         [--family FAMILY ...]
                         [--ckpt {bestinf,lastema,emaN} ...]
"""

import argparse
import pathlib
import sys
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _beaker_listing import OK, RUNNING  # noqa: E402
from _inference_common import (  # noqa: E402
    ARCHS,
    CHECKPOINTS,
    EPOCH_CHECKPOINTS,
    arch_checkpoints,
    inference_run_names,
)
from fm_table import _print_table, _queued, newest_by_name  # noqa: E402
from generate_inference_configs import FAMILIES  # noqa: E402
from submit_inference_jobs import (  # noqa: E402
    beaker_job_name,
    listing_text,
    select_jobs,
)

DEFAULT_ARCH = "sfno-v2"
MISSING = "missing"
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


def expected_jobs(
    arch: str, ckpts: list[str], families: list[str] | None = None
) -> list[tuple[str, str, str]]:
    """(family, ckpt, Beaker job name) of every job submit_inference_jobs.py
    submits for `arch` in `families` (default all) at each of `ckpts`, over
    all of its inference runs; a family has jobs only at its own checkpoints
    (arch_checkpoints).
    """
    runs = inference_run_names(arch)
    selected = FAMILIES if families is None else families
    return [
        (family, ckpt, beaker_job_name(name))
        for ckpt in ckpts
        for family, _, name, _ in select_jobs(
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
        default=DEFAULT_ARCH,
        help=f"Architecture of the training runs (default: {DEFAULT_ARCH}).",
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
    ckpts = arch_checkpoints(args.arch) if args.ckpt is None else args.ckpt

    jobs = expected_jobs(args.arch, ckpts, args.family)
    by_name = newest_by_name(listing_text(args.arch))
    status = {
        name: (
            (by_name[name].status, _queued(by_name[name]))
            if name in by_name
            else (MISSING, False)
        )
        for _, _, name in jobs
    }
    rows = []
    for ckpt in ckpts:
        for family in args.family:
            cell = [status[n] for f, c, n in jobs if f == family and c == ckpt]
            if cell:
                rows.append((family, ckpt, *_counts(cell)))
    rows.append(("Total", "", *_counts([status[n] for _, _, n in jobs])))

    print(f"Inference jobs of {args.arch} ({ARCHS[args.arch]})")
    print(f"As of {datetime.now(UTC):%Y-%m-%d %H:%M} UTC")
    print()
    _print_table(COLUMNS, rows)
    failed = [
        (name, by_name[name].status, by_name[name].id)
        for _, _, name in jobs
        if status[name][0] not in (OK, RUNNING, MISSING)
    ]
    print()
    print(f"Failed ({len(failed)}):")
    for name, state, experiment_id in failed:
        print(f"  {name}  {experiment_id}  ({state})")


if __name__ == "__main__":
    main()
